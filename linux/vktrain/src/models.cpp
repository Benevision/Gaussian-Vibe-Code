#include "models.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>

#include "nlohmann/json.hpp"
#include "vk.h"

namespace fs = std::filesystem;
using nlohmann::json;

namespace vkt {

// ---------------------------------------------------------------- optimisers

void HostAdam::step(std::vector<float>& p, const std::vector<double>& g, double lr, double eps, double b1, double b2) {
    if (m.size() != p.size()) { m.assign(p.size(), 0.0); v.assign(p.size(), 0.0); }
    steps++;
    double bc1 = 1 - std::pow(b1, double(steps)), bc2 = 1 - std::pow(b2, double(steps));
    for (size_t i = 0; i < p.size(); i++) {
        m[i] = b1 * m[i] + (1 - b1) * g[i];
        v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
        p[i] = float(p[i] - lr / bc1 * m[i] / (std::sqrt(v[i]) / std::sqrt(bc2) + eps));
    }
}

// ---------------------------------------------------------------- appearance

void AppearanceModel::setup(const std::vector<const CameraInfo*>& cams, double lr_, double reg_) {
    lr = lr_;
    reg = reg_;
    std::vector<double> exps;
    for (auto* c : cams) if (c->exposure > 0) exps.push_back(c->exposure);
    std::sort(exps.begin(), exps.end());
    double ref = exps.empty() ? 0.0 : exps[exps.size() / 2];
    p.clear(); init.clear(); opt.clear();
    for (auto* c : cams) {
        double gain = 1.0;
        if (ref > 0 && c->exposure > 0) gain = std::min(3.0, std::max(1.0 / 3.0, std::pow(c->exposure / ref, 1 / 2.2)));
        std::vector<float> p0(12, 0.0f);
        p0[0] = p0[4] = p0[8] = float(gain);
        p.push_back(p0);
        init.push_back(p0);
        opt.emplace_back();
    }
}

void AppearanceModel::step(size_t i, const std::vector<float>& grad) {
    std::vector<double> g(12);
    for (int k = 0; k < 12; k++) {
        if (!std::isfinite(grad[size_t(k)])) return;
        g[size_t(k)] = double(grad[size_t(k)]) + reg * (double(p[i][size_t(k)]) - init[i][size_t(k)]);
    }
    opt[i].step(p[i], g, lr, 1e-8);
}

// ---------------------------------------------------------------- bilateral grid

std::vector<float> BilateralGridModel::identity() {
    std::vector<float> g(size_t(12) * 8 * 16 * 16, 0.0f);
    for (int r = 0; r < 3; r++) std::fill_n(&g[size_t(r * 4 + r) * 2048], 2048, 1.0f);
    return g;
}

void BilateralGridModel::setup(size_t views, double lr_, double tvWeight) {
    lr = lr_;
    tv = tvWeight;
    grids.assign(views, identity());
    opt.assign(views, HostAdam());
}

namespace {
// Grid element (k, z, y, x) → index; dims D = 8, H = 16, W = 16.
inline size_t gi(int k, int z, int y, int x) { return ((size_t(k) * 8 + size_t(z)) * 16 + size_t(y)) * 16 + size_t(x); }
}  // namespace

double BilateralGridModel::tvLoss(size_t i) const {
    const std::vector<float>& g = grids[i];
    double total = 0;
    const int dims[3] = {8, 16, 16};
    for (int d = 0; d < 3; d++) {
        double s = 0;
        size_t cnt = 0;
        for (int k = 0; k < 12; k++)
            for (int z = 0; z < 8; z++)
                for (int y = 0; y < 16; y++)
                    for (int x = 0; x < 16; x++) {
                        int c[3] = {z, y, x};
                        if (c[d] + 1 >= dims[d]) continue;
                        c[d]++;
                        double df = double(g[gi(k, c[0], c[1], c[2])]) - g[gi(k, z, y, x)];
                        s += df * df;
                        cnt++;
                    }
        total += s / double(cnt);
    }
    return tv * total;
}

void BilateralGridModel::addTvGrad(size_t i, std::vector<double>& grad) const {
    const std::vector<float>& g = grids[i];
    const int dims[3] = {8, 16, 16};
    for (int d = 0; d < 3; d++) {
        double cnt = 12.0 * 8 * 16 * 16 / dims[d] * (dims[d] - 1);
        double s = tv * 2.0 / cnt;
        for (int k = 0; k < 12; k++)
            for (int z = 0; z < 8; z++)
                for (int y = 0; y < 16; y++)
                    for (int x = 0; x < 16; x++) {
                        int c[3] = {z, y, x};
                        if (c[d] + 1 >= dims[d]) continue;
                        c[d]++;
                        size_t a = gi(k, z, y, x), b = gi(k, c[0], c[1], c[2]);
                        double df = s * (double(g[b]) - g[a]);
                        grad[b] += df;
                        grad[a] -= df;
                    }
    }
}

void BilateralGridModel::step(size_t i, const std::vector<float>& gpuGrad) {
    std::vector<double> g(gpuGrad.size());
    for (size_t k = 0; k < g.size(); k++) {
        if (!std::isfinite(gpuGrad[k])) return;
        g[k] = gpuGrad[k];
    }
    addTvGrad(i, g);
    opt[i].step(grids[i], g, lr, 1e-15);
}

void BilateralGridModel::pinMean() {
    if (grids.empty()) return;
    std::vector<float> id = identity();
    std::vector<double> dev(id.size(), 0.0);
    for (auto& g : grids)
        for (size_t k = 0; k < g.size(); k++) dev[k] += double(g[k]);
    for (size_t k = 0; k < dev.size(); k++) dev[k] = dev[k] / double(grids.size()) - id[k];
    for (auto& g : grids)
        for (size_t k = 0; k < g.size(); k++) g[k] = float(g[k] - dev[k]);
}

// ---------------------------------------------------------------- pose refinement

void rot6d(const double a[6], double R[3][3]) {
    double b1[3] = {a[0], a[1], a[2]}, a2[3] = {a[3], a[4], a[5]};
    auto norm = [](double* v) {   // F.normalize (eps 1e-12)
        double l = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        l = std::max(l, 1e-12);
        for (int k = 0; k < 3; k++) v[k] /= l;
    };
    norm(b1);
    double d = b1[0] * a2[0] + b1[1] * a2[1] + b1[2] * a2[2];
    double b2[3] = {a2[0] - d * b1[0], a2[1] - d * b1[1], a2[2] - d * b1[2]};
    norm(b2);
    double b3[3] = {b1[1] * b2[2] - b1[2] * b2[1], b1[2] * b2[0] - b1[0] * b2[2], b1[0] * b2[1] - b1[1] * b2[0]};
    for (int k = 0; k < 3; k++) { R[0][k] = b1[k]; R[1][k] = b2[k]; R[2][k] = b3[k]; }
}

void refinedW2c(const double w2c[4][4], const double d[9], double out[4][4]) {
    double a[6] = {d[3] + 1, d[4], d[5], d[6], d[7] + 1, d[8]};
    double R[3][3];
    rot6d(a, R);
    // ΔT = [R | t]: ΔT⁻¹ = [Rᵀ | −Rᵀ t]; out = ΔT⁻¹ · w2c
    double inv[4][4] = {};
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) inv[r][c] = R[c][r];
        inv[r][3] = -(R[0][r] * d[0] + R[1][r] * d[1] + R[2][r] * d[2]);
    }
    inv[3][3] = 1;
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) {
            double s = 0;
            for (int k = 0; k < 4; k++) s += inv[r][k] * w2c[k][c];
            out[r][c] = s;
        }
}

namespace {
void poseOutputs(const double w2c[4][4], const double d[9], double o[15]) {
    double m[4][4];
    refinedW2c(w2c, d, m);
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) o[r * 3 + c] = m[r][c];
    for (int r = 0; r < 3; r++) o[9 + r] = m[r][3];
    for (int k = 0; k < 3; k++) o[12 + k] = -(m[0][k] * m[0][3] + m[1][k] * m[1][3] + m[2][k] * m[2][3]);
}
}  // namespace

void poseChain(const double w2c[4][4], const double d[9], const double g15[15], double out[9]) {
    double dd[9];
    std::copy(d, d + 9, dd);
    for (int j = 0; j < 9; j++) {
        double h = 1e-6, op[15], om[15];
        dd[j] = d[j] + h; poseOutputs(w2c, dd, op);
        dd[j] = d[j] - h; poseOutputs(w2c, dd, om);
        dd[j] = d[j];
        double s = 0;
        for (int k = 0; k < 15; k++) s += g15[k] * (op[k] - om[k]) / (2 * h);
        out[j] = s;
    }
}

void PoseRefinerModel::setup(size_t views, double lr_, double reg_) {
    lr = lr_;
    reg = reg_;
    e.assign(views * 9, 0.0);
    m.assign(views * 9, 0.0);
    v.assign(views * 9, 0.0);
    steps = 0;
}

void PoseRefinerModel::step(size_t row, const double grad[9]) {
    const double b1 = 0.9, b2 = 0.999, eps = 1e-8;
    steps++;
    double bc1 = 1 - std::pow(b1, double(steps)), bc2 = 1 - std::pow(b2, double(steps));
    double stepSize = lr * std::sqrt(bc2) / bc1;
    for (int k = 0; k < 9; k++) {
        size_t i = row * 9 + size_t(k);
        double g = grad[k] + (reg > 0 ? 2 * reg * e[i] : 0.0);
        if (!std::isfinite(g)) return;
        m[i] = b1 * m[i] + (1 - b1) * g;
        v[i] = b2 * v[i] + (1 - b2) * g * g;
        e[i] -= stepSize * m[i] / (std::sqrt(v[i]) + eps);
    }
}

// ---------------------------------------------------------------- skybox

SkyBasis skyBasis(const double upIn[3]) {
    SkyBasis b;
    double l = std::sqrt(upIn[0] * upIn[0] + upIn[1] * upIn[1] + upIn[2] * upIn[2]) + 1e-12;
    for (int k = 0; k < 3; k++) b.up[k] = upIn[k] / l;
    double ref[3] = {0, 0, 1};
    if (std::abs(b.up[2]) >= 0.9) { ref[0] = 1; ref[2] = 0; }
    double d = ref[0] * b.up[0] + ref[1] * b.up[1] + ref[2] * b.up[2];
    double f[3] = {ref[0] - b.up[0] * d, ref[1] - b.up[1] * d, ref[2] - b.up[2] * d};
    double fl = std::sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]) + 1e-12;
    for (int k = 0; k < 3; k++) b.forward[k] = f[k] / fl;
    const double* u = b.up;
    const double* fw = b.forward;
    b.right[0] = u[1] * fw[2] - u[2] * fw[1];
    b.right[1] = u[2] * fw[0] - u[0] * fw[2];
    b.right[2] = u[0] * fw[1] - u[1] * fw[0];
    return b;
}

void skyRotation(const SkyBasis& b, const double w2c[4][4], float out[9]) {
    const double* B[3] = {b.right, b.up, b.forward};
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) out[i * 3 + j] = float(B[i][0] * w2c[j][0] + B[i][1] * w2c[j][1] + B[i][2] * w2c[j][2]);
}

namespace {
uint32_t crc32(const uint8_t* d, size_t n, uint32_t c = 0) {
    static uint32_t table[256];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t v = i;
            for (int k = 0; k < 8; k++) v = (v & 1) ? 0xEDB88320u ^ (v >> 1) : v >> 1;
            table[i] = v;
        }
        init = true;
    }
    c ^= 0xFFFFFFFFu;
    for (size_t i = 0; i < n; i++) c = table[(c ^ d[i]) & 255u] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}
void be32(std::vector<uint8_t>& o, uint32_t v) { for (int s = 24; s >= 0; s -= 8) o.push_back(uint8_t(v >> s)); }
void chunk(std::ofstream& f, const char* type, const std::vector<uint8_t>& data) {
    std::vector<uint8_t> b;
    be32(b, uint32_t(data.size()));
    b.insert(b.end(), type, type + 4);
    b.insert(b.end(), data.begin(), data.end());
    be32(b, crc32(b.data() + 4, b.size() - 4));
    f.write(reinterpret_cast<const char*>(b.data()), std::streamsize(b.size()));
}
}  // namespace

void writePng(const std::string& path, const std::vector<uint8_t>& rgb, int w, int h) {
    std::vector<uint8_t> raw;
    raw.reserve(size_t(h) * (size_t(w) * 3 + 1));
    for (int y = 0; y < h; y++) {
        raw.push_back(0);
        raw.insert(raw.end(), rgb.begin() + std::ptrdiff_t(size_t(y) * w * 3), rgb.begin() + std::ptrdiff_t(size_t(y + 1) * w * 3));
    }
    // zlib stream with stored (uncompressed) deflate blocks
    std::vector<uint8_t> z = {0x78, 0x01};
    for (size_t at = 0; at < raw.size() || at == 0;) {
        size_t len = std::min<size_t>(65535, raw.size() - at);
        bool last = at + len >= raw.size();
        z.push_back(last ? 1 : 0);
        z.push_back(uint8_t(len)); z.push_back(uint8_t(len >> 8));
        z.push_back(uint8_t(~len)); z.push_back(uint8_t((~len) >> 8));
        z.insert(z.end(), raw.begin() + std::ptrdiff_t(at), raw.begin() + std::ptrdiff_t(at + len));
        at += len;
        if (last) break;
    }
    uint32_t a = 1, b = 0;
    for (uint8_t c : raw) { a = (a + c) % 65521u; b = (b + a) % 65521u; }
    be32(z, (b << 16) | a);
    std::vector<uint8_t> ihdr;
    be32(ihdr, uint32_t(w)); be32(ihdr, uint32_t(h));
    ihdr.insert(ihdr.end(), {8, 2, 0, 0, 0});   // 8-bit RGB
    std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) throw Error("cannot write " + tmp);
        const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
        f.write(reinterpret_cast<const char*>(sig), 8);
        chunk(f, "IHDR", ihdr);
        chunk(f, "IDAT", z);
        chunk(f, "IEND", {});
        if (!f) throw Error("writing " + tmp + " failed");
    }
    fs::rename(tmp, path);
}

std::string writeSkySidecars(const std::string& plyPath, const std::vector<float>& logits, int W, int H, const SkyBasis& b) {
    fs::path ply(plyPath);
    fs::path png = ply, js = ply;
    png.replace_extension(".sky.png");
    js.replace_extension(".sky.json");
    std::vector<uint8_t> rgb(size_t(W) * H * 3);
    for (size_t i = 0; i < rgb.size(); i++) {
        double s = 1.0 / (1.0 + std::exp(-double(logits[i])));
        rgb[i] = uint8_t(std::clamp(s, 0.0, 1.0) * 255 + 0.5);
    }
    writePng(png.string(), rgb, W, H);
    json j = {{"format", "equirectangular"},
              {"image", png.filename().string()},
              {"up", {b.up[0], b.up[1], b.up[2]}},
              {"forward", {b.forward[0], b.forward[1], b.forward[2]}},
              {"right", {b.right[0], b.right[1], b.right[2]}},
              {"longitude", "atan2(dir·right, dir·forward), left edge = −180°"},
              {"latitude", "asin(dir·up), top = +90°"}};
    std::ofstream f(js, std::ios::trunc);
    if (!f) throw Error("cannot write " + js.string());
    f << j.dump();
    return png.string();
}

// ---------------------------------------------------------------- depth maps

namespace {
float halfToFloat(uint16_t h) {
    uint32_t s = uint32_t(h >> 15) << 31, e = (h >> 10) & 31u, m = h & 1023u;
    uint32_t bits;
    if (e == 0) {
        if (m == 0) bits = s;
        else {   // subnormal
            float v = std::ldexp(float(m), -24);
            return (h >> 15) ? -v : v;
        }
    } else if (e == 31) bits = s | 0x7F800000u | (m << 13);
    else bits = s | ((e + 112u) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}
}  // namespace

std::vector<float> loadDepthF16(const std::string& path, int w, int h) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw DatasetError("cannot read depth map " + path);
    std::vector<uint8_t> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (raw.size() != size_t(w) * h * 2)
        throw DatasetError(fs::path(path).filename().string() + ": expected " + std::to_string(w) + "x" + std::to_string(h) +
                           " float16 values, got " + std::to_string(raw.size() / 2));
    std::vector<float> d(size_t(w) * h);
    for (size_t i = 0; i < d.size(); i++) {
        float v = halfToFloat(uint16_t(raw[2 * i] | (raw[2 * i + 1] << 8)));
        d[i] = (std::isfinite(v) && v > 0) ? v : 0.0f;
    }
    return d;
}

bool priorCacheValid(const std::string& dir) {
    try {
        std::ifstream f(fs::path(dir) / "meta.json");
        if (!f) return false;
        json m = json::parse(f);
        return m.value("model", std::string()) == "depth-anything/Depth-Anything-V2-Small-hf" && m.value("version", 0) == 2;
    } catch (const std::exception&) {
        return false;
    }
}

std::string priorCacheKey(const CameraInfo& c) {
    // depthprior.py cache_key: sha1(image path) + c2w.round(6) + [fx, fy, cx, cy, width, height].round(4), as float64
    // bytes; numpy's round is rint(x · 10^d) / 10^d.
    std::string data = c.image;
    auto put = [&](double v, int dec) {
        double f = std::pow(10.0, dec);
        double r = std::nearbyint(v * f) / f;
        char b[8];
        std::memcpy(b, &r, 8);
        data.append(b, 8);
    };
    for (int r = 0; r < 4; r++)
        for (int k = 0; k < 4; k++) put(c.c2w[r][k], 6);
    for (double v : {c.fx, c.fy, c.cx, c.cy, double(c.width), double(c.height)}) put(v, 4);
    return std::to_string(c.id) + "-" + sha1Hex(data).substr(0, 12);
}

bool readNpyFloat2D(const std::string& path, std::vector<float>& out, int& w, int& h) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[6];
    if (!f.read(magic, 6) || std::memcmp(magic, "\x93NUMPY", 6) != 0) return false;
    uint8_t ver[2];
    if (!f.read(reinterpret_cast<char*>(ver), 2)) return false;
    uint32_t hlen = 0;
    if (ver[0] == 1) {
        uint8_t b[2];
        if (!f.read(reinterpret_cast<char*>(b), 2)) return false;
        hlen = b[0] | (b[1] << 8);
    } else {
        uint8_t b[4];
        if (!f.read(reinterpret_cast<char*>(b), 4)) return false;
        hlen = uint32_t(b[0]) | (uint32_t(b[1]) << 8) | (uint32_t(b[2]) << 16) | (uint32_t(b[3]) << 24);
    }
    if (hlen > 65536) return false;
    std::string hdr(hlen, '\0');
    if (!f.read(hdr.data(), std::streamsize(hlen))) return false;
    if (hdr.find("'<f4'") == std::string::npos || hdr.find("'fortran_order': False") == std::string::npos) return false;
    size_t s = hdr.find("'shape': (");
    if (s == std::string::npos) return false;
    long a = 0, b = 0;
    if (std::sscanf(hdr.c_str() + s + 10, "%ld, %ld", &a, &b) != 2 || a <= 0 || b <= 0 || a > 100000 || b > 100000) return false;
    h = int(a);
    w = int(b);
    out.resize(size_t(w) * h);
    if (!f.read(reinterpret_cast<char*>(out.data()), std::streamsize(out.size() * 4))) return false;
    for (float& v : out) if (!std::isfinite(v) || v < 0) v = 0;
    return true;
}

std::string sha1Hex(const std::string& msg) {
    uint32_t hs[5] = {0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u};
    std::vector<uint8_t> d(msg.begin(), msg.end());
    uint64_t bits = uint64_t(d.size()) * 8;
    d.push_back(0x80);
    while (d.size() % 64 != 56) d.push_back(0);
    for (int s = 56; s >= 0; s -= 8) d.push_back(uint8_t(bits >> s));
    auto rol = [](uint32_t x, int n) { return (x << n) | (x >> (32 - n)); };
    for (size_t at = 0; at < d.size(); at += 64) {
        uint32_t w[80];
        for (int i = 0; i < 16; i++)
            w[i] = (uint32_t(d[at + 4 * size_t(i)]) << 24) | (uint32_t(d[at + 4 * size_t(i) + 1]) << 16) |
                   (uint32_t(d[at + 4 * size_t(i) + 2]) << 8) | uint32_t(d[at + 4 * size_t(i) + 3]);
        for (int i = 16; i < 80; i++) w[i] = rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
        uint32_t a = hs[0], b = hs[1], c = hs[2], dd = hs[3], e = hs[4];
        for (int i = 0; i < 80; i++) {
            uint32_t f, k;
            if (i < 20) { f = (b & c) | (~b & dd); k = 0x5A827999u; }
            else if (i < 40) { f = b ^ c ^ dd; k = 0x6ED9EBA1u; }
            else if (i < 60) { f = (b & c) | (b & dd) | (c & dd); k = 0x8F1BBCDCu; }
            else { f = b ^ c ^ dd; k = 0xCA62C1D6u; }
            uint32_t t = rol(a, 5) + f + e + k + w[i];
            e = dd; dd = c; c = rol(b, 30); b = a; a = t;
        }
        hs[0] += a; hs[1] += b; hs[2] += c; hs[3] += dd; hs[4] += e;
    }
    char out[41];
    for (int i = 0; i < 5; i++) std::snprintf(out + 8 * i, 9, "%08x", hs[i]);
    return std::string(out, 40);
}

// ---------------------------------------------------------------- misc

double medianNearestViewGap(const std::vector<const CameraInfo*>& cams, double sceneExtent) {
    size_t n = cams.size();
    if (n < 3) return 0.0;
    double ext = std::max(sceneExtent, 1e-3);
    std::vector<std::array<double, 3>> pos(n), fwd(n);
    for (size_t i = 0; i < n; i++) {
        double l = 0;
        for (int k = 0; k < 3; k++) { pos[i][size_t(k)] = cams[i]->c2w[k][3]; fwd[i][size_t(k)] = cams[i]->c2w[k][2]; l += fwd[i][size_t(k)] * fwd[i][size_t(k)]; }
        l = std::sqrt(l);
        for (int k = 0; k < 3; k++) fwd[i][size_t(k)] /= l;
    }
    std::vector<double> gaps;
    size_t stride = std::max<size_t>(1, n / 400);
    for (size_t i = 0; i < n; i += stride) {
        double best = std::numeric_limits<double>::infinity();
        for (size_t j = 0; j < n; j++) {
            if (j == i) continue;
            double c = fwd[j][0] * fwd[i][0] + fwd[j][1] * fwd[i][1] + fwd[j][2] * fwd[i][2];
            double ang = std::acos(std::clamp(c, -1.0, 1.0)) * 180.0 / M_PI;
            double dx = pos[j][0] - pos[i][0], dy = pos[j][1] - pos[i][1], dz = pos[j][2] - pos[i][2];
            double off = std::sqrt(dx * dx + dy * dy + dz * dz) / ext * 180.0 / M_PI;
            best = std::min(best, ang + off);
        }
        gaps.push_back(best);
    }
    std::sort(gaps.begin(), gaps.end());
    size_t m = gaps.size();
    return m % 2 ? gaps[m / 2] : 0.5 * (gaps[m / 2 - 1] + gaps[m / 2]);
}

}  // namespace vkt
