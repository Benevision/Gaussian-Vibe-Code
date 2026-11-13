#include "dataset.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

#define STBI_NO_HDR
#define STBI_NO_LINEAR
#define STBI_NO_GIF
#define STBI_NO_PSD
#define STBI_NO_PIC
#define STBI_NO_PNM
#define STBI_NO_TGA
#define STB_IMAGE_IMPLEMENTATION
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wsign-compare"
#endif
#include "stb/stb_image.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace fs = std::filesystem;
using nlohmann::json;

namespace vkt {

// ------------------------------------------------------------------ paths

std::string DatasetInfo::resolve(const std::string& rel) const {
    if (rel.empty() || rel.find('\0') != std::string::npos || rel.find('\\') != std::string::npos)
        throw DatasetError("invalid relative path: " + rel);
    fs::path p(rel);
    if (p.is_absolute()) throw DatasetError("path escapes dataset directory: " + rel);
    for (const auto& part : p)
        if (part == "..") throw DatasetError("path escapes dataset directory: " + rel);
    fs::path base = fs::weakly_canonical(fs::path(root));
    fs::path full = fs::weakly_canonical(base / p);
    auto b = base.string(), f = full.string();
    if (f != b && (f.size() <= b.size() || f.compare(0, b.size(), b) != 0 || (f[b.size()] != '/' && b.back() != '/')))
        throw DatasetError("path escapes dataset directory: " + rel);
    return f;
}

namespace {

bool isRoot(const fs::path& d) { return fs::is_regular_file(d / "dataset.json") || fs::is_directory(d / "images"); }

fs::path findDatasetRoot(const fs::path& dir) {
    fs::path root = dir;
    if (!isRoot(dir)) {
        std::vector<fs::path> cand;
        std::error_code ec;
        for (auto& e : fs::directory_iterator(dir, ec)) {
            std::string name = e.path().filename().string();
            if (!e.is_directory() || name.empty() || name[0] == '.' || name == "__MACOSX") continue;
            if (isRoot(e.path())) cand.push_back(e.path());
        }
        if (cand.size() == 1) root = cand[0];
    }
    if (!fs::is_regular_file(root / "dataset.json"))
        throw DatasetError("dataset.json not found (this upload can only be used for processing jobs)");
    return root;
}

double num(const json& v, const std::string& name) {
    if (!v.is_number() || v.is_boolean() || !std::isfinite(v.get<double>())) throw DatasetError(name + " must be a finite number");
    return v.get<double>();
}

void matrix(const json& v, const std::string& name, double m[4][4]) {
    std::vector<double> flat;
    std::vector<size_t> shape;
    auto err = [&]() { throw DatasetError(name + " must be a 4x4 numeric matrix (row-major)"); };
    if (!v.is_array()) err();
    if (!v.empty() && v[0].is_array()) {
        for (const json& row : v) {
            if (!row.is_array() || row.size() != v[0].size()) err();
            for (const json& e : row) {
                if (!e.is_number() || e.is_boolean()) err();
                flat.push_back(e.get<double>());
            }
        }
        shape = {v.size(), v[0].size()};
    } else {
        for (const json& e : v) {
            if (!e.is_number() || e.is_boolean()) err();
            flat.push_back(e.get<double>());
        }
        shape = {flat.size()};
    }
    if (shape.size() == 1 && shape[0] == 16) shape = {4, 4};   // tolerate flat row-major
    if (shape == std::vector<size_t>{3, 4}) {
        flat.insert(flat.end(), {0, 0, 0, 1});
        shape = {4, 4};
    }
    if (shape != std::vector<size_t>{4, 4}) err();
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) {
            m[r][c] = flat[size_t(r * 4 + c)];
            if (!std::isfinite(m[r][c])) err();
        }
}

void invert4(const double a[4][4], double out[4][4]) {
    double m[4][8];
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 8; c++) m[r][c] = c < 4 ? a[r][c] : (c - 4 == r ? 1.0 : 0.0);
    for (int c = 0; c < 4; c++) {
        int piv = c;
        for (int r = c + 1; r < 4; r++) if (std::abs(m[r][c]) > std::abs(m[piv][c])) piv = r;
        if (std::abs(m[piv][c]) < 1e-300) throw DatasetError("cameraToWorld is singular");
        if (piv != c) for (int k = 0; k < 8; k++) std::swap(m[c][k], m[piv][k]);
        double d = m[c][c];
        for (int k = 0; k < 8; k++) m[c][k] /= d;
        for (int r = 0; r < 4; r++) {
            if (r == c) continue;
            double f = m[r][c];
            if (f != 0) for (int k = 0; k < 8; k++) m[r][k] -= f * m[c][k];
        }
    }
    for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) out[r][c] = m[r][c + 4];
}

std::vector<uint8_t> readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw DatasetError("cannot read " + path);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// EXIF orientation (tag 0x0112) of a JPEG file; 1 when absent.
int jpegOrientation(const std::vector<uint8_t>& d) {
    if (d.size() < 4 || d[0] != 0xFF || d[1] != 0xD8) return 1;
    size_t at = 2;
    while (at + 4 <= d.size() && d[at] == 0xFF) {
        uint8_t marker = d[at + 1];
        if (marker == 0xDA || marker == 0xD9) break;
        size_t len = (size_t(d[at + 2]) << 8) | d[at + 3];
        if (marker == 0xE1 && len >= 8 && at + 2 + len <= d.size() && std::memcmp(&d[at + 4], "Exif\0\0", 6) == 0) {
            size_t t = at + 10, end = at + 2 + len;
            if (t + 8 > end) return 1;
            bool le = d[t] == 'I';
            auto u16 = [&](size_t p) -> uint32_t { return le ? d[p] | (d[p + 1] << 8) : (d[p] << 8) | d[p + 1]; };
            auto u32 = [&](size_t p) -> uint32_t {
                return le ? uint32_t(d[p]) | (uint32_t(d[p + 1]) << 8) | (uint32_t(d[p + 2]) << 16) | (uint32_t(d[p + 3]) << 24)
                          : (uint32_t(d[p]) << 24) | (uint32_t(d[p + 1]) << 16) | (uint32_t(d[p + 2]) << 8) | uint32_t(d[p + 3]);
            };
            size_t ifd = t + u32(t + 4);
            if (ifd + 2 > end) return 1;
            uint32_t count = u16(ifd);
            for (uint32_t i = 0; i < count; i++) {
                size_t e = ifd + 2 + size_t(i) * 12;
                if (e + 12 > end) break;
                if (u16(e) == 0x0112) {
                    uint32_t o = u16(e + 8);
                    return o >= 1 && o <= 8 ? int(o) : 1;
                }
            }
            return 1;
        }
        at += 2 + len;
    }
    return 1;
}

// PIL ImageOps.exif_transpose for orientation o.
void applyOrientation(std::vector<uint8_t>& px, int& w, int& h, int ch, int o) {
    if (o <= 1) return;
    bool swap = o >= 5;
    int nw = swap ? h : w, nh = swap ? w : h;
    std::vector<uint8_t> out(px.size());
    for (int y = 0; y < nh; y++)
        for (int x = 0; x < nw; x++) {
            int sx = 0, sy = 0;
            switch (o) {
                case 2: sx = w - 1 - x; sy = y; break;                 // mirror horizontal
                case 3: sx = w - 1 - x; sy = h - 1 - y; break;         // rotate 180
                case 4: sx = x; sy = h - 1 - y; break;                 // mirror vertical
                case 5: sx = y; sy = x; break;                         // transpose
                case 6: sx = y; sy = h - 1 - x; break;                 // rotate 90 CW
                case 7: sx = w - 1 - y; sy = h - 1 - x; break;         // transverse
                case 8: sx = w - 1 - y; sy = x; break;                 // rotate 270 CW
                default: sx = x; sy = y;
            }
            std::memcpy(&out[(size_t(y) * nw + x) * ch], &px[(size_t(sy) * w + sx) * ch], size_t(ch));
        }
    px.swap(out);
    w = nw;
    h = nh;
}

struct Decoded {
    std::vector<uint8_t> px;
    int w = 0, h = 0, ch = 0;
};

Decoded decode(const std::string& path, int wantCh /* 0: rgb or rgba as stored */) {
    std::vector<uint8_t> file = readFile(path);
    int w = 0, h = 0, comp = 0;
    if (!stbi_info_from_memory(file.data(), int(file.size()), &w, &h, &comp))
        throw DatasetError("cannot decode " + path + " (" + (stbi_failure_reason() ? stbi_failure_reason() : "?") +
                           "; JPEG and PNG are supported)");
    int ch = wantCh ? wantCh : (comp == 2 || comp == 4 ? 4 : 3);
    unsigned char* data = stbi_load_from_memory(file.data(), int(file.size()), &w, &h, &comp, ch);
    if (!data) throw DatasetError("cannot decode " + path + ": " + (stbi_failure_reason() ? stbi_failure_reason() : "?"));
    Decoded d;
    d.px.assign(data, data + size_t(w) * h * ch);
    stbi_image_free(data);
    d.w = w; d.h = h; d.ch = ch;
    applyOrientation(d.px, d.w, d.h, ch, jpegOrientation(file));
    return d;
}

double lanczos3(double x) {
    if (x < 0) x = -x;
    if (x >= 3.0) return 0.0;
    if (x < 1e-12) return 1.0;
    double px = M_PI * x;
    return 3.0 * std::sin(px) * std::sin(px / 3.0) / (px * px);
}
double triangle(double x) {
    if (x < 0) x = -x;
    return x < 1.0 ? 1.0 - x : 0.0;
}

// One axis of PIL's resampling (ImagingResample precompute_coeffs + horizontal/vertical pass), in float.
void resampleAxis(const std::vector<float>& src, std::vector<float>& dst, int len, int nlen, int stride, int count,
                  int ch, Filter f, bool alongX, int w) {
    double scale = double(len) / nlen;
    double fs_ = std::max(scale, 1.0);
    double support = (f == Filter::Lanczos ? 3.0 : 1.0) * fs_;
    for (int o = 0; o < nlen; o++) {
        double center = (o + 0.5) * scale;
        int xmin = std::max(int(center - support + 0.5), 0);
        int xmax = std::min(int(center + support + 0.5), len);
        std::vector<double> wts;
        double ww = 0;
        for (int x = xmin; x < xmax; x++) {
            double t = (x - center + 0.5) / fs_;
            double wv = f == Filter::Lanczos ? lanczos3(t) : triangle(t);
            wts.push_back(wv);
            ww += wv;
        }
        if (ww != 0) for (double& v : wts) v /= ww;
        for (int line = 0; line < count; line++)
            for (int c = 0; c < ch; c++) {
                double s = 0;
                for (int x = xmin; x < xmax; x++) {
                    size_t si = alongX ? (size_t(line) * w + x) * ch + c : (size_t(x) * stride + line) * ch + c;
                    s += wts[size_t(x - xmin)] * src[si];
                }
                size_t di = alongX ? (size_t(line) * nlen + o) * ch + c : (size_t(o) * stride + line) * ch + c;
                dst[di] = float(s);
            }
    }
}

}  // namespace

std::vector<uint8_t> resample(const std::vector<uint8_t>& src, int w, int h, int ch, int nw, int nh, Filter f) {
    if (w == nw && h == nh) return src;
    std::vector<float> a(src.begin(), src.end());
    std::vector<float> b(size_t(nw) * h * ch);
    if (w != nw) resampleAxis(a, b, w, nw, 0, h, ch, f, true, w);
    else b = a;
    std::vector<float> c(size_t(nw) * nh * ch);
    if (h != nh) resampleAxis(b, c, h, nh, nw, nw, ch, f, false, nw);
    else c = b;
    std::vector<uint8_t> out(c.size());
    for (size_t i = 0; i < c.size(); i++) out[i] = uint8_t(std::clamp(std::lround(c[i]), 0L, 255L));
    return out;
}

std::vector<float> areaResize(const std::vector<float>& src, int w, int h, int ch, int nw, int nh) {
    std::vector<float> out(size_t(nw) * nh * ch);
    for (int y = 0; y < nh; y++) {
        int y0 = int(std::floor(double(y) * h / nh)), y1 = int(std::ceil(double(y + 1) * h / nh));
        for (int x = 0; x < nw; x++) {
            int x0 = int(std::floor(double(x) * w / nw)), x1 = int(std::ceil(double(x + 1) * w / nw));
            double inv = 1.0 / double((y1 - y0) * (x1 - x0));
            for (int c = 0; c < ch; c++) {
                double s = 0;
                for (int yy = y0; yy < y1; yy++)
                    for (int xx = x0; xx < x1; xx++) s += src[size_t(c) * w * h + size_t(yy) * w + xx];
                out[size_t(c) * nw * nh + size_t(y) * nw + x] = float(s * inv);
            }
        }
    }
    return out;
}

DatasetInfo parseDataset(const std::string& dir) {
    if (!fs::is_directory(dir)) throw DatasetError("dataset directory not found: " + dir);
    DatasetInfo ds;
    fs::path root = fs::weakly_canonical(findDatasetRoot(fs::path(dir)));
    ds.root = root.string();
    json raw;
    try {
        std::ifstream f(root / "dataset.json");
        raw = json::parse(f);
    } catch (const std::exception& e) {
        throw DatasetError(std::string("cannot read dataset.json: ") + e.what());
    }
    if (!raw.is_object()) throw DatasetError("dataset.json must be a JSON object");
    if (raw.value("format", std::string("splatdataset")) != "splatdataset") throw DatasetError("unsupported dataset format");
    if (raw.contains("version") && (!raw["version"].is_number_integer() || raw["version"].get<int64_t>() > 1))
        throw DatasetError("unsupported dataset version " + raw["version"].dump());
    if (raw.contains("convention") && raw["convention"] != "opencv")
        throw DatasetError("unsupported camera convention " + raw["convention"].dump() + " (expected 'opencv')");
    if (!raw.contains("cameras") || !raw["cameras"].is_array() || raw["cameras"].empty())
        throw DatasetError("dataset.json has no cameras");

    size_t i = 0;
    for (const json& c : raw["cameras"]) {
        std::string label = "camera " + std::to_string(i);
        if (!c.is_object()) throw DatasetError(label + " must be an object");
        CameraInfo cam;
        if (!c.contains("image") || !c["image"].is_string()) throw DatasetError(label + ": missing image");
        cam.image = c["image"].get<std::string>();
        ds.resolve(cam.image);
        if (!c.contains("width") || !c.contains("height") || !c["width"].is_number_integer() || !c["height"].is_number_integer() ||
            c["width"].get<int64_t>() <= 0 || c["height"].get<int64_t>() <= 0)
            throw DatasetError(label + ": width/height must be positive integers");
        cam.width = int(c["width"].get<int64_t>());
        cam.height = int(c["height"].get<int64_t>());
        auto get = [&](const char* k) { return num(c.contains(k) ? c[k] : json(), label + "." + k); };
        cam.fx = get("fx"); cam.fy = get("fy"); cam.cx = get("cx"); cam.cy = get("cy");
        if (cam.fx <= 0 || cam.fy <= 0) throw DatasetError(label + ": focal lengths must be positive");
        matrix(c.contains("cameraToWorld") ? c["cameraToWorld"] : json(), label + ".cameraToWorld", cam.c2w);
        if (c.contains("depth") && !c["depth"].is_null()) {
            if (!c["depth"].is_string()) throw DatasetError(label + ": depth must be a path");
            cam.depth = c["depth"].get<std::string>();
            ds.resolve(cam.depth);
            auto dim = [&](const char* k) -> int {
                if (!c.contains(k) || !c[k].is_number_integer() || c[k].get<int64_t>() <= 0 || c[k].get<int64_t>() > 1000000)
                    throw DatasetError(label + ": depthWidth/depthHeight required with depth");
                return int(c[k].get<int64_t>());
            };
            cam.depthWidth = dim("depthWidth");
            cam.depthHeight = dim("depthHeight");
        }
        for (const char* key : {"mask", "sky"}) {
            if (c.contains(key) && !c[key].is_null()) {
                if (!c[key].is_string()) throw DatasetError(label + ": " + key + " must be a path");
                std::string v = c[key].get<std::string>();
                ds.resolve(v);
                (std::strcmp(key, "mask") == 0 ? cam.mask : cam.sky) = v;
            }
        }
        cam.id = int64_t(i);
        if (c.contains("id") && c["id"].is_number() && !c["id"].is_boolean()) cam.id = int64_t(c["id"].get<double>());
        if (c.contains("holdout")) {
            const json& h = c["holdout"];
            cam.holdout = h.is_boolean() ? h.get<bool>() : (h.is_number() ? h.get<double>() != 0 : (h.is_string() && !h.get<std::string>().empty()));
        }
        if (c.contains("exposure") && c["exposure"].is_number() && !c["exposure"].is_boolean() && c["exposure"].get<double>() > 0)
            cam.exposure = c["exposure"].get<double>();
        ds.cameras.push_back(cam);
        i++;
    }
    if (std::all_of(ds.cameras.begin(), ds.cameras.end(), [](const CameraInfo& c) { return c.holdout; }))
        throw DatasetError("all cameras are holdout; nothing to train on");

    double mean[3] = {0, 0, 0};
    for (auto& c : ds.cameras) for (int k = 0; k < 3; k++) mean[k] += c.c2w[k][3] / double(ds.cameras.size());
    auto vec3 = [&](const char* key, double out[3]) {
        if (raw.contains(key) && raw[key].is_array() && raw[key].size() == 3) {
            for (int k = 0; k < 3; k++) out[k] = num(raw[key][size_t(k)], key);
            return true;
        }
        return false;
    };
    if (!vec3("sceneCenter", ds.center)) std::copy(mean, mean + 3, ds.center);
    if (raw.contains("sceneExtent") && raw["sceneExtent"].is_number() && !raw["sceneExtent"].is_boolean() &&
        std::isfinite(raw["sceneExtent"].get<double>()) && raw["sceneExtent"].get<double>() > 0) {
        ds.extent = raw["sceneExtent"].get<double>();
    } else {
        double mx = 0;
        for (auto& c : ds.cameras) {
            double d2 = 0;
            for (int k = 0; k < 3; k++) d2 += (c.c2w[k][3] - ds.center[k]) * (c.c2w[k][3] - ds.center[k]);
            mx = std::max(mx, std::sqrt(d2));
        }
        ds.extent = mx * 1.1 > 0 ? mx * 1.1 : 1.0;
    }
    if (!vec3("upVector", ds.up)) { ds.up[0] = 0; ds.up[1] = -1; ds.up[2] = 0; }
    if (raw.contains("gravityUp") && raw["gravityUp"].is_array() && raw["gravityUp"].size() == 3) {
        bool ok = true;
        for (int k = 0; k < 3; k++) {
            const json& e = raw["gravityUp"][size_t(k)];
            if (!e.is_number() || e.is_boolean()) ok = false; else ds.gravityUp[k] = e.get<double>();
        }
        ds.hasGravityUp = ok;
    }
    bool explicitPoints = raw.contains("points") && !raw["points"].is_null();
    if (explicitPoints) {
        if (!raw["points"].is_string()) throw DatasetError("points must be a path");
        ds.points = raw["points"].get<std::string>();
        ds.resolve(ds.points);
    } else if (fs::is_regular_file(root / "points3d.ply")) {
        ds.points = "points3d.ply";
    }
    size_t missing = 0;
    std::string firstMissing;
    for (auto& c : ds.cameras)
        if (!fs::is_regular_file(ds.resolve(c.image))) { if (!missing++) firstMissing = c.image; }
    if (missing) throw DatasetError(std::to_string(missing) + " image(s) missing, e.g. " + firstMissing);
    for (auto& c : ds.cameras)
        if (!c.depth.empty() && !fs::is_regular_file(ds.resolve(c.depth))) throw DatasetError("depth file missing: " + c.depth);
    if (!ds.points.empty() && !fs::is_regular_file(ds.resolve(ds.points))) {
        if (explicitPoints) throw DatasetError("point cloud missing: " + ds.points);
        ds.points.clear();
    }
    return ds;
}

void targetSize(int w, int h, int maxSize, int& nw, int& nh) {
    int longest = std::max(w, h);
    if (longest <= maxSize) { nw = w; nh = h; return; }
    double s = double(maxSize) / longest;
    nw = std::max(1, int(std::nearbyint(w * s)));   // Python round(): half to even, like nearbyint
    nh = std::max(1, int(std::nearbyint(h * s)));
}

Camera View::camera() const {
    Camera c;
    for (int r = 0; r < 4; r++) for (int k = 0; k < 4; k++) c.w2c[r][k] = w2c[r][k];
    c.fx = fx; c.fy = fy; c.cx = cx; c.cy = cy;
    c.width = width; c.height = height;
    return c;
}

View loadView(const DatasetInfo& ds, const CameraInfo& cam, int maxSize) {
    View v;
    v.cam = &cam;
    Decoded im = decode(ds.resolve(cam.image), 0);
    // Intrinsics are in dataset.json width/height pixel units: a file of another size is first resized to them.
    if (im.w != cam.width || im.h != cam.height) {
        im.px = resample(im.px, im.w, im.h, im.ch, cam.width, cam.height, Filter::Lanczos);
        im.w = cam.width; im.h = cam.height;
    }
    int nw = 0, nh = 0;
    targetSize(cam.width, cam.height, maxSize, nw, nh);
    if (nw != im.w || nh != im.h) im.px = resample(im.px, im.w, im.h, im.ch, nw, nh, Filter::Lanczos);
    v.width = nw; v.height = nh;
    size_t hw = size_t(nw) * nh;
    v.rgb.resize(hw * 3);
    if (im.ch == 4) v.alpha.resize(hw);
    for (size_t p = 0; p < hw; p++) {
        for (int c = 0; c < 3; c++) v.rgb[p * 3 + c] = im.px[p * im.ch + c];
        if (im.ch == 4) v.alpha[p] = im.px[p * 4 + 3];
    }
    if (!v.alpha.empty() && std::all_of(v.alpha.begin(), v.alpha.end(), [](uint8_t a) { return a == 255; })) v.alpha.clear();
    if (!cam.mask.empty() && fs::is_regular_file(ds.resolve(cam.mask))) {
        Decoded m = decode(ds.resolve(cam.mask), 1);
        if (m.w != nw || m.h != nh) m.px = resample(m.px, m.w, m.h, 1, nw, nh, Filter::Bilinear);
        v.mask = std::move(m.px);
        if (std::all_of(v.mask.begin(), v.mask.end(), [](uint8_t a) { return a == 255; })) v.mask.clear();
    }
    v.fx = cam.fx * nw / double(cam.width);
    v.fy = cam.fy * nh / double(cam.height);
    v.cx = cam.cx * nw / double(cam.width);
    v.cy = cam.cy * nh / double(cam.height);
    invert4(cam.c2w, v.w2c);
    return v;
}

bool cameraSeesBox(const CameraInfo& cam, const double lo[3], const double hi[3], int samples, double nearZ) {
    const double* c = nullptr;
    double centre[3] = {cam.c2w[0][3], cam.c2w[1][3], cam.c2w[2][3]};
    c = centre;
    if (c[0] >= lo[0] && c[0] <= hi[0] && c[1] >= lo[1] && c[1] <= hi[1] && c[2] >= lo[2] && c[2] <= hi[2]) return true;
    double w2c[4][4];
    invert4(cam.c2w, w2c);
    for (int a = 0; a < samples; a++)
        for (int b = 0; b < samples; b++)
            for (int d = 0; d < samples; d++) {
                double t[3] = {double(a) / (samples - 1), double(b) / (samples - 1), double(d) / (samples - 1)};
                double p[3], q[3];
                for (int k = 0; k < 3; k++) p[k] = lo[k] + t[k] * (hi[k] - lo[k]);
                for (int r = 0; r < 3; r++) q[r] = w2c[r][0] * p[0] + w2c[r][1] * p[1] + w2c[r][2] * p[2] + w2c[r][3];
                if (q[2] <= nearZ) continue;
                double u = cam.fx * q[0] / q[2] + cam.cx, v = cam.fy * q[1] / q[2] + cam.cy;
                if (u >= 0 && u < cam.width && v >= 0 && v < cam.height) return true;
            }
    return false;
}

std::vector<const CameraInfo*> selectCameras(const DatasetInfo& ds, const std::vector<int64_t>* ids, const double* lo, const double* hi) {
    std::vector<const CameraInfo*> out;
    for (auto& c : ds.cameras) {
        if (ids && std::find(ids->begin(), ids->end(), c.id) == ids->end()) continue;
        if (lo && hi && !cameraSeesBox(c, lo, hi)) continue;
        out.push_back(&c);
    }
    return out;
}

bool exposureLocked(const std::vector<const CameraInfo*>& cams) {
    if (cams.empty()) return false;
    double mn = INFINITY, mx = 0;
    for (auto* c : cams) { mn = std::min(mn, c->exposure); mx = std::max(mx, c->exposure); }
    return mn > 0 && mx / mn <= 1.1;
}

namespace {
inline uint32_t pack(float r, float g, float b, float m) {
    auto q = [](float v) { return uint32_t(std::clamp(std::lround(v * 255.0f), 0L, 255L)); };
    return q(r) | (q(g) << 8) | (q(b) << 16) | (q(m) << 24);
}
}  // namespace

void View::compact() {
    if (!alpha.empty() || !packed.empty()) return;
    const float none[3] = {0, 0, 0};
    packed = packTarget(*this, none);
    rgb.clear(); rgb.shrink_to_fit();
    mask.clear(); mask.shrink_to_fit();
}

void View::pixel(size_t p, float c[3], float& a, float& m) const {
    if (!packed.empty()) {
        uint32_t t = packed[p];
        for (int k = 0; k < 3; k++) c[k] = float((t >> (8 * k)) & 255u) / 255.0f;
        a = 1.0f;
        m = float(t >> 24) / 255.0f;
        return;
    }
    for (int k = 0; k < 3; k++) c[k] = rgb[p * 3 + size_t(k)] / 255.0f;
    a = alpha.empty() ? 1.0f : alpha[p] / 255.0f;
    m = mask.empty() ? 1.0f : mask[p] / 255.0f;
}

std::vector<uint32_t> packTarget(const View& v, const float bg[3]) {
    if (!v.packed.empty()) return v.packed;
    size_t hw = size_t(v.width) * v.height;
    std::vector<uint32_t> out(hw);
    for (size_t p = 0; p < hw; p++) {
        uint32_t m = v.mask.empty() ? 255u : v.mask[p];
        if (v.alpha.empty()) {
            out[p] = uint32_t(v.rgb[p * 3]) | (uint32_t(v.rgb[p * 3 + 1]) << 8) | (uint32_t(v.rgb[p * 3 + 2]) << 16) | (m << 24);
        } else {
            float a = v.alpha[p] / 255.0f;
            float c[3];
            for (int k = 0; k < 3; k++) c[k] = v.rgb[p * 3 + k] / 255.0f * a + (1 - a) * bg[k];
            out[p] = pack(c[0], c[1], c[2], m / 255.0f);
        }
    }
    return out;
}

std::vector<uint32_t> packTargetScaled(const View& v, const float bg[3], double scale, Camera& cam) {
    int w = v.width, h = v.height;
    int nh = std::max(8, int(std::nearbyint(h * scale))), nw = std::max(8, int(std::nearbyint(w * scale)));
    double sy = double(nh) / h, sx = double(nw) / w;
    cam = v.camera();
    cam.width = nw; cam.height = nh;
    cam.fx = v.fx * sx; cam.fy = v.fy * sy;
    cam.cx = (v.cx + 0.5) * sx - 0.5;
    cam.cy = (v.cy + 0.5) * sy - 0.5;
    size_t hw = size_t(w) * h;
    int ch = 5;   // r g b alpha mask
    std::vector<float> planes(hw * ch);
    for (size_t p = 0; p < hw; p++) {
        float c[3], a, m;
        v.pixel(p, c, a, m);
        for (int k = 0; k < 3; k++) planes[k * hw + p] = c[k];
        planes[3 * hw + p] = a;
        planes[4 * hw + p] = m;
    }
    std::vector<float> s = areaResize(planes, w, h, ch, nw, nh);
    size_t nhw = size_t(nw) * nh;
    std::vector<uint32_t> out(nhw);
    for (size_t p = 0; p < nhw; p++) {
        float a = s[3 * nhw + p];
        float c[3];
        for (int k = 0; k < 3; k++) c[k] = s[k * nhw + p] * a + (1 - a) * bg[k];
        out[p] = pack(c[0], c[1], c[2], s[4 * nhw + p]);
    }
    return out;
}

}  // namespace vkt
