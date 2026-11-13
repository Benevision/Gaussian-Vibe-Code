#include "plyio.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "vk.h"

namespace vkt {

namespace {

struct Prop {
    std::string name, type;
    bool list = false;
};
struct Element {
    std::string name;
    size_t count = 0;
    std::vector<Prop> props;
};

size_t typeSize(const std::string& t) {
    static const std::map<std::string, size_t> sizes = {
        {"char", 1}, {"int8", 1}, {"uchar", 1}, {"uint8", 1}, {"short", 2}, {"int16", 2}, {"ushort", 2}, {"uint16", 2},
        {"int", 4}, {"int32", 4}, {"uint", 4}, {"uint32", 4}, {"float", 4}, {"float32", 4}, {"double", 8}, {"float64", 8}};
    auto it = sizes.find(t);
    if (it == sizes.end()) throw Error("unknown PLY type " + t);
    return it->second;
}
bool isIntType(const std::string& t) { return t != "float" && t != "float32" && t != "double" && t != "float64"; }

double readValue(const uint8_t* p, const std::string& t, bool bigEndian) {
    uint8_t b[8];
    size_t n = typeSize(t);
    std::memcpy(b, p, n);
    if (bigEndian && n >= 2 && n <= 8) std::reverse(b, b + n);
    if (t == "char" || t == "int8") { int8_t v; std::memcpy(&v, b, 1); return v; }
    if (t == "uchar" || t == "uint8") return b[0];
    if (t == "short" || t == "int16") { int16_t v; std::memcpy(&v, b, 2); return v; }
    if (t == "ushort" || t == "uint16") { uint16_t v; std::memcpy(&v, b, 2); return v; }
    if (t == "int" || t == "int32") { int32_t v; std::memcpy(&v, b, 4); return v; }
    if (t == "uint" || t == "uint32") { uint32_t v; std::memcpy(&v, b, 4); return v; }
    if (t == "float" || t == "float32") { float v; std::memcpy(&v, b, 4); return v; }
    double v; std::memcpy(&v, b, 8); return v;
}

void writeAtomically(const std::string& path, const std::string& header, const void* data, size_t bytes) {
    std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) throw Error("cannot write " + tmp);
        f.write(header.data(), std::streamsize(header.size()));
        f.write(static_cast<const char*>(data), std::streamsize(bytes));
        if (!f) throw Error("writing " + tmp + " failed");
    }
    std::filesystem::rename(tmp, path);
}

}  // namespace

void readPointCloud(const std::string& path, std::vector<float>& xyz, std::vector<float>& rgb) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw Error("cannot read " + path);
    std::string line;
    std::getline(f, line);
    if (line.rfind("ply", 0) != 0) throw Error("not a PLY file");
    std::string fmt;
    std::vector<Element> elements;
    for (;;) {
        if (!std::getline(f, line)) throw Error("unexpected end of header");
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream ss(line);
        std::string key;
        ss >> key;
        if (key == "format") ss >> fmt;
        else if (key == "element") {
            Element e;
            ss >> e.name >> e.count;
            elements.push_back(e);
        } else if (key == "property") {
            if (elements.empty()) throw Error("property before element");
            Prop p;
            std::string t;
            ss >> t;
            if (t == "list") {
                std::string a, b;
                ss >> a >> b >> p.name;
                p.list = true;
            } else {
                typeSize(t);
                p.type = t;
                ss >> p.name;
            }
            elements.back().props.push_back(p);
        } else if (key == "end_header") break;
    }
    if (fmt != "ascii" && fmt != "binary_little_endian" && fmt != "binary_big_endian") throw Error("unsupported PLY format " + fmt);
    bool big = fmt == "binary_big_endian";
    for (const Element& e : elements) {
        bool hasList = std::any_of(e.props.begin(), e.props.end(), [](const Prop& p) { return p.list; });
        if (hasList) {
            if (e.name == "vertex") throw Error("list properties on vertex are not supported");
            throw Error("variable-size element '" + e.name + "' before vertex is not supported");
        }
        size_t rowBytes = 0;
        for (auto& p : e.props) rowBytes += typeSize(p.type);
        if (e.name != "vertex") {
            if (fmt == "ascii") for (size_t i = 0; i < e.count; i++) std::getline(f, line);
            else f.seekg(std::streamoff(rowBytes * e.count), std::ios::cur);
            continue;
        }
        std::map<std::string, size_t> idx;
        for (size_t k = 0; k < e.props.size(); k++) idx[e.props[k].name] = k;
        if (!idx.count("x") || !idx.count("y") || !idx.count("z")) throw Error("point cloud has no x/y/z");
        const char* names[3][3] = {{"red", "green", "blue"}, {"r", "g", "b"}, {"diffuse_red", "diffuse_green", "diffuse_blue"}};
        int colSet = -1;
        for (int s = 0; s < 3 && colSet < 0; s++)
            if (idx.count(names[s][0]) && idx.count(names[s][1]) && idx.count(names[s][2])) colSet = s;
        std::vector<double> row(e.props.size());
        std::vector<uint8_t> buf(rowBytes);
        xyz.clear();
        rgb.clear();
        for (size_t i = 0; i < e.count; i++) {
            if (fmt == "ascii") {
                if (!std::getline(f, line)) throw Error("PLY file truncated");
                std::istringstream ss(line);
                for (auto& v : row) ss >> v;
            } else {
                if (!f.read(reinterpret_cast<char*>(buf.data()), std::streamsize(rowBytes))) throw Error("PLY file truncated");
                size_t at = 0;
                for (size_t k = 0; k < e.props.size(); k++) {
                    row[k] = readValue(buf.data() + at, e.props[k].type, big);
                    at += typeSize(e.props[k].type);
                }
            }
            float p[3] = {float(row[idx["x"]]), float(row[idx["y"]]), float(row[idx["z"]])};
            if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2])) continue;
            xyz.insert(xyz.end(), p, p + 3);
            if (colSet >= 0) {
                for (int c = 0; c < 3; c++) {
                    const Prop& pr = e.props[idx[names[colSet][c]]];
                    double v = row[idx[names[colSet][c]]];
                    float out;
                    if (isIntType(pr.type)) out = float(v / (typeSize(pr.type) == 2 ? 65535.0 : 255.0));
                    else out = float(std::clamp(v, 0.0, 1.0));
                    rgb.push_back(out);
                }
            }
        }
        return;
    }
    throw Error("PLY has no vertex element");
}

size_t writeGaussianPly(const std::string& path, const Params& p, int shDegree, const double* up, const std::vector<uint8_t>* keep) {
    const int nCols = 62;   // x y z nx ny nz f_dc×3 f_rest×45 opacity scale×3 rot×4
    std::vector<float> data;
    data.reserve(size_t(p.n) * nCols);
    int nc = (shDegree + 1) * (shDegree + 1);
    size_t written = 0;
    for (int i = 0; i < p.n; i++) {
        if (keep && !keep->empty() && !(*keep)[size_t(i)]) continue;
        written++;
        for (int k = 0; k < 3; k++) data.push_back(p.pos[size_t(i) * 3 + k]);
        for (int k = 0; k < 3; k++) data.push_back(0.0f);
        const float* sh = &p.sh[size_t(i) * 48];
        for (int c = 0; c < 3; c++) data.push_back(sh[c]);
        for (int c = 0; c < 3; c++)   // channel-major: f_rest_0..14 = R coefficients 1..15, then G, then B
            for (int k = 1; k < 16; k++) data.push_back(k < nc ? sh[k * 3 + c] : 0.0f);
        data.push_back(p.opLogit[size_t(i)]);
        for (int k = 0; k < 3; k++) data.push_back(p.logScale[size_t(i) * 3 + k]);
        const float* q = &p.quat[size_t(i) * 4];
        double len = std::sqrt(double(q[0]) * q[0] + double(q[1]) * q[1] + double(q[2]) * q[2] + double(q[3]) * q[3]);
        if (len > 0) for (int k = 0; k < 4; k++) data.push_back(float(q[k] / std::max(len, 1e-12)));
        else { data.push_back(1); data.push_back(0); data.push_back(0); data.push_back(0); }
    }
    std::string h = "ply\nformat binary_little_endian 1.0\n";
    if (up) {
        char buf[160];
        std::snprintf(buf, sizeof buf, "comment splatstudio_up %.6g %.6g %.6g\n", up[0], up[1], up[2]);
        h += buf;
    }
    h += "element vertex " + std::to_string(written) + "\n";
    std::vector<std::string> names = {"x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"};
    for (int i = 0; i < 45; i++) names.push_back("f_rest_" + std::to_string(i));
    for (const char* s : {"opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"}) names.push_back(s);
    for (auto& s : names) h += "property float " + s + "\n";
    h += "end_header\n";
    writeAtomically(path, h, data.data(), data.size() * sizeof(float));
    return written;
}

size_t writePreviewSplat(const std::string& path, const Params& p, size_t maxSplats) {
    const double SH_C0 = 0.28209479177387814;
    size_t n = size_t(p.n);
    std::vector<float> score(n), op(n);
    for (size_t i = 0; i < n; i++) {
        op[i] = float(1.0 / (1.0 + std::exp(-double(p.opLogit[i]))));
        score[i] = op[i] * std::exp(p.logScale[i * 3]) * std::exp(p.logScale[i * 3 + 1]) * std::exp(p.logScale[i * 3 + 2]);
    }
    std::vector<uint32_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0u);
    size_t m = std::min(maxSplats, n);
    auto cmp = [&](uint32_t a, uint32_t b) { return score[a] > score[b] || (score[a] == score[b] && a < b); };
    if (m < n) std::partial_sort(idx.begin(), idx.begin() + std::ptrdiff_t(m), idx.end(), cmp);
    else std::sort(idx.begin(), idx.end(), cmp);
    idx.resize(m);
    std::vector<uint8_t> rec(m * 32);
    auto u8 = [](double v) { return uint8_t(std::clamp(std::nearbyint(v), 0.0, 255.0)); };
    for (size_t j = 0; j < m; j++) {
        size_t i = idx[j];
        uint8_t* r = &rec[j * 32];
        float pos[3] = {p.pos[i * 3], p.pos[i * 3 + 1], p.pos[i * 3 + 2]};
        float sc[3] = {std::exp(p.logScale[i * 3]), std::exp(p.logScale[i * 3 + 1]), std::exp(p.logScale[i * 3 + 2])};
        std::memcpy(r, pos, 12);
        std::memcpy(r + 12, sc, 12);
        for (int c = 0; c < 3; c++) r[24 + c] = u8(std::clamp(0.5 + SH_C0 * p.sh[i * 48 + size_t(c)], 0.0, 1.0) * 255.0);
        r[27] = u8(double(op[i]) * 255.0);
        const float* q = &p.quat[i * 4];
        double len = std::max(std::sqrt(double(q[0]) * q[0] + double(q[1]) * q[1] + double(q[2]) * q[2] + double(q[3]) * q[3]), 1e-12);
        for (int k = 0; k < 4; k++) r[28 + k] = u8(q[k] / len * 128.0 + 128.0);
    }
    writeAtomically(path, std::string(), rec.data(), rec.size());
    return m;
}

}  // namespace vkt
