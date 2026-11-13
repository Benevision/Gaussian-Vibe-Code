// splatwork-vk-train: the SplatWork Vulkan trainer program. Stage 1 (kernels) only offers:
//   splatwork-vk-train --info                       device summary as one JSON line
//   splatwork-vk-train --dump SCENE.bin OUT.bin     render one scene and back-propagate a given image gradient
//                      [--atomics auto|native|cas] [--device N]
// The training loop (dataset, TrainingConfig, progress, checkpoints) is stage 2.
//
// SCENE.bin (little-endian), written by compare_gsplat.py:
//   char[4] "SWVS", u32 version = 1
//   u32 n, width, height, shDegree, antialiased
//   f64 w2c[16] (row-major world→camera, OpenCV), f64 fx, fy, cx, cy
//   f32 bg[3], f32 dilation
//   f32 pos[n*3], logScale[n*3], quat[n*4] (w,x,y,z), opLogit[n], sh[n*48] ([n][16][3])
//   f32 dLdImage[3*height*width] (planes r, g, b)
// OUT.bin:
//   char[4] "SWVO", u32 version = 1, u32 n, width, height, tileInstances
//   f32 image[3*h*w] (GPU), f32 alpha[h*w] (1 − final T), f32 imageCpu[3*h*w] (double-precision reference)
//   f32 grad pos[n*3], logScale[n*3], quat[n*4], opLogit[n], sh[n*48]
//   u32 tilesTouched[n]
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "cpu_ref.h"
#include "gpu_splat.h"

using namespace vkt;

namespace {

struct Reader {
    std::vector<char> data;
    size_t at = 0;
    template <class T> T get() {
        if (at + sizeof(T) > data.size()) throw Error("scene file is truncated");
        T v;
        std::memcpy(&v, data.data() + at, sizeof(T));
        at += sizeof(T);
        return v;
    }
    template <class T> void get(std::vector<T>& v, size_t count) {
        if (at + count * sizeof(T) > data.size()) throw Error("scene file is truncated");
        v.resize(count);
        std::memcpy(v.data(), data.data() + at, count * sizeof(T));
        at += count * sizeof(T);
    }
};

struct Writer {
    std::ofstream f;
    explicit Writer(const std::string& path) : f(path, std::ios::binary) { if (!f) throw Error("cannot write " + path); }
    template <class T> void put(T v) { f.write(reinterpret_cast<const char*>(&v), sizeof(T)); }
    template <class T> void put(const std::vector<T>& v) { f.write(reinterpret_cast<const char*>(v.data()), std::streamsize(v.size() * sizeof(T))); }
};

int dump(const std::string& scenePath, const std::string& outPath, const ContextOptions& co) {
    Reader r;
    {
        std::ifstream f(scenePath, std::ios::binary);
        if (!f) throw Error("cannot read " + scenePath);
        r.data.assign(std::istreambuf_iterator<char>(f), {});
    }
    char magic[4];
    for (char& c : magic) c = r.get<char>();
    if (std::memcmp(magic, "SWVS", 4) != 0 || r.get<uint32_t>() != 1) throw Error("not a SWVS v1 scene file");
    int n = int(r.get<uint32_t>());
    Camera cam;
    cam.width = int(r.get<uint32_t>());
    cam.height = int(r.get<uint32_t>());
    RenderOpts opt;
    opt.shDegree = int(r.get<uint32_t>());
    opt.antialiased = r.get<uint32_t>() != 0;
    for (auto& row : cam.w2c) for (double& v : row) v = r.get<double>();
    cam.fx = r.get<double>(); cam.fy = r.get<double>(); cam.cx = r.get<double>(); cam.cy = r.get<double>();
    for (float& v : opt.bg) v = r.get<float>();
    opt.dilation = r.get<float>();
    opt.absGrad = false;
    opt.hasDepth = false;
    Params P;
    P.resize(n);
    for (int k = 0; k < 5; k++) r.get(P.group(k), size_t(n) * Params::width(k));
    size_t hw = size_t(cam.width) * cam.height;
    std::vector<float> gimg;
    r.get(gimg, 3 * hw);
    gimg.resize(4 * hw, 0.0f);

    Context ctx(co);
    std::fprintf(stderr, "device: %s, subgroup %u, atomics %s\n", ctx.info.name.c_str(), ctx.info.subgroupSize, ctx.nativeAtomics ? "native" : "cas");
    GpuSplat gs(ctx);
    gs.setParams(P);
    gs.setCamera(cam, opt);
    uint32_t m = gs.forward();
    auto img = gs.read<float>(gs.img, 4 * hw);
    auto fT = gs.read<float>(gs.finalT, hw);
    ctx.upload(gs.gradImg, gimg.data(), gimg.size() * 4);
    gs.rasterBackward();
    gs.projectBackward();
    Params G;
    gs.getGrads(G);
    auto tt = gs.read<uint32_t>(gs.tilesTouched, size_t(n));

    ref::Cam rc = ref::makeCam(cam, opt);
    auto pr = ref::projectAll(ref::toDouble(P), rc, true);
    std::vector<double> cImg, cT;
    std::vector<uint32_t> cN;
    ref::rasterDecide(pr, rc, cImg, cT, cN, nullptr);

    Writer w(outPath);
    w.f.write("SWVO", 4);
    w.put<uint32_t>(1); w.put<uint32_t>(uint32_t(n)); w.put<uint32_t>(uint32_t(cam.width)); w.put<uint32_t>(uint32_t(cam.height));
    w.put<uint32_t>(m);
    w.put(std::vector<float>(img.begin(), img.begin() + 3 * hw));
    std::vector<float> alpha(hw);
    for (size_t p = 0; p < hw; p++) alpha[p] = 1.0f - fT[p];
    w.put(alpha);
    w.put(std::vector<float>(cImg.begin(), cImg.begin() + 3 * hw));
    for (int k = 0; k < 5; k++) w.put(G.group(k));
    w.put(tt);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    ContextOptions co;
    std::string mode, scene, out;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); } return argv[++i]; };
        if (a == "--info") mode = "info";
        else if (a == "--dump") { mode = "dump"; scene = next(); out = next(); }
        else if (a == "--device") co.deviceIndex = std::stoi(next());
        else if (a == "--atomics") {
            std::string v = next();
            co.atomics = v == "native" ? ContextOptions::Atomics::Native : v == "cas" ? ContextOptions::Atomics::Cas : ContextOptions::Atomics::Auto;
        } else if (a == "-h" || a == "--help") { mode = ""; break; }
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    try {
        if (mode == "info") {
            Context ctx(co);
            std::printf("{\"device\": \"%s\", \"driver\": \"%s\", \"vendorID\": %u, \"deviceID\": %u, \"vulkan\": \"%u.%u\", "
                        "\"subgroupSize\": %u, \"floatAtomicAdd\": %s}\n",
                        ctx.info.name.c_str(), ctx.info.driver.c_str(), ctx.info.vendorID, ctx.info.deviceID,
                        VK_API_VERSION_MAJOR(ctx.info.apiVersion), VK_API_VERSION_MINOR(ctx.info.apiVersion), ctx.info.subgroupSize,
                        ctx.info.floatAtomicAdd ? "true" : "false");
            return 0;
        }
        if (mode == "dump") return dump(scene, out, co);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "splatwork-vk-train: %s\n", e.what());
        return 1;
    }
    std::printf("usage: splatwork-vk-train --info | --dump SCENE.bin OUT.bin [--atomics auto|native|cas] [--device N]\n"
                "(stage 1: kernels only; training comes in stage 2)\n");
    return mode.empty() ? 0 : 2;
}
