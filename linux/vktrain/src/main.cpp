// splatwork-vk-train: the SplatWork Vulkan trainer program.
//   splatwork-vk-train --info                       device summary as one JSON line
//   splatwork-vk-train --dump SCENE.bin OUT.bin     render one scene and back-propagate a given image gradient
//   splatwork-vk-train train --dataset DIR --out result.ply [--config cfg.json] [--set key=value ...]
//                      [--iterations N] [--job-dir DIR] [--resume FILE] [--preview-every N] [--opacity-reset]
//   common options:    [--atomics auto|native|cas] [--device N]
//
// train: trains one prepared dataset (docs/PROTOCOL.md §2) with one TrainingConfig (§3) like the CUDA worker's
// trainer.py. Output on stdout, one JSON object per line ("type": status | log | preview | checkpoint | done |
// paused | cancelled | error — the messages trainer.subprocess_entry puts on its queue). With --job-dir the
// trainer writes DIR/preview.splat every previewInterval steps (--preview-every overrides it),
// DIR/checkpoint.ply on request and on pause, and DIR/resume.vkstate on pause.
// Control: one command per line on stdin — "checkpoint" (write checkpoint.ply at the next step), "pause"
// (write checkpoint.ply and the resume file, then exit 3), "cancel" (exit 4); SIGTERM / SIGINT cancel too.
// End of input on stdin is not a command (a closed stdin keeps training). --resume FILE continues from a saved
// state (and a later pause saves there again). Exit codes: 0 finished, 1 error, 2 invalid arguments or
// config, 3 paused, 4 cancelled.
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
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <csignal>
#include <iostream>
#include <thread>

#include "config.h"
#include "cpu_ref.h"
#include "gpu_splat.h"
#include "trainer.h"

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

Control* g_control = nullptr;
extern "C" void onSignal(int) {
    if (g_control) g_control->cancel = true;
}

int train(int argc, char** argv, int first, ContextOptions co) {
    TrainOptions opt;
    std::string configPath;
    std::vector<std::string> sets;
    int64_t iterations = -1;
    for (int i = first; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw ConfigError("missing value for " + a);
            return argv[++i];
        };
        if (a == "--dataset") opt.datasetDir = next();
        else if (a == "--config") configPath = next();
        else if (a == "--out") opt.outPath = next();
        else if (a == "--job-dir") opt.jobDir = next();
        else if (a == "--resume") opt.resumeFrom = next();
        else if (a == "--preview-every") opt.previewEvery = std::stoll(next());
        else if (a == "--set") sets.push_back(next());
        else if (a == "--iterations") iterations = std::stoll(next());
        else if (a == "--opacity-reset") opt.opacityReset = true;
        else if (a == "--pause-at") opt.pauseAt = std::stoll(next());        // test aid
        else if (a == "--trace-every") opt.traceEvery = std::stoll(next());  // test aid
        else if (a == "--dump-eval") opt.dumpEval = next();                  // debugging aid
        else if (a == "--device") co.deviceIndex = std::stoi(next());
        else if (a == "--atomics") {
            std::string v = next();
            co.atomics = v == "native" ? ContextOptions::Atomics::Native : v == "cas" ? ContextOptions::Atomics::Cas : ContextOptions::Atomics::Auto;
        } else throw ConfigError("unknown argument " + a);
    }
    if (opt.datasetDir.empty() || opt.outPath.empty()) throw ConfigError("train needs --dataset DIR and --out FILE");
    nlohmann::json cj = nlohmann::json::object();
    if (!configPath.empty()) {
        std::ifstream f(configPath);
        if (!f) throw ConfigError("cannot read " + configPath);
        try { cj = nlohmann::json::parse(f); } catch (const std::exception& e) { throw ConfigError(std::string("config: ") + e.what()); }
    }
    for (const std::string& kv : sets) {   // like cli.py --set: the value as JSON, else as a string
        size_t eq = kv.find('=');
        if (eq == std::string::npos) throw ConfigError("--set expects key=value, got " + kv);
        std::string key = kv.substr(0, eq), value = kv.substr(eq + 1);
        try { cj[key] = nlohmann::json::parse(value); } catch (...) { cj[key] = value; }
    }
    if (iterations >= 0) cj["iterations"] = iterations;
    TrainConfig cfg = TrainConfig::fromJson(cj);
    opt.ctx = co;
    namespace fs = std::filesystem;
    if (!opt.jobDir.empty()) fs::create_directories(opt.jobDir);
    if (fs::path(opt.outPath).has_parent_path()) fs::create_directories(fs::path(opt.outPath).parent_path());

    static Control control;
    g_control = &control;
    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);
    // stdin commands; the thread is left blocked in getline when training ends (the process exits).
    std::thread([]() {
        std::string line;
        while (std::getline(std::cin, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
            if (line == "cancel" || line == "stop") g_control->cancel = true;
            else if (line == "pause") g_control->pause = true;
            else if (line == "checkpoint") g_control->checkpoint = true;
        }
    }).detach();
    Emitter em;
    return runTraining(opt, cfg, control, em);
}

void printError(const std::string& msg) {
    std::printf("%s\n", nlohmann::json({{"type", "error"}, {"message", msg}}).dump().c_str());
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
    ContextOptions co;
    std::string mode, scene, out;
    if (argc >= 2 && std::string(argv[1]) == "train") {
        int rc = EXIT_ERROR;
        try {
            rc = train(argc, argv, 2, co);
        } catch (const ConfigError& e) {
            printError(std::string("invalid config: ") + e.what());
            std::fprintf(stderr, "splatwork-vk-train: %s\n", e.what());
            rc = EXIT_USAGE;
        } catch (const std::exception& e) {
            printError(e.what());
            rc = EXIT_ERROR;
        }
        std::fflush(stdout);
        std::_Exit(rc);   // don't wait for the stdin thread
    }
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
                        "\"subgroupSize\": %u, \"floatAtomicAdd\": %s, \"nativeAtomics\": %s}\n",
                        ctx.info.name.c_str(), ctx.info.driver.c_str(), ctx.info.vendorID, ctx.info.deviceID,
                        VK_API_VERSION_MAJOR(ctx.info.apiVersion), VK_API_VERSION_MINOR(ctx.info.apiVersion), ctx.info.subgroupSize,
                        ctx.info.floatAtomicAdd ? "true" : "false", ctx.nativeAtomics ? "true" : "false");
            return 0;
        }
        if (mode == "dump") return dump(scene, out, co);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "splatwork-vk-train: %s\n", e.what());
        return 1;
    }
    std::printf("usage: splatwork-vk-train --info\n"
                "       splatwork-vk-train --dump SCENE.bin OUT.bin\n"
                "       splatwork-vk-train train --dataset DIR --out result.ply [--config cfg.json] [--set key=value ...]\n"
                "                          [--iterations N] [--job-dir DIR] [--resume FILE] [--preview-every N] [--opacity-reset]\n"
                "       options: [--atomics auto|native|cas] [--device N]\n");
    return mode.empty() ? 0 : 2;
}
