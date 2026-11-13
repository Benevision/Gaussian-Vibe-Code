// The training loop, ported by behaviour from linux/splatworker/trainer.py train() (the CUDA / gsplat trainer):
// same initialisation, view order, learning rates and schedules, loss, strategies (gsplat 1.5 MCMCStrategy /
// DefaultStrategy, see strategy.h), evaluation, previews, checkpoints and pause / resume. The GPU side runs
// like the Mac's MetalTrainer: per step one forward submission (projection, sort, tile lists), one for the
// rasterisation + loss, one for backward + regularisers + Adam + MCMC noise; densification on a host copy.
#include "trainer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>

#include <iterator>
#include <sstream>

#include "compact.h"
#include "dataset.h"
#include "gpu_splat.h"
#include "models.h"
#include "plyio.h"
#include "strategy.h"

namespace fs = std::filesystem;
using nlohmann::json;

namespace vkt {

void Emitter::emit(const json& j) {
    std::string line = j.dump(-1, ' ', false, json::error_handler_t::replace);
    std::lock_guard<std::mutex> lock(mu_);
    std::fputs(line.c_str(), out_);
    std::fputc('\n', out_);
    std::fflush(out_);
}

std::string TrainOptions::resumePath() const {
    if (!resumeFrom.empty()) return resumeFrom;
    if (!jobDir.empty()) return (fs::path(jobDir) / "resume.vkstate").string();
    return std::string();
}

namespace {

using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

std::string fmt(const char* f, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return buf;
}

json numOrNull(double v) { return std::isfinite(v) ? json(v) : json(nullptr); }

// trainer.py progressive_scale: ¼ → ½ → ¾ → full resolution over the first half of the densification window.
double progressiveScale(int64_t step, int64_t densifyStop) {
    double half = double(std::max<int64_t>(densifyStop, 1)) / 2;
    if (step < half / 3) return 0.25;
    if (step < 2 * half / 3) return 0.5;
    if (step < half) return 0.75;
    return 1.0;
}

// ------------------------------------------------------------------ resume file

constexpr uint32_t RESUME_VERSION = 2;

// The stage-2b models and optimiser states (trainer.py save_state's appearance / bilagrid / pose / skybox /
// transient entries).
struct Extras {
    bool compact = false;   // optimizerPrecision "compact": the saved moments are compact.h carriers
    bool appOn = false, gridOn = false, poseOn = false, skyOn = false;
    AppearanceModel app;
    BilateralGridModel grid;
    PoseRefinerModel pose;
    std::vector<float> skyTex, skyM, skyV;   // host copies (save / resume)
    int64_t skySteps = 0;
    float tThreshold = 0;
    bool tHas = false;
    double tFraction = 0;
};

struct Saved {
    uint32_t trainViews = 0, evalViews = 0;
    int64_t step = 0, adamSteps = 0;
    SplatState s;
    std::vector<float> grad2d, count;
    std::vector<uint32_t> order;
    std::string rng, strategy;
    double psnrEma = NAN, lossEma = NAN, evalPsnr = NAN, elapsed = 0;
    int64_t previewStep = -1;
    bool capLogged = false;
    Extras x;
};

struct Out {
    std::ofstream f;
    template <class T> void put(const T& v) { f.write(reinterpret_cast<const char*>(&v), sizeof(T)); }
    template <class T> void vec(const std::vector<T>& v) {
        put(uint64_t(v.size()));
        f.write(reinterpret_cast<const char*>(v.data()), std::streamsize(v.size() * sizeof(T)));
    }
    void str(const std::string& s) { put(uint64_t(s.size())); f.write(s.data(), std::streamsize(s.size())); }
};
struct In {
    std::istream& f;
    template <class T> T get() {
        T v{};
        if (!f.read(reinterpret_cast<char*>(&v), sizeof(T))) throw Error("the saved training state is truncated");
        return v;
    }
    template <class T> void vec(std::vector<T>& v) {
        uint64_t n = get<uint64_t>();
        if (n > (uint64_t(1) << 36)) throw Error("the saved training state is corrupt");
        v.resize(size_t(n));
        if (n && !f.read(reinterpret_cast<char*>(v.data()), std::streamsize(n * sizeof(T))))
            throw Error("the saved training state is truncated");
    }
    std::string str() {
        std::vector<char> v;
        vec(v);
        return std::string(v.begin(), v.end());
    }
};

void writeParams(Out& o, const Params& p) {
    o.put(int64_t(p.n));
    for (int k = 0; k < 5; k++) o.vec(p.group(k));
}
void readParams(In& in, Params& p) {
    int64_t n = in.get<int64_t>();
    p.n = int(n);
    for (int k = 0; k < 5; k++) {
        in.vec(p.group(k));
        if (p.group(k).size() != size_t(n) * Params::width(k)) throw Error("the saved training state is corrupt");
    }
}

void writeExtras(Out& o, const Extras& x) {
    o.put(uint8_t(x.compact));
    o.put(uint8_t(x.appOn));
    if (x.appOn) {
        o.put(uint64_t(x.app.p.size()));
        for (size_t i = 0; i < x.app.p.size(); i++) {
            o.vec(x.app.p[i]); o.vec(x.app.opt[i].m); o.vec(x.app.opt[i].v); o.put(x.app.opt[i].steps);
        }
    }
    o.put(uint8_t(x.gridOn));
    if (x.gridOn) {
        o.put(uint64_t(x.grid.grids.size()));
        for (size_t i = 0; i < x.grid.grids.size(); i++) {
            o.vec(x.grid.grids[i]); o.vec(x.grid.opt[i].m); o.vec(x.grid.opt[i].v); o.put(x.grid.opt[i].steps);
        }
    }
    o.put(uint8_t(x.poseOn));
    if (x.poseOn) { o.vec(x.pose.e); o.vec(x.pose.m); o.vec(x.pose.v); o.put(x.pose.steps); }
    o.put(uint8_t(x.skyOn));
    if (x.skyOn) { o.vec(x.skyTex); o.vec(x.skyM); o.vec(x.skyV); o.put(x.skySteps); }
    o.put(x.tThreshold); o.put(uint8_t(x.tHas)); o.put(x.tFraction);
}

// Reads into x, whose models were set up from the config: the saved state must have the same shape.
void readExtras(In& in, Extras& x) {
    auto corrupt = []() { throw Error("the saved training state does not match this job's features; it can't be resumed"); };
    if ((in.get<uint8_t>() != 0) != x.compact)
        throw Error("the saved training state uses another optimizerPrecision; it can't be resumed");
    bool app = in.get<uint8_t>() != 0;
    if (app != x.appOn) corrupt();
    if (app) {
        if (in.get<uint64_t>() != x.app.p.size()) corrupt();
        for (size_t i = 0; i < x.app.p.size(); i++) {
            in.vec(x.app.p[i]); in.vec(x.app.opt[i].m); in.vec(x.app.opt[i].v); x.app.opt[i].steps = in.get<int64_t>();
            if (x.app.p[i].size() != 12) corrupt();
        }
    }
    bool grid = in.get<uint8_t>() != 0;
    if (grid != x.gridOn) corrupt();
    if (grid) {
        if (in.get<uint64_t>() != x.grid.grids.size()) corrupt();
        for (size_t i = 0; i < x.grid.grids.size(); i++) {
            in.vec(x.grid.grids[i]); in.vec(x.grid.opt[i].m); in.vec(x.grid.opt[i].v); x.grid.opt[i].steps = in.get<int64_t>();
            if (x.grid.grids[i].size() != size_t(BILAGRID_SIZE)) corrupt();
        }
    }
    bool pose = in.get<uint8_t>() != 0;
    if (pose != x.poseOn) corrupt();
    if (pose) {
        size_t n = x.pose.e.size();
        in.vec(x.pose.e); in.vec(x.pose.m); in.vec(x.pose.v); x.pose.steps = in.get<int64_t>();
        if (x.pose.e.size() != n || x.pose.m.size() != n || x.pose.v.size() != n) corrupt();
    }
    bool sky = in.get<uint8_t>() != 0;
    if (sky != x.skyOn) corrupt();
    if (sky) {
        size_t n = x.skyTex.size();
        in.vec(x.skyTex); in.vec(x.skyM); in.vec(x.skyV); x.skySteps = in.get<int64_t>();
        if (x.skyTex.size() != n || x.skyM.size() != n || x.skyV.size() != n) corrupt();
    }
    x.tThreshold = in.get<float>(); x.tHas = in.get<uint8_t>() != 0; x.tFraction = in.get<double>();
}

void saveState(const std::string& path, const Saved& s) {
    std::string tmp = path + ".tmp";
    {
        Out o;
        o.f.open(tmp, std::ios::binary | std::ios::trunc);
        if (!o.f) throw Error("cannot write " + tmp);
        o.f.write("SWVR", 4);
        o.put(RESUME_VERSION);
        o.put(s.trainViews); o.put(s.evalViews);
        o.put(s.step); o.put(s.adamSteps);
        writeParams(o, s.s.p); writeParams(o, s.s.m1); writeParams(o, s.s.m2);
        o.vec(s.grad2d); o.vec(s.count); o.vec(s.order);
        o.str(s.rng); o.str(s.strategy);
        o.put(s.psnrEma); o.put(s.lossEma); o.put(s.evalPsnr); o.put(s.elapsed);
        o.put(s.previewStep); o.put(uint8_t(s.capLogged));
        writeExtras(o, s.x);
        if (!o.f) throw Error("writing " + tmp + " failed");
    }
    fs::rename(tmp, path);   // atomic: a crash mid-save leaves the previous state (or none), never half a file
}

// Everything but the stage-2b extras, which stay in `extrasBlob` until the models are set up (readExtras).
Saved loadState(const std::string& path, std::string& extrasBlob) {
    std::ifstream file(path, std::ios::binary);
    In in{file};
    if (!in.f) throw Error("cannot read the saved training state " + path);
    char magic[4];
    in.f.read(magic, 4);
    if (!in.f || std::memcmp(magic, "SWVR", 4) != 0) throw Error(path + " is not a Vulkan trainer state");
    if (in.get<uint32_t>() != RESUME_VERSION)
        throw Error("the saved state of this paused job is from another trainer version; it can't be resumed");
    Saved s;
    s.trainViews = in.get<uint32_t>(); s.evalViews = in.get<uint32_t>();
    s.step = in.get<int64_t>(); s.adamSteps = in.get<int64_t>();
    readParams(in, s.s.p); readParams(in, s.s.m1); readParams(in, s.s.m2);
    in.vec(s.grad2d); in.vec(s.count); in.vec(s.order);
    s.rng = in.str(); s.strategy = in.str();
    s.psnrEma = in.get<double>(); s.lossEma = in.get<double>(); s.evalPsnr = in.get<double>(); s.elapsed = in.get<double>();
    s.previewStep = in.get<int64_t>(); s.capLogged = in.get<uint8_t>() != 0;
    extrasBlob.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    return s;
}

// ------------------------------------------------------------------ initial points (trainer.py init_points)

void initPoints(const DatasetInfo& ds, const TrainConfig& cfg, const double* lo, const double* hi, HostRng& rng,
                Emitter& em, std::vector<float>& xyz, std::vector<float>& rgb) {
    xyz.clear();
    rgb.clear();
    if (!ds.points.empty()) {
        try {
            readPointCloud(ds.resolve(ds.points), xyz, rgb);
            if (lo && hi) {
                std::vector<float> x2, c2;
                for (size_t i = 0; i < xyz.size() / 3; i++) {
                    bool in = true;
                    for (int k = 0; k < 3; k++) in = in && xyz[i * 3 + k] >= lo[k] && xyz[i * 3 + k] <= hi[k];
                    if (!in) continue;
                    x2.insert(x2.end(), &xyz[i * 3], &xyz[i * 3] + 3);
                    if (!rgb.empty()) c2.insert(c2.end(), &rgb[i * 3], &rgb[i * 3] + 3);
                }
                xyz.swap(x2);
                rgb.swap(c2);
                em.log(fmt("region: kept %zu initial points inside the expanded box", xyz.size() / 3));
            }
            if (xyz.empty()) {
                em.log("point cloud empty (after region filter); falling back to random init");
            } else {
                em.log(fmt("initialising from %s: %zu points", ds.points.c_str(), xyz.size() / 3));
                if (rgb.empty()) { rgb.resize(xyz.size()); for (float& v : rgb) v = float(rng.uniform()); }
            }
        } catch (const std::exception& e) {
            em.log(std::string("failed to read point cloud (") + e.what() + "); falling back to random init");
            xyz.clear();
            rgb.clear();
        }
    }
    if (xyz.empty()) {
        int64_t n = cfg.randomInitCount;
        double blo[3], bhi[3];
        for (int k = 0; k < 3; k++) {
            blo[k] = lo ? lo[k] : ds.center[k] - ds.extent * 1.5;
            bhi[k] = hi ? hi[k] : ds.center[k] + ds.extent * 1.5;
        }
        xyz.resize(size_t(n) * 3);
        rgb.resize(size_t(n) * 3);
        for (size_t i = 0; i < size_t(n); i++)
            for (int k = 0; k < 3; k++) xyz[i * 3 + k] = float(blo[k] + rng.uniform() * (bhi[k] - blo[k]));
        for (float& v : rgb) v = float(rng.uniform());
        em.log(fmt("random init: %lld gaussians in box [%g, %g, %g] .. [%g, %g, %g]", (long long)n, blo[0], blo[1], blo[2],
                   bhi[0], bhi[1], bhi[2]));
    }
    size_t count = xyz.size() / 3;
    if (int64_t(count) > cfg.maxGaussians) {
        std::vector<size_t> idx(count);
        std::iota(idx.begin(), idx.end(), size_t(0));
        size_t m = size_t(cfg.maxGaussians);
        for (size_t i = 0; i < m; i++) std::swap(idx[i], idx[i + size_t(rng.uniform() * double(count - i)) % (count - i)]);
        std::vector<float> x2, c2;
        for (size_t j = 0; j < m; j++) {
            x2.insert(x2.end(), &xyz[idx[j] * 3], &xyz[idx[j] * 3] + 3);
            c2.insert(c2.end(), &rgb[idx[j] * 3], &rgb[idx[j] * 3] + 3);
        }
        xyz.swap(x2);
        rgb.swap(c2);
        em.log(fmt("subsampled initial points to maxGaussians=%lld", (long long)cfg.maxGaussians));
    }
    if (cfg.backgroundShell > 0 && !(lo && hi)) {
        double centre[3] = {0, 0, 0};
        for (auto& c : ds.cameras) for (int k = 0; k < 3; k++) centre[k] += c.c2w[k][3] / double(ds.cameras.size());
        std::vector<float> shell = backgroundShell(centre, 3.0 * ds.extent, cfg.backgroundShell);
        double mean[3] = {0, 0, 0};
        size_t n = rgb.size() / 3;
        for (size_t i = 0; i < n; i++) for (int k = 0; k < 3; k++) mean[k] += rgb[i * 3 + k] / double(n);
        xyz.insert(xyz.end(), shell.begin(), shell.end());
        for (int64_t i = 0; i < cfg.backgroundShell; i++) for (int k = 0; k < 3; k++) rgb.push_back(float(mean[k]));
        em.log(fmt("background shell: %lld splats at 3× the scene size", (long long)cfg.backgroundShell));
    }
}

struct Cancelled {};

}  // namespace

int runTraining(const TrainOptions& opt, const TrainConfig& cfg, Control& ctl, Emitter& em) {
    try {
        auto tStart = Clock::now();
        double elapsedBefore = 0;   // from a resumed state
        em.status({{"state", "loading"}, {"message", "loading dataset"}, {"totalSteps", cfg.iterations}});

        Context ctx(opt.ctx);
        em.log(fmt("Vulkan trainer on %s (%s), float atomics: %s%s", ctx.info.name.c_str(), ctx.info.driver.c_str(),
                   ctx.nativeAtomics ? "native" : "compare-and-swap",
                   ctx.info.nativeAtomicsBroken ? " (native atomics failed the start-up probe)" : ""));

        DatasetInfo ds = parseDataset(opt.datasetDir);
        double lo[3], hi[3];
        bool region = cfg.hasRegion;
        if (region) cfg.region.expanded(lo, hi);
        auto cams = selectCameras(ds, cfg.hasCameraIds ? &cfg.cameraIds : nullptr, region ? lo : nullptr, region ? hi : nullptr);
        if (cfg.hasCameraIds) em.log(fmt("cameraIds: %zu of %zu cameras selected", cams.size(), ds.cameras.size()));
        if (region)
            em.log(fmt("region: expanded box [%g, %g, %g] .. [%g, %g, %g]; %zu cameras see it", lo[0], lo[1], lo[2], hi[0], hi[1],
                       hi[2], cams.size()));
        std::vector<const CameraInfo*> trainCams, evalCams;
        for (auto* c : cams) (c->holdout ? evalCams : trainCams).push_back(c);
        if (trainCams.empty()) throw Error("no training cameras left after cameraIds/region selection");

        FeatureContext fc;
        for (auto* c : trainCams) { fc.hasDepth |= !c->depth.empty(); fc.hasSky |= !c->sky.empty(); }
        std::vector<const CameraInfo*> allCams = trainCams;
        allCams.insert(allCams.end(), evalCams.begin(), evalCams.end());
        fc.exposureLocked = exposureLocked(allCams);
        for (const std::string& note : unsupportedNotes(cfg, fc)) em.log(note);
        const double depthWeight = cfg.effectiveDepthWeight(fc.hasDepth);
        const bool loadDepth = depthWeight > 0 || cfg.normalWeight > 0;

        // Monocular depth priors (trainer.py: no sensor depth, normalWeight > 0, depthPriors auto / on). The depth
        // model (Depth Anything V2, PyTorch) can't run here: the maps a CUDA run cached next to the dataset are used.
        std::map<int64_t, std::vector<float>> priorMaps;
        std::map<int64_t, std::pair<int, int>> priorSize;
        if (!fc.hasDepth && cfg.normalWeight > 0 && (cfg.depthPriors == "auto" || cfg.depthPriors == "on")) {
            std::string dir = (fs::path(ds.root) / "depth-prior").string();
            if (priorCacheValid(dir)) {
                for (auto* c : allCams) {
                    std::vector<float> d;
                    int w = 0, h = 0;
                    if (readNpyFloat2D((fs::path(dir) / (priorCacheKey(*c) + ".npy")).string(), d, w, h)) {
                        priorMaps[c->id] = std::move(d);
                        priorSize[c->id] = {w, h};
                    }
                }
            }
            if (!priorMaps.empty())
                em.log(fmt("depth priors: %zu of %zu photos have a predicted depth map (cached in depth-prior/)", priorMaps.size(),
                           allCams.size()));
            else
                em.log("depth priors: the Vulkan trainer can't predict depth maps itself and the dataset has none cached "
                       "(depth-prior/ from a CUDA run); training without them");
        }

        // Photos, decoded and resized once and kept in RAM (opaque ones packed, 4 bytes / pixel); depth maps at
        // their own (sensor / prior) resolution.
        struct ViewDepth { std::vector<float> d; int w = 0, h = 0; bool prior = false; };
        std::vector<View> views;
        std::vector<ViewDepth> depths(allCams.size());
        views.reserve(allCams.size());
        for (size_t i = 0; i < allCams.size(); i++) {
            if (ctl.cancel) throw Cancelled{};
            views.push_back(loadView(ds, *allCams[i], int(cfg.maxImageSize)));
            views.back().compact();
            const CameraInfo& c = *allCams[i];
            if ((loadDepth || i >= trainCams.size()) && !c.depth.empty()) {   // holdouts: for the depth error log
                depths[i].d = loadDepthF16(ds.resolve(c.depth), c.depthWidth, c.depthHeight);
                depths[i].w = c.depthWidth; depths[i].h = c.depthHeight;
            } else if (priorMaps.count(c.id)) {
                depths[i].d = priorMaps[c.id];
                depths[i].w = priorSize[c.id].first; depths[i].h = priorSize[c.id].second;
                depths[i].prior = true;
            }
            if (i % 20 == 0) em.status({{"message", fmt("loading images %zu/%zu", i + 1, allCams.size())}});
        }
        size_t nTrain = trainCams.size(), nEval = evalCams.size();
        double pixels = 0;
        for (auto& v : views) pixels += double(v.width) * v.height;
        em.log(fmt("%zu train / %zu holdout images (max side %lld), depth supervision weight %g; pixels ≈ %.2f GB kept in RAM",
                   nTrain, nEval, (long long)cfg.maxImageSize, depthWeight, pixels * 4 / 1073741824.0));

        // ---------------------------------------------------------- stage-2b features (trainer.py)
        Extras X;
        X.compact = cfg.optimizerPrecision == "compact";
        X.skyOn = cfg.skybox == "on" || (cfg.skybox == "auto" && fc.hasSky);
        const int skyH = int(cfg.skyboxResolution), skyW = 2 * skyH;
        const SkyBasis skyB = skyBasis(ds.up);
        if (X.skyOn) {
            size_t withMask = 0;
            for (size_t i = 0; i < nTrain; i++) withMask += !views[i].sky.empty();
            em.log(fmt("learned skybox %d×%d at infinity; %zu photos have a sky mask (splats kept out, weight %g)", skyW, skyH,
                       withMask, cfg.skyAlphaWeight));
            const double init[3] = {0.6, 0.65, 0.7};
            X.skyTex.resize(size_t(skyW) * skyH * 3);
            for (size_t i = 0; i < X.skyTex.size(); i++) {
                double c = std::clamp(init[i % 3], 0.02, 0.98);
                X.skyTex[i] = float(std::log(c / (1 - c)));
            }
            X.skyM.assign(X.skyTex.size(), 0.0f);
            X.skyV.assign(X.skyTex.size(), 0.0f);
        }
        const double viewGap = medianNearestViewGap(trainCams, ds.extent);
        const bool transientOn = cfg.transientMasking && viewGap <= cfg.transientMaxViewGap;
        if (cfg.transientMasking && !transientOn)
            em.log(fmt("sparse coverage (photos ~%.0f° apart): not ignoring moving objects — real scenery would be mistaken for them",
                       viewGap));
        const int64_t transientStart = std::max<int64_t>(cfg.transientStart, int64_t(double(cfg.iterations) * 0.3));
        bool useAppearance = cfg.appearanceCorrection;
        if (useAppearance && fc.exposureLocked) {
            useAppearance = false;
            em.log("exposure is the same in every photo: per-photo colour correction off");
        }
        if (cfg.bilateralGrid) {
            useAppearance = false;
            X.gridOn = true;
            X.grid.setup(nTrain, cfg.bilateralGridLR, cfg.bilateralGridTV);
            em.log(fmt("per-photo local colour correction: %zu grids of 16×16×8", nTrain));
        }
        if (useAppearance) {
            X.appOn = true;
            X.app.setup(allCams, cfg.appearanceLR, cfg.appearanceReg);
        }
        if (cfg.poseRefinement) {
            X.poseOn = true;
            X.pose.setup(nTrain, cfg.poseRefinementLR, cfg.poseRefinementReg);
        }
        const bool decoupled = cfg.decoupledSSIM && (X.appOn || X.gridOn);

        // ---------------------------------------------------------- state
        Saved resume;
        bool resuming = !opt.resumeFrom.empty();
        if (resuming) {
            std::string blob;
            resume = loadState(opt.resumeFrom, blob);
            if (resume.trainViews != nTrain || resume.evalViews != nEval)
                throw Error("the dataset changed since this job was paused (different photos); it can't be resumed");
            if (resume.strategy != cfg.strategy) throw Error("the saved state is from another strategy; it can't be resumed");
            std::istringstream xs(blob);
            In xin{xs};
            readExtras(xin, X);
            elapsedBefore = resume.elapsed;
            em.log(fmt("resuming from the state saved at step %lld", (long long)resume.step));
        }
        const double sceneScale = ds.extent;
        HostRng rng(uint64_t(cfg.seed));
        SplatState st;
        if (!resuming) {
            HostRng initRng(uint64_t(cfg.seed) ^ 0x5DEECE66DULL);
            std::vector<float> xyz, rgb;
            initPoints(ds, cfg, region ? lo : nullptr, region ? hi : nullptr, initRng, em, xyz, rgb);
            if (!priorMaps.empty() && cfg.priorInitPoints > 0) {
                // depthprior.py init_points: about priorInitPoints points back-projected (every 2nd pixel) from the
                // training photos' prior maps, spread evenly over them; region filter; up to maxGaussians in total.
                std::vector<size_t> have;
                for (size_t i = 0; i < nTrain; i++) if (depths[i].prior) have.push_back(i);
                size_t per = std::max<size_t>(1, size_t(cfg.priorInitPoints) / std::max<size_t>(have.size(), 1));
                std::vector<float> px, pc;
                for (size_t i : have) {
                    const ViewDepth& vd = depths[i];
                    const CameraInfo& c = *allCams[i];
                    const View& v = views[i];
                    double sx = vd.w / double(c.width), sy = vd.h / double(c.height);
                    double fx = c.fx * sx, fy = c.fy * sy, cx = c.cx * sx, cy = c.cy * sy;
                    std::vector<float> cp, cc;
                    for (int y = 0; y < vd.h; y += 2)
                        for (int x = 0; x < vd.w; x += 2) {
                            float z = vd.d[size_t(y) * size_t(vd.w) + size_t(x)];
                            if (!(z > 0)) continue;
                            double Xc[3] = {(x + 0.5 - cx) / fx * z, (y + 0.5 - cy) / fy * z, z};
                            for (int r = 0; r < 3; r++)
                                cp.push_back(float(c.c2w[r][0] * Xc[0] + c.c2w[r][1] * Xc[1] + c.c2w[r][2] * Xc[2] + c.c2w[r][3]));
                            int ix = std::min(v.width - 1, int((x + 0.5) / vd.w * v.width));
                            int iy = std::min(v.height - 1, int((y + 0.5) / vd.h * v.height));
                            float col[3], a, m;
                            v.pixel(size_t(iy) * size_t(v.width) + size_t(ix), col, a, m);
                            cc.insert(cc.end(), col, col + 3);
                        }
                    size_t cnt = cp.size() / 3;
                    std::vector<size_t> idx(cnt);
                    std::iota(idx.begin(), idx.end(), size_t(0));
                    size_t take = std::min(per, cnt);
                    for (size_t j = 0; j < take; j++) std::swap(idx[j], idx[j + size_t(initRng.uniform() * double(cnt - j)) % (cnt - j)]);
                    for (size_t j = 0; j < take; j++) {
                        size_t q = idx[j];
                        bool in = true;
                        if (region) for (int k = 0; k < 3; k++) in = in && cp[q * 3 + size_t(k)] >= lo[k] && cp[q * 3 + size_t(k)] <= hi[k];
                        if (!in) continue;
                        px.insert(px.end(), &cp[q * 3], &cp[q * 3] + 3);
                        pc.insert(pc.end(), &cc[q * 3], &cc[q * 3] + 3);
                    }
                }
                size_t room = size_t(std::max<int64_t>(0, cfg.maxGaussians - int64_t(xyz.size() / 3)));
                size_t np = std::min(px.size() / 3, room);
                std::vector<size_t> idx(px.size() / 3);
                std::iota(idx.begin(), idx.end(), size_t(0));
                for (size_t j = 0; j < np && np < idx.size(); j++)
                    std::swap(idx[j], idx[j + size_t(initRng.uniform() * double(idx.size() - j)) % (idx.size() - j)]);
                for (size_t j = 0; j < np; j++) {
                    xyz.insert(xyz.end(), &px[idx[j] * 3], &px[idx[j] * 3] + 3);
                    rgb.insert(rgb.end(), &pc[idx[j] * 3], &pc[idx[j] * 3] + 3);
                }
                if (np) em.log(fmt("depth priors: added %zu starting points back-projected from the predicted depth", np));
            }
            st = initSplats(xyz, rgb, float(cfg.initialOpacity), initRng);
        } else {
            st = std::move(resume.s);
            rng.load(resume.rng);
        }
        if (st.n() == 0) throw Error("no splats to train");

        const bool mcmc = cfg.strategy == "mcmc";
        const double lr0 = cfg.lrPosition * sceneScale;
        const double gamma = std::pow(cfg.lrPositionFinal / cfg.lrPosition, 1.0 / double(cfg.iterations));
        const int64_t densifyStop = cfg.effectiveDensifyStop();
        const double opacityReg = cfg.effectiveOpacityReg(), scaleReg = cfg.effectiveScaleReg();
        const double growThreshold = cfg.effectiveGradThreshold();
        const int64_t previewInterval = opt.previewEvery >= 0 ? opt.previewEvery : cfg.previewInterval;
        em.log(fmt("strategy=%s gaussians=%d sceneExtent=%.3f ssim=valid-window rasterize_mode=%s densify %lld..%lld/%lld "
                   "opacityReg=%g scaleReg=%g",
                   cfg.strategy.c_str(), st.n(), sceneScale, cfg.antialiased ? "antialiased" : "classic", (long long)cfg.densifyStart,
                   (long long)densifyStop, (long long)cfg.densifyInterval, opacityReg, scaleReg));

        float fixedBg[3] = {0, 0, 0};
        bool randomBg = cfg.background == "random";
        if (cfg.background == "white") fixedBg[0] = fixedBg[1] = fixedBg[2] = 1;
        const float* evalBg = fixedBg;   // random: black for evaluation (trainer.py eval_bg)

        GpuSplat gs(ctx);
        int64_t startStep = 0;
        gs.compactMoments = X.compact;
        gs.compactSeed = uint32_t(cfg.seed);
        {
            // per splat on the GPU: fp32 params + grads (59 values each) + Adam state (compact.h)
            int state = X.compact ? 11 * 4 + compact::SH_WORDS * 4 : 59 * 8;
            em.log(fmt("optimizerPrecision=%s%s: ~%d B/splat for params+grads+Adam state (state %d B; fp32 Adam would be %d B)",
                       cfg.optimizerPrecision.c_str(), X.compact ? " (8-bit SH-rest moments, bf16 others)" : "", 2 * 59 * 4 + state,
                       state, 59 * 8));
        }
        gs.setState(st.p, &st.m1, &st.m2, resuming ? int(resume.adamSteps) : 0);
        Buffer grad2dBuf, countBuf;
        auto resetStats = [&](const std::vector<float>* g, const std::vector<float>* c) {
            ctx.destroy(grad2dBuf);
            ctx.destroy(countBuf);
            grad2dBuf = ctx.createBuffer(size_t(gs.n) * 4);
            countBuf = ctx.createBuffer(size_t(gs.n) * 4);
            if (g && c && g->size() == size_t(gs.n) && c->size() == size_t(gs.n)) {
                ctx.upload(grad2dBuf, g->data(), g->size() * 4);
                ctx.upload(countBuf, c->data(), c->size() * 4);
            } else {
                ctx.begin();
                ctx.fill(grad2dBuf, 0);
                ctx.fill(countBuf, 0);
                ctx.submitWait();
            }
        };
        if (!mcmc) resetStats(resuming ? &resume.grad2d : nullptr, resuming ? &resume.count : nullptr);

        // skybox panorama, its gradient and Adam moments on the GPU (trainer.py Skybox: Adam lr 0.01, eps 1e-15)
        Buffer skyTexB, skyGradB, skyMB, skyVB;
        if (X.skyOn) {
            size_t bytes = X.skyTex.size() * 4;
            skyTexB = ctx.createBuffer(bytes); skyGradB = ctx.createBuffer(bytes);
            skyMB = ctx.createBuffer(bytes); skyVB = ctx.createBuffer(bytes);
            ctx.upload(skyTexB, X.skyTex.data(), bytes);
            ctx.upload(skyMB, X.skyM.data(), bytes);
            ctx.upload(skyVB, X.skyV.data(), bytes);
        }
        if (transientOn) gs.setTransientState(X.tThreshold, X.tHas);
        auto syncSky = [&]() {   // GPU → host copies (save, result)
            if (!X.skyOn) return;
            X.skyTex = gs.read<float>(skyTexB, X.skyTex.size());
            X.skyM = gs.read<float>(skyMB, X.skyM.size());
            X.skyV = gs.read<float>(skyVB, X.skyV.size());
        };
        auto setSky = [&](const double w2c[4][4]) {
            gs.feat.sky = true;
            gs.feat.skyTex = &skyTexB; gs.feat.skyGrad = &skyGradB; gs.feat.skyM = &skyMB; gs.feat.skyV = &skyVB;
            gs.feat.skyW = uint32_t(skyW); gs.feat.skyH = uint32_t(skyH);
            skyRotation(skyB, w2c, gs.feat.skyRot);
        };

        RenderOpts base;
        base.antialiased = cfg.antialiased;
        base.ssimWeight = float(cfg.ssimWeight);
        base.absGrad = !mcmc && cfg.absGrad;
        base.hasDepth = false;
        base.ssimValid = true;   // fused-ssim padding="valid" (trainer.py _ssim_fn)

        double scenUp[3];
        if (ds.hasGravityUp) std::copy(ds.gravityUp, ds.gravityUp + 3, scenUp);
        else std::copy(ds.up, ds.up + 3, scenUp);

        auto download = [&](SplatState& s) {
            gs.getParams(s.p);
            gs.getMoments(s.m1, s.m2);
        };
        auto upload = [&](const SplatState& s) { gs.setState(s.p, &s.m1, &s.m2, gs.adamSteps); };

        // trainer.py evaluate(): holdout PSNR of the clamped render over the black (or fixed) background — or the
        // learned sky — with the photo's exposure-seeded colour correction (no bilateral grid), masked pixels scored
        // against the render itself, averaged over the holdout photos.
        double evalDepthErr = 0, evalDepthCnt = 0;
        auto evaluate = [&](int shDeg, bool dump = false) -> double {
            if (nEval == 0) return NAN;
            double sum = 0;
            evalDepthErr = evalDepthCnt = 0;
            for (size_t k = nTrain; k < views.size(); k++) {
                const View& v = views[k];
                RenderOpts ro = base;
                ro.shDegree = shDeg;
                ro.hasDepth = false;
                std::copy(evalBg, evalBg + 3, ro.bg);
                gs.feat = LossFeatures();
                if (X.skyOn) {
                    setSky(v.w2c);
                    ro.bg[0] = ro.bg[1] = ro.bg[2] = 0;
                }
                if (X.appOn) {
                    gs.feat.corr = LossFeatures::Affine;
                    gs.setCorrection(X.app.p[k].data(), 12);
                }
                gs.setCamera(v.camera(), ro);
                gs.forwardComposite();
                gs.feat = LossFeatures();
                std::vector<float> im = gs.image();
                size_t hw = size_t(v.width) * v.height;
                if (!depths[k].d.empty()) {   // not in trainer.py: how far the rendered depth is from the photo's map
                    const ViewDepth& vd = depths[k];
                    std::vector<float> D = gs.read<float>(gs.img, 4 * hw), T = gs.read<float>(gs.finalT, hw);
                    for (int y = 0; y < v.height; y++)
                        for (int x = 0; x < v.width; x++) {
                            size_t p = size_t(y) * size_t(v.width) + size_t(x);
                            float d = vd.d[size_t(((2 * y + 1) * vd.h) / (2 * v.height)) * size_t(vd.w) + size_t(((2 * x + 1) * vd.w) / (2 * v.width))];
                            if (!(d > 0)) continue;
                            evalDepthErr += std::abs(D[3 * hw + p] / std::max(1.0f - T[p], 1e-10f) - d);
                            evalDepthCnt += 1;
                        }
                }
                double se = 0;
                for (size_t p = 0; p < hw; p++) {
                    float c[3], a, m;
                    v.pixel(p, c, a, m);
                    for (int ch = 0; ch < 3; ch++) {
                        double pred = std::clamp(double(im[size_t(ch) * hw + p]), 0.0, 1.0);
                        double gt = c[ch] * a + (1 - a) * evalBg[ch];
                        gt = gt * m + pred * (1 - m);
                        se += (pred - gt) * (pred - gt);
                    }
                }
                if (dump && !opt.dumpEval.empty()) {
                    fs::create_directories(opt.dumpEval);
                    for (int which = 0; which < 2; which++) {
                        std::ofstream f((fs::path(opt.dumpEval) / fmt("eval%zu_%s.ppm", k - nTrain, which ? "target" : "render")).string(),
                                        std::ios::binary);
                        f << "P6\n" << v.width << " " << v.height << "\n255\n";
                        for (size_t p = 0; p < hw; p++) {
                            float c[3], a, m;
                            v.pixel(p, c, a, m);
                            for (int ch = 0; ch < 3; ch++) {
                                double val = which ? c[ch] : std::clamp(double(im[size_t(ch) * hw + p]), 0.0, 1.0);
                                f.put(char(uint8_t(std::lround(val * 255))));
                            }
                        }
                    }
                }
                double mse = se / double(hw * 3);
                sum += -10 * std::log10(std::max(mse, 1e-10));
            }
            return sum / double(nEval);
        };

        auto writeCheckpoint = [&](int64_t step) {
            if (opt.jobDir.empty()) return;
            Params p;
            gs.getParams(p);
            size_t n = writeGaussianPly((fs::path(opt.jobDir) / "checkpoint.ply").string(), p, int(cfg.shDegree), scenUp);
            em.log(fmt("step %lld: checkpoint written (%zu splats)", (long long)step, n));
            em.emit({{"type", "checkpoint"}, {"step", step}});
        };

        // ---------------------------------------------------------- loop state
        std::vector<uint32_t> order;
        double psnrEma = NAN, lossEma = NAN, lastEval = NAN;
        int64_t previewStep = -1;
        bool capLogged = false;
        if (resuming) {
            startStep = resume.step;
            order = resume.order;
            psnrEma = resume.psnrEma; lossEma = resume.lossEma; lastEval = resume.evalPsnr;
            previewStep = resume.previewStep;
            capLogged = resume.capLogged;
            em.status({{"step", startStep}, {"numGaussians", gs.n}, {"psnr", numOrNull(psnrEma)}, {"evalPsnr", numOrNull(lastEval)},
                       {"loss", numOrNull(lossEma)}});
        }
        resume = Saved();   // free the host copy
        em.status({{"state", "training"}, {"message", "training"}, {"step", startStep}, {"numGaussians", gs.n}});
        auto trainStart = Clock::now();
        auto lastReport = Clock::time_point();
        bool pauseWarned = false;
        std::map<std::pair<size_t, int>, std::vector<uint32_t>> scaledCache;   // progressive resolution targets
        double lastScale = 1.0;
        int shDeg = 0;
        double sumForward = 0, sumBackward = 0, sumDensify = 0;

        // Everything the loop needs to carry on exactly where it stopped (trainer.py save_state).
        auto saveTo = [&](const std::string& path, int64_t step) {
            Saved s;
            s.trainViews = uint32_t(nTrain); s.evalViews = uint32_t(nEval);
            s.step = step; s.adamSteps = gs.adamSteps;
            download(s.s);
            if (!mcmc) {
                s.grad2d = gs.read<float>(grad2dBuf, size_t(gs.n));
                s.count = gs.read<float>(countBuf, size_t(gs.n));
            }
            s.order = order;
            s.rng = rng.save();
            s.strategy = cfg.strategy;
            s.psnrEma = psnrEma; s.lossEma = lossEma; s.evalPsnr = lastEval;
            s.elapsed = elapsedBefore + seconds(tStart, Clock::now());
            s.previewStep = previewStep;
            s.capLogged = capLogged;
            syncSky();
            if (transientOn) {
                GpuSplat::TransientState t = gs.transientState();
                X.tThreshold = t.threshold;
                X.tHas = t.hasThreshold;
            }
            s.x = X;
            saveState(path, s);
        };
        // valid depth pixels of view vi at a render size (trainer.py: (depth_gt > 0) & isfinite, the map resized
        // nearest to the photo, then to the progressive scale), as depth_normal.comp samples it
        std::map<std::tuple<size_t, int, int>, double> validCache;
        auto validDepth = [&](size_t vi, int rw, int rh) -> double {
            auto key = std::make_tuple(vi, rw, rh);
            auto it = validCache.find(key);
            if (it != validCache.end()) return it->second;
            const ViewDepth& vd = depths[vi];
            const View& v = views[vi];
            float sx = float(v.width) / float(rw), sy = float(v.height) / float(rh);
            double cnt = 0;
            for (int y = 0; y < rh; y++) {
                uint32_t yv = rh == v.height ? uint32_t(y) : std::min(uint32_t(std::floor(float(y) * sy)), uint32_t(v.height - 1));
                uint32_t ys = ((2 * yv + 1) * uint32_t(vd.h)) / (2 * uint32_t(v.height));
                for (int x = 0; x < rw; x++) {
                    uint32_t xv = rw == v.width ? uint32_t(x) : std::min(uint32_t(std::floor(float(x) * sx)), uint32_t(v.width - 1));
                    uint32_t xs = ((2 * xv + 1) * uint32_t(vd.w)) / (2 * uint32_t(v.width));
                    cnt += vd.d[size_t(ys) * size_t(vd.w) + xs] > 0;
                }
            }
            return validCache[key] = cnt;
        };

        for (int64_t step = startStep; step < cfg.iterations; step++) {
            if (ctl.cancel) throw Cancelled{};
            if (step == opt.pauseAt && step != startStep) ctl.pause = true;
            if (ctl.checkpoint.exchange(false)) writeCheckpoint(step);
            if (step == opt.snapshotAt) {   // tests: the state a pause would save, without stopping
                saveTo((fs::path(opt.jobDir.empty() ? "." : opt.jobDir) / "snapshot.vkstate").string(), step);
                em.emit({{"type", "snapshot"}, {"step", step}});
            }
            if (ctl.pause) {
                std::string rp = opt.resumePath();
                if (rp.empty()) {
                    if (!pauseWarned) em.log("pause needs --job-dir or --resume (where to save the state); carrying on");
                    pauseWarned = true;
                } else {
                    writeCheckpoint(step);   // a paused job always has a downloadable checkpoint
                    em.status({{"message", "saving the training state"}});
                    saveTo(rp, step);
                    em.log(fmt("paused at step %lld: training state saved", (long long)step));
                    em.emit({{"type", "paused"}, {"step", step}, {"resumeFile", rp}});
                    return EXIT_PAUSED;
                }
            }
            if (order.empty()) {   // trainer.py: a fresh shuffled pass over the training photos, drawn from the end
                order.resize(nTrain);
                std::iota(order.begin(), order.end(), 0u);
                for (size_t i = nTrain; i > 1; i--) {
                    size_t j = size_t(rng.uniform() * double(i));
                    std::swap(order[i - 1], order[std::min(j, i - 1)]);
                }
            }
            uint32_t vi = order.back();
            order.pop_back();
            const View& v = views[vi];
            shDeg = int(std::min<int64_t>(step / cfg.shDegreeInterval, cfg.shDegree));

            float bg[3];
            if (X.skyOn) std::fill(bg, bg + 3, 0.0f);   // the backdrop is the sky (no random background drawn)
            else if (randomBg) for (float& b : bg) b = float(rng.uniform());
            else std::copy(fixedBg, fixedBg + 3, bg);
            RenderOpts ro = base;
            ro.shDegree = shDeg;
            std::copy(bg, bg + 3, ro.bg);
            Camera cam = v.camera();
            std::vector<uint32_t> scaledTarget;
            const std::vector<uint32_t>* target = &v.packed;
            double sc = cfg.progressiveResolution ? progressiveScale(step, densifyStop) : 1.0;
            if (sc != lastScale) { scaledCache.clear(); lastScale = sc; }
            if (sc < 1.0) {
                auto key = std::make_pair(size_t(vi), int(sc * 4));
                auto it = scaledCache.find(key);
                if (v.alpha.empty() && it != scaledCache.end()) {   // opaque photo: the scaled target doesn't depend on bg
                    int nh = std::max(8, int(std::nearbyint(v.height * sc))), nw = std::max(8, int(std::nearbyint(v.width * sc)));
                    double sy = double(nh) / v.height, sx = double(nw) / v.width;
                    cam.width = nw; cam.height = nh;
                    cam.fx = v.fx * sx; cam.fy = v.fy * sy;
                    cam.cx = (v.cx + 0.5) * sx - 0.5; cam.cy = (v.cy + 0.5) * sy - 0.5;
                    target = &it->second;
                } else {
                    scaledTarget = packTargetScaled(v, bg, sc, cam);
                    if (v.alpha.empty()) target = &(scaledCache[key] = std::move(scaledTarget));
                    else target = &scaledTarget;
                }
            } else if (!v.alpha.empty()) {
                scaledTarget = packTarget(v, bg);
                target = &scaledTarget;
            }

            // ---- the stage-2b parts of this step's loss
            gs.feat = LossFeatures();
            const int rw = cam.width, rh = cam.height;
            if (X.poseOn && step >= cfg.poseRefinementStart) {   // trainer.py: w2c = pose_refiner.apply(view_idx, w2c)
                refinedW2c(v.w2c, X.pose.row(vi), cam.w2c);
                gs.feat.pose = true;
            }
            std::vector<uint32_t> aux;
            if (X.skyOn) {
                setSky(cam.w2c);
                if (!v.sky.empty() && cfg.skyAlphaWeight > 0) {
                    // sky mask at this step's resolution (area-averaged like trainer.py resized()); penalty normaliser Σ m
                    std::vector<float> m(v.sky.size());
                    for (size_t p = 0; p < m.size(); p++) m[p] = v.sky[p] / 255.0f;
                    if (rw != v.width || rh != v.height) m = areaResize(m, v.width, v.height, 1, rw, rh);
                    aux.resize(m.size());
                    double ms = 0;
                    for (size_t p = 0; p < m.size(); p++) {
                        ms += m[p];
                        aux[p] = uint32_t(std::clamp(std::lround(m[p] * 255.0f), 0L, 255L));
                    }
                    gs.feat.skyMask = true;
                    gs.feat.skyPenalty = float(-cfg.skyAlphaWeight / std::max(ms, 1.0));
                }
            }
            if (X.appOn) {
                gs.feat.corr = LossFeatures::Affine;
                gs.setCorrection(X.app.p[vi].data(), 12);
            } else if (X.gridOn) {
                gs.feat.corr = LossFeatures::Grid;
                gs.setCorrection(X.grid.grids[vi].data(), size_t(BILAGRID_SIZE));
            }
            gs.feat.decoupledSSIM = decoupled;
            if (transientOn) {
                gs.feat.transient = true;
                gs.feat.transientActive = step >= transientStart;
                gs.feat.transientSensitivity = float(cfg.transientSensitivity);
            }
            const ViewDepth& vd = depths[vi];
            if (!vd.d.empty()) {
                double dw = vd.prior ? cfg.monoDepthWeight : depthWeight;   // predicted maps: their own weight
                double cnt = dw > 0 ? validDepth(vi, rw, rh) : 0;
                if (dw > 0 && cnt > 0) gs.feat.depthScale = float(dw / cnt);
                if (cfg.normalWeight > 0 && step >= cfg.normalStart) {
                    gs.feat.normalScale = float(cfg.normalWeight / (double(rw) * rh));
                    gs.feat.normalK = std::max(1, int(std::nearbyint(double(rw) / std::max(vd.w, 1))));
                }
                gs.feat.edgeAware = cfg.depthEdgeAware;
                if (gs.feat.anyDepth()) gs.setDepthTarget(vd.d.data(), vd.w, vd.h, v.width, v.height);
            }
            ro.hasDepth = gs.feat.anyDepth();

            auto t0 = Clock::now();
            gs.setCamera(cam, ro);
            gs.setTarget(*target);
            if (!aux.empty()) gs.setAux(aux);
            double sqErr = 0;
            double loss = gs.forwardLoss(nullptr, nullptr, &sqErr);
            auto t1 = Clock::now();

            // DefaultStrategy safety cap: no growth at maxGaussians (pruning continues)
            AdcParams adc;
            if (!mcmc) {
                bool atCap = gs.n >= cfg.maxGaussians;
                adc.growGrad2d = atCap ? INFINITY : growThreshold;
                adc.pruneOpa = cfg.pruneOpacity;
                adc.sceneScale = sceneScale;
                adc.resetEvery = cfg.opacityResetInterval;
                if (atCap && !capLogged) {
                    em.log(fmt("step %lld: reached maxGaussians=%lld; growth paused", (long long)step, (long long)cfg.maxGaussians));
                    capLogged = true;
                }
            }
            const bool refineNow = step < densifyStop && step > cfg.densifyStart && step % cfg.densifyInterval == 0;
            std::array<AdamGroup, 5> groups;
            float lrPos = float(lr0 * std::pow(gamma, double(step)));
            groups[0] = {lrPos, lrPos, 1, 1};
            groups[1] = {float(cfg.lrScale), float(cfg.lrScale), 1, 1};
            groups[2] = {float(cfg.lrRotation), float(cfg.lrRotation), 1, 1};
            groups[3] = {float(cfg.lrOpacity), float(cfg.lrOpacity), 1, 1};
            groups[4] = {float(cfg.lrColor), float(cfg.lrSHRest), 48, 3};
            GpuSplat::StepExtras ex;
            double invN = 1.0 / double(gs.n);
            ex.wOp = float(opacityReg * invN);
            ex.wScale = float(scaleReg * invN / 3.0);
            ex.wFlat = float(cfg.flatnessReg * invN);
            ex.wAniso = float(cfg.anisotropyReg * invN);
            ex.maxRatio = float(cfg.anisotropyMaxRatio);
            if (!mcmc && step < densifyStop) {
                ex.grad2d = &grad2dBuf;
                ex.count = &countBuf;
                ex.absGrad = cfg.absGrad;
            }
            // MCMC noise uses the position LR after the scheduler step (trainer.py: pos_sched.get_last_lr())
            float noiseScaler = mcmc ? float(lr0 * std::pow(gamma, double(step + 1)) * cfg.noiseLR) : 0.0f;
            if (mcmc && !refineNow) {
                ex.noiseScaler = noiseScaler;
                ex.noiseSeed = uint32_t(cfg.seed);
                ex.noiseStep = uint32_t(step);
            }
            if (X.skyOn) {
                ex.skyLr = 0.01f;
                ex.skySteps = int(++X.skySteps);
            }
            gs.backwardStep(groups, ex);
            // per-photo models: their optimisers step on the host with the gradients the GPU accumulated
            if (X.appOn) X.app.step(vi, gs.readCorrGrad());
            else if (X.gridOn) {
                X.grid.step(vi, gs.readCorrGrad());
                if ((step + 1) % 50 == 0) X.grid.pinMean();
            }
            if (gs.feat.pose) {
                double g15[15], gd[9];
                gs.readPoseGrad(g15);
                poseChain(v.w2c, X.pose.row(vi), g15, gd);
                X.pose.step(vi, gd);
            }
            double stepOutliers = -1;
            if (transientOn && step >= transientStart) {
                GpuSplat::TransientState ts = gs.transientState();
                stepOutliers = double(ts.outliers) / (double(rw) * rh);
                X.tFraction = 0.9 * X.tFraction + 0.1 * stepOutliers;
            }
            gs.feat = LossFeatures();
            auto t2 = Clock::now();

            // ------------------------------------------------ strategy (after the optimiser step, as gsplat)
            if (mcmc && refineNow) {
                SplatState s;
                download(s);
                McmcParams mp;
                mp.capMax = cfg.maxGaussians;
                mp.minOpacity = cfg.pruneOpacity;
                mcmcRefine(s, mp, rng);
                mcmcNoise(s, noiseScaler, uint32_t(cfg.seed), uint32_t(step));
                if (cfg.longAxisSplit) spreadCloneGroups(s);
                upload(s);
            } else if (!mcmc && step < densifyStop) {
                bool reset = opt.opacityReset && step % cfg.opacityResetInterval == 0 && step > 0;
                if (refineNow || reset) {
                    SplatState s;
                    download(s);
                    if (refineNow) {
                        std::vector<float> g = gs.read<float>(grad2dBuf, size_t(gs.n)), c = gs.read<float>(countBuf, size_t(gs.n));
                        adcRefine(s, g, c, adc, step, rng);
                    }
                    // gsplat 1.5's DefaultStrategy writes `step % reset_every == 0 & step > 0`, which Python parses as
                    // a chained comparison that is always false: the CUDA trainer never resets opacities. Matched by
                    // default; --opacity-reset turns the textbook reset (to 2 × pruneOpacity) on.
                    if (reset) resetOpacity(s, cfg.pruneOpacity * 2.0);
                    upload(s);
                    if (refineNow) resetStats(nullptr, nullptr);   // gsplat zeroes grad2d / count after a refinement
                }
            }
            auto t3 = Clock::now();
            sumForward += seconds(t0, t1);
            sumBackward += seconds(t1, t2);
            sumDensify += seconds(t2, t3);
            if (gs.n == 0) throw Error("every splat was pruned");

            // ------------------------------------------------ bookkeeping (trainer.py, every 10 steps)
            int64_t cur = step + 1;
            if (opt.traceEvery > 0 && cur % opt.traceEvery == 0) {
                json tr = {{"type", "trace"}, {"step", cur}, {"loss", loss}, {"psnr", -10 * std::log10(std::max(sqErr, 1e-10))},
                           {"numGaussians", gs.n}, {"camera", allCams[vi]->id}};
                if (stepOutliers >= 0) tr["outliers"] = stepOutliers;   // transient masking: fraction dropped this step
                em.emit(tr);
            }
            if (cur % 10 == 0 || cur == cfg.iterations) {
                lossEma = std::isnan(lossEma) ? loss : 0.9 * lossEma + 0.1 * loss;
                double p = -10 * std::log10(std::max(sqErr, 1e-10));
                psnrEma = std::isnan(psnrEma) ? p : 0.9 * psnrEma + 0.1 * p;
            }
            if (cfg.evalInterval && nEval && (cur % cfg.evalInterval == 0 || cur == cfg.iterations)) {
                lastEval = evaluate(shDeg, cur == cfg.iterations);
                em.log(fmt("step %lld: holdout PSNR %.2f dB", (long long)cur, lastEval));
                if (evalDepthCnt > 0)
                    em.log(fmt("step %lld: holdout depth error %.4f (mean |rendered depth − depth map|)", (long long)cur,
                               evalDepthErr / evalDepthCnt));
            }
            if (previewInterval > 0 && !opt.jobDir.empty() && cur % previewInterval == 0 && cur < cfg.iterations) {
                Params p;
                gs.getParams(p);
                writePreviewSplat((fs::path(opt.jobDir) / "preview.splat").string(), p);
                previewStep = cur;
                em.emit({{"type", "preview"}, {"step", cur}});
            }
            auto now = Clock::now();
            if (seconds(lastReport, now) > 1.0 || cur == cfg.iterations || lastReport == Clock::time_point()) {
                lastReport = now;
                double elapsed = elapsedBefore + seconds(tStart, now);
                double rate = seconds(trainStart, now) / double(std::max<int64_t>(1, cur - startStep));
                em.status({{"step", cur}, {"totalSteps", cfg.iterations}, {"loss", numOrNull(lossEma)}, {"psnr", numOrNull(psnrEma)},
                           {"evalPsnr", numOrNull(lastEval)}, {"numGaussians", gs.n}, {"elapsedSeconds", elapsed},
                           {"etaSeconds", rate * double(cfg.iterations - cur)}, {"message", "training"}});
                if (cur % 100 == 0 || cur == cfg.iterations)
                    em.log(fmt("step %lld/%lld loss %.4f psnr %.2f gaussians %d sh %d", (long long)cur, (long long)cfg.iterations,
                               std::isnan(lossEma) ? 0.0 : lossEma, std::isnan(psnrEma) ? 0.0 : psnrEma, gs.n, shDeg));
            }
        }

        // ---------------------------------------------------------- result
        Params p;
        gs.getParams(p);
        std::vector<uint8_t> keep;
        if (region) {   // export only the splats inside the expanded box (export_ply's region_box)
            keep.resize(size_t(p.n));
            for (int i = 0; i < p.n; i++) {
                bool in = true;
                for (int k = 0; k < 3; k++) in = in && p.pos[size_t(i) * 3 + k] >= lo[k] && p.pos[size_t(i) * 3 + k] <= hi[k];
                keep[size_t(i)] = in;
            }
        }
        size_t finalN = writeGaussianPly(opt.outPath, p, int(cfg.shDegree), scenUp, region ? &keep : nullptr);
        std::string rp = opt.resumePath();
        if (!rp.empty()) { std::error_code ec; fs::remove(rp, ec); }   // finished: the saved state is no longer needed
        double elapsed = elapsedBefore + seconds(tStart, Clock::now());
        double trainSec = seconds(trainStart, Clock::now());
        int64_t steps = cfg.iterations - startStep;
        if (region) em.log(fmt("region: %zu of %d splats inside the expanded box written", finalN, p.n));
        json featureStats = json::object();
        if (X.skyOn) {   // uncropped, next to the PLY (trainer.py: the Mac merger crops with open outer faces)
            syncSky();
            std::string png = writeSkySidecars(opt.outPath, X.skyTex, skyW, skyH, skyB);
            em.log("skybox saved: " + fs::path(png).filename().string());
        }
        if (X.poseOn) {
            double tr = 0, ang = 0;
            for (size_t i = 0; i < nTrain; i++) {
                const double* d = X.pose.row(i);
                tr += std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
                double a[6] = {d[3] + 1, d[4], d[5], d[6], d[7] + 1, d[8]}, R[3][3];
                rot6d(a, R);
                ang += std::acos(std::clamp((R[0][0] + R[1][1] + R[2][2] - 1) / 2, -1.0, 1.0)) * 180 / M_PI;
            }
            featureStats["poseTranslation"] = tr / double(nTrain);
            featureStats["poseRotationDeg"] = ang / double(nTrain);
            em.log(fmt("pose refinement: mean correction %.4g (translation), %.3f° (rotation) over %zu photos", tr / double(nTrain),
                       ang / double(nTrain), nTrain));
        }
        if (transientOn) {
            featureStats["transientFraction"] = X.tFraction;
            em.log(fmt("transient masking: %.1f %% of the pixels ignored as moving objects (recent average)", 100 * X.tFraction));
        }
        if (evalDepthCnt > 0) featureStats["evalDepthError"] = evalDepthErr / evalDepthCnt;
        em.log(fmt("time per step: forward + loss %.1f ms, backward + Adam %.1f ms, densification %.1f ms (mean over %lld steps)",
                   1e3 * sumForward / double(std::max<int64_t>(steps, 1)), 1e3 * sumBackward / double(std::max<int64_t>(steps, 1)),
                   1e3 * sumDensify / double(std::max<int64_t>(steps, 1)), (long long)steps));
        em.log(fmt("finished in %.1fs, %zu gaussians written to %s", elapsed, finalN, fs::path(opt.outPath).filename().string().c_str()));
        em.emit({{"type", "done"}, {"numGaussians", finalN}, {"psnr", numOrNull(std::isnan(lastEval) ? psnrEma : lastEval)},
                 {"evalPsnr", numOrNull(lastEval)}, {"trainPsnr", numOrNull(psnrEma)}, {"loss", numOrNull(lossEma)},
                 {"elapsedSeconds", elapsed}, {"trainSeconds", trainSec}, {"steps", steps},
                 {"stepsPerSecond", trainSec > 0 ? double(steps) / trainSec : 0.0},
                 {"previewStep", previewStep >= 0 ? json(previewStep) : json(nullptr)},
                 {"msPerStep", {{"forwardLoss", 1e3 * sumForward / double(std::max<int64_t>(steps, 1))},
                                {"backwardAdam", 1e3 * sumBackward / double(std::max<int64_t>(steps, 1))},
                                {"densify", 1e3 * sumDensify / double(std::max<int64_t>(steps, 1))}}},
                 {"features", featureStats}});
        ctx.destroy(grad2dBuf);
        ctx.destroy(countBuf);
        for (Buffer* b : {&skyTexB, &skyGradB, &skyMB, &skyVB}) ctx.destroy(*b);
        return EXIT_DONE;
    } catch (const Cancelled&) {
        em.emit({{"type", "cancelled"}});
        return EXIT_CANCELLED;
    } catch (const std::exception& e) {
        em.emit({{"type", "error"}, {"message", e.what()}});
        return EXIT_ERROR;
    }
}

}  // namespace vkt
