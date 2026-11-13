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

#include "dataset.h"
#include "gpu_splat.h"
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

constexpr uint32_t RESUME_VERSION = 1;

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
    std::ifstream f;
    template <class T> T get() {
        T v{};
        if (!f.read(reinterpret_cast<char*>(&v), sizeof(T))) throw Error("the saved training state is truncated");
        return v;
    }
    template <class T> void vec(std::vector<T>& v) {
        uint64_t n = get<uint64_t>();
        if (n > (uint64_t(1) << 36)) throw Error("the saved training state is corrupt");
        v.resize(size_t(n));
        if (!f.read(reinterpret_cast<char*>(v.data()), std::streamsize(n * sizeof(T)))) throw Error("the saved training state is truncated");
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
        if (!o.f) throw Error("writing " + tmp + " failed");
    }
    fs::rename(tmp, path);   // atomic: a crash mid-save leaves the previous state (or none), never half a file
}

Saved loadState(const std::string& path) {
    In in;
    in.f.open(path, std::ios::binary);
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

        // Photos, decoded and resized once and kept in RAM (opaque ones packed, 4 bytes / pixel).
        std::vector<View> views;
        views.reserve(allCams.size());
        for (size_t i = 0; i < allCams.size(); i++) {
            if (ctl.cancel) throw Cancelled{};
            views.push_back(loadView(ds, *allCams[i], int(cfg.maxImageSize)));
            views.back().compact();
            if (i % 20 == 0) em.status({{"message", fmt("loading images %zu/%zu", i + 1, allCams.size())}});
        }
        size_t nTrain = trainCams.size(), nEval = evalCams.size();
        double pixels = 0;
        for (auto& v : views) pixels += double(v.width) * v.height;
        em.log(fmt("%zu train / %zu holdout images (max side %lld); pixels ≈ %.2f GB kept in RAM", nTrain, nEval,
                   (long long)cfg.maxImageSize, pixels * 4 / 1073741824.0));

        // ---------------------------------------------------------- state
        Saved resume;
        bool resuming = !opt.resumeFrom.empty();
        if (resuming) {
            resume = loadState(opt.resumeFrom);
            if (resume.trainViews != nTrain || resume.evalViews != nEval)
                throw Error("the dataset changed since this job was paused (different photos); it can't be resumed");
            if (resume.strategy != cfg.strategy) throw Error("the saved state is from another strategy; it can't be resumed");
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

        // trainer.py evaluate(): holdout PSNR of the clamped render over the black (or fixed) background,
        // masked pixels scored against the render itself, averaged over the holdout photos.
        auto evaluate = [&](int shDeg) -> double {
            if (nEval == 0) return NAN;
            double sum = 0;
            for (size_t k = nTrain; k < views.size(); k++) {
                const View& v = views[k];
                RenderOpts ro = base;
                ro.shDegree = shDeg;
                std::copy(evalBg, evalBg + 3, ro.bg);
                gs.setCamera(v.camera(), ro);
                gs.forward();
                std::vector<float> im = gs.image();
                size_t hw = size_t(v.width) * v.height;
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

        for (int64_t step = startStep; step < cfg.iterations; step++) {
            if (ctl.cancel) throw Cancelled{};
            if (step == opt.pauseAt && step != startStep) ctl.pause = true;
            if (ctl.checkpoint.exchange(false)) writeCheckpoint(step);
            if (ctl.pause) {
                std::string rp = opt.resumePath();
                if (rp.empty()) {
                    if (!pauseWarned) em.log("pause needs --job-dir or --resume (where to save the state); carrying on");
                    pauseWarned = true;
                } else {
                    writeCheckpoint(step);   // a paused job always has a downloadable checkpoint
                    em.status({{"message", "saving the training state"}});
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
                    saveState(rp, s);
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
            if (randomBg) for (float& b : bg) b = float(rng.uniform());
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

            auto t0 = Clock::now();
            gs.setCamera(cam, ro);
            gs.setTarget(*target);
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
            gs.backwardStep(groups, ex);
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
            if (opt.traceEvery > 0 && cur % opt.traceEvery == 0)
                em.emit({{"type", "trace"}, {"step", cur}, {"loss", loss}, {"psnr", -10 * std::log10(std::max(sqErr, 1e-10))},
                         {"numGaussians", gs.n}});
            if (cur % 10 == 0 || cur == cfg.iterations) {
                lossEma = std::isnan(lossEma) ? loss : 0.9 * lossEma + 0.1 * loss;
                double p = -10 * std::log10(std::max(sqErr, 1e-10));
                psnrEma = std::isnan(psnrEma) ? p : 0.9 * psnrEma + 0.1 * p;
            }
            if (cfg.evalInterval && nEval && (cur % cfg.evalInterval == 0 || cur == cfg.iterations)) {
                lastEval = evaluate(shDeg);
                em.log(fmt("step %lld: holdout PSNR %.2f dB", (long long)cur, lastEval));
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
                                {"densify", 1e3 * sumDensify / double(std::max<int64_t>(steps, 1))}}}});
        ctx.destroy(grad2dBuf);
        ctx.destroy(countBuf);
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
