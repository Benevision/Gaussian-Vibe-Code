// vktrain-selftest: checks every kernel (stage 1 and the stage-2 training kernels and host strategies) of the Vulkan trainer against a double-precision CPU reference
// (forward) and against central finite differences of that reference (gradients), on small random scenes,
// with each float-atomic path the device supports. Prints a table; exits non-zero on any failure.
//
//   vktrain-selftest [--path auto|native|cas|both] [--device N] [--quick] [--fit-steps N]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "cpu_ref.h"
#include "gpu_splat.h"
#include "compact.h"
#include "models.h"
#include "strategy.h"

using namespace vkt;

namespace {

struct Row {
    std::string kernel, quantity, scene, path;
    double err, tol;
    bool infoOnly;
    bool pass() const { return infoOnly || (std::isfinite(err) && err <= tol); }
};
std::vector<Row> g_rows;
std::string g_path;

void addRow(const std::string& kernel, const std::string& qty, const std::string& scene, double err, double tol, bool info = false) {
    g_rows.push_back({kernel, qty, scene, g_path, err, tol, info});
}

// max_i |a_i − b_i| / max(|b_i|, floorFrac · max_j |b_j|): relative error, with a floor so tiny reference
// values (where float rounding dominates) don't blow it up.
double relErr(const std::vector<double>& a, const std::vector<double>& b, double floorFrac = 1e-2) {
    if (a.empty()) return 0;
    double mx = 0;
    for (double v : b) mx = std::max(mx, std::abs(v));
    double fl = std::max(floorFrac * mx, 1e-30);
    double e = 0;
    for (size_t i = 0; i < a.size(); i++) {
        double d = std::abs(a[i] - b[i]) / std::max(std::abs(b[i]), fl);
        if (!std::isfinite(d)) return INFINITY;
        e = std::max(e, d);
    }
    return e;
}

double maxAbs(const std::vector<double>& a, const std::vector<double>& b) {
    double e = 0;
    for (size_t i = 0; i < a.size(); i++) e = std::max(e, std::abs(a[i] - b[i]));
    return e;
}

struct Rng {
    std::mt19937_64 g;
    explicit Rng(uint64_t s) : g(s) {}
    double uni(double a, double b) { return std::uniform_real_distribution<double>(a, b)(g); }
    double normal() { return std::normal_distribution<double>(0, 1)(g); }
    uint32_t u32() { return uint32_t(g()); }
    int below(int n) { return int(g() % uint64_t(n)); }
};

uint32_t fbits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

struct Case {
    std::string name;
    int n, deg;
    bool aa;
    int w, h;
    double eye[3];
    double roll, fx, fy, cx, cy;
    float bg[3];
    uint64_t seed;
};

// Random splats around the origin; ~10 % behind the camera, ~10 % beside the view (partly / fully off-screen,
// some beyond the Jacobian clamp). Opacities stay ≤ 0.9 so the 0.99 alpha clamp never applies (it has no
// gradient, by design, like the Mac and gsplat).
Params randomScene(Rng& r, int n, const double eye[3], int deg) {
    Params p;
    p.resize(n);
    double back[3] = {eye[0], eye[1], eye[2]};
    double bl = std::sqrt(back[0] * back[0] + back[1] * back[1] + back[2] * back[2]);
    for (int i = 0; i < n; i++) {
        double u = r.uni(0, 1);
        double x[3];
        if (u < 0.8) {
            for (double& v : x) v = r.uni(-1.2, 1.2);
        } else if (u < 0.9) {   // behind (or right next to) the camera
            double d = r.uni(0.05, 1.5);
            for (int k = 0; k < 3; k++) x[k] = eye[k] + back[k] / bl * d + r.uni(-0.4, 0.4);
        } else {   // beside the view frustum
            double a = r.uni(0, 6.2831853), m = r.uni(2.0, 3.5);
            double side[3] = {std::cos(a) * m, std::sin(a) * m, r.uni(-0.5, 0.5)};
            for (int k = 0; k < 3; k++) x[k] = side[k];
        }
        for (int k = 0; k < 3; k++) {
            p.pos[size_t(i) * 3 + k] = float(x[k]);
            p.logScale[size_t(i) * 3 + k] = float(r.uni(-3.2, -1.5));
        }
        double qn = r.uni(0.5, 2.0);
        for (int k = 0; k < 4; k++) p.quat[size_t(i) * 4 + k] = float(r.normal() * qn);
        p.opLogit[size_t(i)] = float(r.uni(-2.5, 2.2));
        for (int k = 0; k < 16; k++)
            for (int c = 0; c < 3; c++)
                p.sh[size_t(i) * 48 + k * 3 + c] = float(k == 0 ? r.uni(-1.2, 1.2) : (k < (deg + 1) * (deg + 1) ? r.uni(-0.3, 0.3) : r.uni(-0.3, 0.3)));
    }
    return p;
}

std::vector<uint32_t> randomTarget(Rng& r, int w, int h) {
    std::vector<uint32_t> t(size_t(w) * h);
    double f[3][3];
    for (auto& row : f) for (double& v : row) v = r.uni(0.05, 0.3);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            uint32_t px = 0xFF000000u;
            for (int c = 0; c < 3; c++) {
                double v = 0.5 + 0.3 * std::sin(f[c][0] * x + f[c][1] * y + c) + r.uni(-0.15, 0.15);
                px |= uint32_t(std::clamp(v, 0.0, 1.0) * 255.0 + 0.5) << (8 * c);
            }
            t[size_t(y) * w + x] = px;
        }
    return t;
}

std::vector<double> targetPlanes(const std::vector<uint32_t>& t, int w, int h) {
    size_t hw = size_t(w) * h;
    std::vector<double> out(3 * hw);
    for (size_t p = 0; p < hw; p++)
        for (int c = 0; c < 3; c++) out[c * hw + p] = double((t[p] >> (8 * c)) & 255u) / 255.0;
    return out;
}

// GPU projection outputs as reference-projection records (to check the later stages in isolation).
std::vector<ref::Proj> gpuProjections(GpuSplat& gs) {
    int n = gs.n;
    auto p2d = gs.read<float>(gs.p2d, size_t(n) * 2);
    auto con = gs.read<float>(gs.conicO, size_t(n) * 4);
    auto rgbd = gs.read<float>(gs.rgbd, size_t(n) * 4);
    auto rects = gs.read<uint32_t>(gs.rects, size_t(n) * 4);
    auto tt = gs.read<uint32_t>(gs.tilesTouched, size_t(n));
    std::vector<ref::Proj> out(static_cast<size_t>(n));
    for (int i = 0; i < n; i++) {
        ref::Proj& p = out[size_t(i)];
        p.valid = tt[size_t(i)] > 0;
        p.tiles = tt[size_t(i)];
        p.mean[0] = p2d[size_t(i) * 2]; p.mean[1] = p2d[size_t(i) * 2 + 1];
        for (int k = 0; k < 3; k++) { p.conic[k] = con[size_t(i) * 4 + k]; p.rgb[k] = rgbd[size_t(i) * 4 + k]; }
        p.op = con[size_t(i) * 4 + 3];
        p.depth = rgbd[size_t(i) * 4 + 3];
        for (int k = 0; k < 4; k++) p.rect[k] = int(rects[size_t(i) * 4 + k]);
    }
    return out;
}

double fdStep(double v) { return 1e-6 * std::max(1.0, std::abs(v)); }

// ------------------------------------------------------------------ sort / scan on their own

void testPrimitives(Context& ctx, GpuSplat& gs, bool quick) {
    Rng r(7);
    double sortErr = 0, scanErr = 0;
    std::vector<uint32_t> sizes = {1, 2, 255, 2048, 2049, 70001};
    if (!quick) sizes.push_back(300007);
    for (uint32_t n : sizes)
        for (int bits : {16, 32}) {
            std::vector<uint32_t> keys(n), vals(n);
            uint32_t mask = bits == 32 ? 0xFFFFFFFFu : 0xFFFFu;
            for (uint32_t i = 0; i < n; i++) {
                keys[i] = (r.u32() % 4 == 0 ? r.u32() % 97 : r.u32()) & mask;   // plenty of duplicates
                vals[i] = i;
            }
            Buffer kb = ctx.createBuffer(size_t(n) * 4), vb = ctx.createBuffer(size_t(n) * 4);
            ctx.upload(kb, keys.data(), size_t(n) * 4);
            ctx.upload(vb, vals.data(), size_t(n) * 4);
            gs.sortPairs(kb, vb, n, bits);
            auto gk = gs.read<uint32_t>(kb, n), gv = gs.read<uint32_t>(vb, n);
            std::vector<uint32_t> idx(n);
            for (uint32_t i = 0; i < n; i++) idx[i] = i;
            std::stable_sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) { return keys[a] < keys[b]; });
            for (uint32_t i = 0; i < n; i++)
                if (gv[i] != idx[i] || gk[i] != keys[idx[i]]) sortErr++;
            ctx.destroy(kb); ctx.destroy(vb);
        }
    std::vector<uint32_t> scanSizes = {1, 1000, 1024, 1025, 300000};
    if (!quick) scanSizes.push_back(1100000);
    for (uint32_t n : scanSizes) {
        std::vector<uint32_t> v(n);
        for (auto& x : v) x = r.u32() % 50;
        Buffer b = ctx.createBuffer(size_t(n) * 4);
        ctx.upload(b, v.data(), size_t(n) * 4);
        gs.exclusiveScan(b, n);
        auto g = gs.read<uint32_t>(b, n);
        uint32_t run = 0;
        for (uint32_t i = 0; i < n; i++) { if (g[i] != run) scanErr++; run += v[i]; }
        ctx.destroy(b);
    }
    double probe = 0;
    for (uint32_t n : {1u, 255u, 4096u, 100003u}) probe += gs.atomicProbe(n);
    addRow("float atomic add (probe)", "wrong sums", "-", probe, 0);
    addRow("radix sort (standalone)", "mismatches, n<=300k, 16/32 bit", "-", sortErr, 0);
    addRow("exclusive scan (standalone)", "mismatches, n<=1.1M", "-", scanErr, 0);
}

// ------------------------------------------------------------------ one scene

void runCase(Context& ctx, GpuSplat& gs, const Case& cs, bool quick) {
    Rng r(cs.seed);
    const double origin[3] = {0, 0, 0};
    Camera cam = lookAt(cs.eye, origin, cs.roll, cs.fx, cs.fy, cs.cx, cs.cy, cs.w, cs.h);
    RenderOpts opt;
    opt.shDegree = cs.deg;
    opt.antialiased = cs.aa;
    for (int k = 0; k < 3; k++) opt.bg[k] = cs.bg[k];
    Params P = randomScene(r, cs.n, cs.eye, cs.deg);
    std::vector<uint32_t> tgt = randomTarget(r, cs.w, cs.h);
    gs.setParams(P);
    gs.setCamera(cam, opt);
    gs.setTarget(tgt);
    const int n = cs.n;
    const size_t hw = size_t(cs.w) * cs.h;
    const std::string& sc = cs.name;

    uint32_t M = gs.forward();
    ref::Cam rc = ref::makeCam(cam, opt);
    ref::ParamsD PD = ref::toDouble(P);
    std::vector<ref::Proj> cpuP = ref::projectAll(PD, rc, true);
    std::vector<ref::Proj> gpuP = gpuProjections(gs);

    // ---- projection forward
    {
        int visMismatch = 0, rectMismatch = 0, both = 0, clamped = 0, behind = 0;
        std::vector<double> g[5], c[5];   // mean, conic, opacity, rgb, depth
        for (int i = 0; i < n; i++) {
            const ref::Proj &a = gpuP[size_t(i)], &b = cpuP[size_t(i)];
            double tz = 0;
            for (int k = 0; k < 3; k++) tz += rc.W[2][k] * PD.g[0][size_t(i) * 3 + k];
            if (tz + rc.t0[2] < 0.01) behind++;
            if (a.valid != b.valid) { visMismatch++; continue; }
            if (!a.valid) continue;
            both++;
            if (b.clamped) clamped++;
            if (std::memcmp(a.rect, b.rect, sizeof(a.rect)) != 0) rectMismatch++;
            for (int k = 0; k < 2; k++) { g[0].push_back(a.mean[k]); c[0].push_back(b.mean[k]); }
            for (int k = 0; k < 3; k++) { g[1].push_back(a.conic[k]); c[1].push_back(b.conic[k]); }
            g[2].push_back(a.op); c[2].push_back(b.op);
            for (int k = 0; k < 3; k++) { g[3].push_back(a.rgb[k]); c[3].push_back(b.rgb[k]); }
            g[4].push_back(a.depth); c[4].push_back(b.depth);
        }
        std::printf("  [%s] %d splats: %d visible, %d behind the camera, %d visible with clamped Jacobian, %u tile instances\n",
                    sc.c_str(), n, both, behind, clamped, M);
        const char* names[5] = {"mean2D", "conic", "opacity (x comp)", "rgb", "depth"};
        for (int k = 0; k < 5; k++) addRow("projection fwd", names[k], sc, relErr(g[k], c[k]), 1e-4);
        addRow("projection fwd", "visibility mismatches / n", sc, double(visMismatch) / n, 0.01);
        addRow("projection fwd", "tile-rect mismatches / visible", sc, both ? double(rectMismatch) / both : 0, 0.01);
    }

    // ---- depth sort, offsets, tile keys, ranges (exact, from the GPU's own projection outputs)
    {
        auto sKeys = gs.read<uint32_t>(gs.depthKeys, size_t(n));
        auto sVals = gs.read<uint32_t>(gs.sortVals, size_t(n));
        auto offs = gs.read<uint32_t>(gs.offsets, size_t(n) + 1);
        std::vector<uint32_t> keys(static_cast<size_t>(n));
        for (int i = 0; i < n; i++) keys[size_t(i)] = gpuP[size_t(i)].valid ? fbits(float(gpuP[size_t(i)].depth)) : 0xFFFFFFFFu;
        std::vector<uint32_t> idx(static_cast<size_t>(n));
        for (int i = 0; i < n; i++) idx[size_t(i)] = uint32_t(i);
        std::stable_sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) { return keys[a] < keys[b]; });
        double sortMis = 0, offMis = 0, tkMis = 0, rgMis = 0;
        for (int i = 0; i < n; i++)
            if (sVals[size_t(i)] != idx[size_t(i)] || sKeys[size_t(i)] != keys[idx[size_t(i)]]) sortMis++;
        std::vector<uint32_t> cpuOff(size_t(n) + 1, 0);
        for (int i = 0; i < n; i++) cpuOff[size_t(i) + 1] = cpuOff[size_t(i)] + gpuP[idx[size_t(i)]].tiles;
        for (int i = 0; i <= n; i++) if (offs[size_t(i)] != cpuOff[size_t(i)]) offMis++;
        std::vector<std::pair<uint32_t, uint32_t>> inst;
        for (int i = 0; i < n; i++) {
            const ref::Proj& p = gpuP[idx[size_t(i)]];
            for (int y = p.rect[1]; y < p.rect[3]; y++)
                for (int x = p.rect[0]; x < p.rect[2]; x++) inst.push_back({uint32_t(y * rc.tilesX + x), idx[size_t(i)]});
        }
        std::stable_sort(inst.begin(), inst.end(), [](auto& a, auto& b) { return a.first < b.first; });
        if (inst.size() != M) tkMis += std::abs(double(inst.size()) - double(M));
        auto tk = gs.read<uint32_t>(gs.tileKeys, M), tv = gs.read<uint32_t>(gs.tileVals, M);
        for (size_t i = 0; i < std::min<size_t>(M, inst.size()); i++)
            if (tk[i] != inst[i].first || tv[i] != inst[i].second) tkMis++;
        size_t nt = size_t(rc.tilesX) * rc.tilesY;
        std::vector<uint32_t> cr(nt * 2, 0);
        for (size_t i = 0; i < inst.size(); i++) {
            uint32_t t = inst[i].first;
            if (i == 0 || inst[i - 1].first != t) cr[t * 2] = uint32_t(i);
            if (i + 1 == inst.size() || inst[i + 1].first != t) cr[t * 2 + 1] = uint32_t(i + 1);
        }
        auto gr = gs.read<uint32_t>(gs.ranges, nt * 2);
        for (size_t i = 0; i < nt * 2; i++) if (gr[i] != cr[i]) rgMis++;
        addRow("depth sort (radix)", "mismatches", sc, sortMis, 0);
        addRow("scan (tile offsets)", "mismatches", sc, offMis, 0);
        addRow("duplicate + tile sort", "mismatches", sc, tkMis, 0);
        addRow("tile ranges", "mismatches", sc, rgMis, 0);
    }

    // ---- rasterise forward (reference fed with the GPU's projection outputs)
    ref::Structure stG;
    std::vector<double> gImgD;
    {
        auto img = gs.read<float>(gs.img, 4 * hw);
        auto fT = gs.read<float>(gs.finalT, hw);
        auto nc = gs.read<uint32_t>(gs.nContrib, hw);
        std::vector<double> cImg, cT;
        std::vector<uint32_t> cN;
        ref::rasterDecide(gpuP, rc, cImg, cT, cN, &stG);
        gImgD.assign(img.begin(), img.end());
        std::vector<double> gRgb(gImgD.begin(), gImgD.begin() + 3 * hw), cRgb(cImg.begin(), cImg.begin() + 3 * hw);
        std::vector<double> gDep(gImgD.begin() + 3 * hw, gImgD.end()), cDep(cImg.begin() + 3 * hw, cImg.end());
        int ncMis = 0;
        for (size_t p = 0; p < hw; p++) if (nc[p] != cN[p]) ncMis++;
        addRow("rasterize fwd", "rgb (max abs)", sc, maxAbs(gRgb, cRgb), 1e-4);
        addRow("rasterize fwd", "depth", sc, relErr(gDep, cDep), 1e-4);
        addRow("rasterize fwd", "final T (max abs)", sc, maxAbs(std::vector<double>(fT.begin(), fT.end()), cT), 1e-4);
        addRow("rasterize fwd", "last-contributor mismatches / px", sc, double(ncMis) / double(hw), 0.01);
    }

    // ---- loss value and dL/dimage (finite differences of the reference loss on the GPU image)
    std::vector<double> tP = targetPlanes(tgt, cs.w, cs.h);
    {
        double gl1, gss;
        double gL = gs.loss(&gl1, &gss);
        double cl1, css;
        double cL = ref::loss(gImgD, tP, cs.w, cs.h, opt.ssimWeight, &cl1, &css);
        addRow("loss (L1 + SSIM)", "loss value", sc, std::abs(gL - cL) / std::abs(cL), 1e-4);
        addRow("loss (L1 + SSIM)", "mean SSIM", sc, std::abs(gss - css) / std::abs(css), 1e-4);
        auto gi = gs.read<float>(gs.gradImg, 4 * hw);
        std::vector<double> ga, fd;
        int samples = quick ? 60 : 200;
        std::vector<double> im = gImgD;
        for (int s = 0; s < samples; s++) {
            size_t j = size_t(r.below(int(3 * hw)));
            double v = im[j], h = 1e-6;
            im[j] = v + h; double lp = ref::loss(im, tP, cs.w, cs.h, opt.ssimWeight);
            im[j] = v - h; double lm = ref::loss(im, tP, cs.w, cs.h, opt.ssimWeight);
            im[j] = v;
            ga.push_back(gi[j]);
            fd.push_back((lp - lm) / (2 * h));
        }
        addRow("loss gradient (ssim/blur/loss_grad)", "dL/dimage vs FD", sc, relErr(ga, fd), 1e-3);
    }

    // ---- rasterise backward: random upstream gradient on all 4 planes; f = Σ G·image over the frozen structure
    std::vector<int> visible;
    for (int i = 0; i < n; i++) if (gpuP[size_t(i)].valid) visible.push_back(i);
    {
        std::vector<float> G(4 * hw);
        for (float& v : G) v = float(r.uni(-1, 1));
        ctx.upload(gs.gradImg, G.data(), G.size() * 4);
        gs.rasterBackward();
        auto acc = gs.read<float>(gs.acc, size_t(n) * 12);
        auto f = [&](const std::vector<ref::Proj>& pr, std::vector<double>* perPixel) {
            std::vector<double> im;
            ref::rasterFrozen(pr, stG, rc, im);
            double s = 0;
            if (perPixel) perPixel->assign(hw, 0.0);
            for (int c = 0; c < 4; c++)
                for (size_t p = 0; p < hw; p++) {
                    double v = double(G[c * hw + p]) * im[c * hw + p];
                    s += v;
                    if (perPixel) (*perPixel)[p] += v;
                }
            return s;
        };
        std::vector<double> ga[7], fd[7];   // mean, conic, opacity, rgb, depth, |mean| (abs), zero check
        std::vector<int> sample = visible;
        std::shuffle(sample.begin(), sample.end(), r.g);
        sample.resize(std::min<size_t>(sample.size(), quick ? 12 : 40));
        std::vector<ref::Proj> pr = gpuP;
        for (int g : sample) {
            for (int q = 0; q < 10; q++) {
                double v = pr[size_t(g)].q(q), h = fdStep(v);
                std::vector<double> pp, pm;
                bool isMean = q < 2;
                pr[size_t(g)].q(q) = v + h; double fp = f(pr, isMean ? &pp : nullptr);
                pr[size_t(g)].q(q) = v - h; double fm = f(pr, isMean ? &pm : nullptr);
                pr[size_t(g)].q(q) = v;
                int grp = q < 2 ? 0 : q < 5 ? 1 : q == 5 ? 2 : q < 9 ? 3 : 4;
                ga[grp].push_back(acc[size_t(g) * 12 + q]);
                fd[grp].push_back((fp - fm) / (2 * h));
                if (isMean) {
                    double s = 0;
                    for (size_t p = 0; p < hw; p++) s += std::abs((pp[p] - pm[p]) / (2 * h));
                    ga[5].push_back(acc[size_t(g) * 12 + 10 + q]);
                    fd[5].push_back(s);
                }
            }
        }
        double nonzero = 0;   // splats that are not drawn must get nothing
        for (int i = 0; i < n; i++)
            if (!gpuP[size_t(i)].valid)
                for (int q = 0; q < 12; q++) if (acc[size_t(i) * 12 + q] != 0.0f) nonzero++;
        const char* names[6] = {"dmean2D", "dconic", "dopacity", "drgb", "ddepth", "abs dmean2D (absgrad)"};
        for (int k = 0; k < 6; k++) addRow("rasterize bwd", names[k], sc, relErr(ga[k], fd[k]), 2e-3);
        addRow("rasterize bwd", "nonzero grads of undrawn splats", sc, nonzero, 0);
    }

    // ---- projection backward: random upstream screen-space gradients; f_g = Σ_q acc_gq · proj_gq(params_g)
    {
        std::vector<float> accR(size_t(n) * 12);
        for (float& v : accR) v = float(r.uni(-1, 1));
        ctx.upload(gs.acc, accR.data(), accR.size() * 4);
        gs.projectBackward();
        Params gG;
        gs.getGrads(gG);
        std::vector<double> ga[5], fd[5], gaC[5], fdC[5];
        double nonzero = 0;
        for (int i = 0; i < n; i++) {
            if (!gpuP[size_t(i)].valid) {
                for (int k = 0; k < 5; k++)
                    for (int j = 0; j < Params::width(k); j++) if (gG.group(k)[size_t(i) * Params::width(k) + j] != 0.0f) nonzero++;
                continue;
            }
            bool clamped = cpuP[size_t(i)].clamped;
            auto fg = [&](ref::ParamsD& Q) {
                ref::Proj p;
                ref::project(Q, rc, i, p, false);
                double s = 0;
                for (int q = 0; q < 10; q++) s += double(accR[size_t(i) * 12 + q]) * p.q(q);
                return s;
            };
            for (int k = 0; k < 5; k++)
                for (int j = 0; j < Params::width(k); j++) {
                    size_t at = size_t(i) * Params::width(k) + j;
                    double v = PD.g[k][at], h = fdStep(v);
                    PD.g[k][at] = v + h; double fp = fg(PD);
                    PD.g[k][at] = v - h; double fm = fg(PD);
                    PD.g[k][at] = v;
                    (clamped ? gaC : ga)[k].push_back(gG.group(k)[at]);
                    (clamped ? fdC : fd)[k].push_back((fp - fm) / (2 * h));
                }
        }
        const char* names[5] = {"dposition", "dlog-scale", "dquaternion", "dopacity logit", "dSH"};
        for (int k = 0; k < 5; k++) addRow("projection bwd", names[k], sc, relErr(ga[k], fd[k]), 1e-3);
        for (int k = 0; k < 5; k++)
            if (!gaC[k].empty()) addRow("projection bwd (clamped J, info)", names[k], sc, relErr(gaC[k], fdC[k]), 0, true);
        addRow("projection bwd", "nonzero grads of undrawn splats", sc, nonzero, 0);
    }

    // ---- end to end: GPU gradient of the full loss vs FD of the double-precision reference pipeline
    Params gG;
    double L0 = 0;
    {
        gs.forward();
        L0 = gs.loss();
        gs.rasterBackward();
        gs.projectBackward();
        gs.getGrads(gG);
        ref::Structure stC;
        std::vector<double> im, fT;
        std::vector<uint32_t> nC;
        ref::rasterDecide(cpuP, rc, im, fT, nC, &stC);
        std::vector<ref::Proj> pr = cpuP;
        auto L = [&](ref::ParamsD& Q, int g) {
            ref::project(Q, rc, g, pr[size_t(g)], false);
            ref::rasterFrozen(pr, stC, rc, im);
            return ref::loss(im, tP, cs.w, cs.h, opt.ssimWeight);
        };
        std::vector<int> sample;
        for (int i = 0; i < n; i++) if (cpuP[size_t(i)].valid && gpuP[size_t(i)].valid && !cpuP[size_t(i)].clamped) sample.push_back(i);
        std::shuffle(sample.begin(), sample.end(), r.g);
        sample.resize(std::min<size_t>(sample.size(), quick ? 6 : 20));
        std::vector<double> ga[5], fd[5];
        int nc = (cs.deg + 1) * (cs.deg + 1);
        for (int g : sample) {
            std::vector<std::pair<int, int>> which;
            for (int j = 0; j < 3; j++) which.push_back({0, j});
            for (int j = 0; j < 3; j++) which.push_back({1, j});
            for (int j = 0; j < 4; j++) which.push_back({2, j});
            which.push_back({3, 0});
            for (int j = 0; j < 3; j++) which.push_back({4, j});
            if (nc > 1) for (int j = 0; j < 2; j++) which.push_back({4, 3 + r.below(3 * nc - 3)});
            for (auto [k, j] : which) {
                size_t at = size_t(g) * Params::width(k) + j;
                double v = PD.g[k][at], h = 1e-5 * std::max(1.0, std::abs(v));
                PD.g[k][at] = v + h; double lp = L(PD, g);
                PD.g[k][at] = v - h; double lm = L(PD, g);
                PD.g[k][at] = v;
                ref::project(PD, rc, g, pr[size_t(g)], false);
                ga[k].push_back(gG.group(k)[at]);
                fd[k].push_back((lp - lm) / (2 * h));
            }
        }
        const char* names[5] = {"dL/dposition", "dL/dlog-scale", "dL/dquaternion", "dL/dopacity logit", "dL/dSH"};
        for (int k = 0; k < 5; k++) addRow("end to end (all kernels)", names[k], sc, relErr(ga[k], fd[k], 5e-2), 2e-2);
    }

    // ---- descent: a small step against the gradient lowers the GPU loss
    {
        double g2 = 0;
        for (int k = 0; k < 5; k++) for (float v : gG.group(k)) g2 += double(v) * v;
        double eta = 0.01 * L0 / std::max(g2, 1e-30);
        Params P2 = P;
        for (int k = 0; k < 5; k++)
            for (size_t j = 0; j < P2.group(k).size(); j++) P2.group(k)[j] -= float(eta * gG.group(k)[j]);
        gs.setParams(P2);
        gs.forward();
        double L1 = gs.loss();
        gs.setParams(P);
        std::printf("  [%s] descent: loss %.6f -> %.6f after a step of 1%% predicted decrease\n", sc.c_str(), L0, L1);
        addRow("descent check", "loss(p - eta g) / loss(p)", sc, L1 / L0, 1.0 - 1e-4);
    }
}

// ------------------------------------------------------------------ Adam against a double-precision step

void testAdam(Context& ctx, GpuSplat& gs) {
    Rng r(11);
    Params P;
    P.resize(37);
    for (int k = 0; k < 5; k++) for (float& v : P.group(k)) v = float(r.uni(-1, 1));
    gs.setParams(P);
    std::array<AdamGroup, 5> groups;
    for (int k = 0; k < 5; k++) groups[size_t(k)] = {float(1e-3 * (k + 1)), float(1e-3 * (k + 1)), 1, 1};
    groups[4] = {2.5e-3f, 1.25e-4f, 48, 3};
    std::vector<double> m[5], v[5], p[5];
    for (int k = 0; k < 5; k++) {
        p[k].assign(P.group(k).begin(), P.group(k).end());
        m[k].assign(p[k].size(), 0); v[k].assign(p[k].size(), 0);
    }
    for (int step = 1; step <= 3; step++) {
        for (int k = 0; k < 5; k++) {
            std::vector<float> g(P.group(k).size());
            for (float& x : g) x = float(r.normal() * 0.1);
            if (k == 1) g[3] = NAN;   // non-finite gradients count as zero
            ctx.upload(gs.grads[k], g.data(), g.size() * 4);
            double bc1 = 1 - std::pow(0.9, step), bc2 = 1 - std::pow(0.999, step);
            for (size_t i = 0; i < g.size(); i++) {
                double gi = std::isfinite(g[i]) ? g[i] : 0.0;
                m[k][i] = 0.9 * m[k][i] + 0.1 * gi;
                v[k][i] = 0.999 * v[k][i] + 0.001 * gi * gi;
                const AdamGroup& G = groups[size_t(k)];
                double lr = (i % G.period) < G.split ? G.lr : G.lr2;
                p[k][i] -= lr * (m[k][i] / bc1) / (std::sqrt(v[k][i] / bc2) + 1e-15);
            }
        }
        gs.adamStep(groups);
    }
    Params out;
    gs.getParams(out);
    std::vector<double> gd, cd;
    for (int k = 0; k < 5; k++)
        for (size_t i = 0; i < p[k].size(); i++) {
            gd.push_back(double(out.group(k)[i]) - P.group(k)[i]);
            cd.push_back(p[k][i] - P.group(k)[i]);
        }
    addRow("adam", "parameter change, 3 steps", "37 splats", relErr(gd, cd), 1e-3);
}

// ------------------------------------------------------------------ fitting a target image

void testFit(GpuSplat& gs, int steps) {
    Rng r(42);
    const double eye[3] = {0.3, -0.4, -4.0}, origin[3] = {0, 0, 0};
    Camera cam = lookAt(eye, origin, 0.1, 60, 60, 32, 24, 64, 48);
    RenderOpts opt;
    opt.shDegree = 0;
    opt.bg[0] = opt.bg[1] = opt.bg[2] = 0.0f;
    Params truth = randomScene(r, 150, eye, 0);
    gs.setParams(truth);
    gs.setCamera(cam, opt);
    gs.forward();
    size_t hw = 64 * 48;
    auto img = gs.read<float>(gs.img, 3 * hw);
    std::vector<uint32_t> tgt(hw);
    for (size_t p = 0; p < hw; p++) {
        uint32_t px = 0xFF000000u;
        for (int c = 0; c < 3; c++) px |= uint32_t(std::clamp(double(img[c * hw + p]), 0.0, 1.0) * 255.0 + 0.5) << (8 * c);
        tgt[p] = px;
    }
    Params init = randomScene(r, 150, eye, 0);
    for (float& v : init.opLogit) v = 0.0f;
    gs.setParams(init);
    gs.setCamera(cam, opt);
    gs.setTarget(tgt);
    std::array<AdamGroup, 5> groups{AdamGroup{0.01f, 0.01f, 1, 1}, AdamGroup{0.02f, 0.02f, 1, 1}, AdamGroup{0.02f, 0.02f, 1, 1},
                                    AdamGroup{0.05f, 0.05f, 1, 1}, AdamGroup{0.02f, 0.001f, 48, 3}};
    double first = 0, last = 0;
    for (int s = 0; s < steps; s++) {
        double l = gs.trainStep(groups);
        if (s == 0) first = l;
        last = l;
    }
    gs.forward();
    last = gs.loss();
    std::printf("  fit: %d Adam steps, loss %.5f -> %.5f (%.1f %%)\n", steps, first, last, 100.0 * last / first);
    addRow("training (adam fit)", "final / initial loss", std::to_string(steps) + " steps", last / first, 0.6);
}

// ------------------------------------------------------------------ stage 2: loss variants and training kernels

// Valid-window SSIM (fused-ssim padding="valid") and soft loss masks (target := m·photo + (1 − m)·render, the
// render held constant): GPU loss and dL/dimage against the double-precision reference and its finite
// differences in image space.
void testLossVariants(Context& ctx, GpuSplat& gs) {
    Rng r(23);
    const int w = 40, h = 30;
    const double eye[3] = {0, 0, -4}, origin[3] = {0, 0, 0};
    Camera cam = lookAt(eye, origin, 0.0, 40, 40, 20, 15, w, h);
    Params P = randomScene(r, 20, eye, 0);
    gs.setParams(P);
    RenderOpts opt;
    opt.ssimValid = true;
    opt.ssimWeight = 0.3f;
    gs.setCamera(cam, opt);
    size_t hw = size_t(w) * h;
    std::vector<float> img(4 * hw, 0.0f);
    for (size_t i = 0; i < 3 * hw; i++) img[i] = float(r.uni(-0.05, 1.05));   // a few values outside [0, 1]
    std::vector<uint32_t> tgt(hw);
    std::vector<double> tP(3 * hw), x(3 * hw), m(hw);
    for (size_t p = 0; p < hw; p++) {
        uint32_t px = 0;
        for (int c = 0; c < 3; c++) px |= uint32_t(r.below(256)) << (8 * c);
        int mm = r.below(4) == 0 ? r.below(256) : 255;   // a quarter of the pixels partly or fully masked
        px |= uint32_t(mm) << 24;
        tgt[p] = px;
        m[p] = mm / 255.0;
        for (int c = 0; c < 3; c++) {
            x[c * hw + p] = img[c * hw + p];
            double t = ((px >> (8 * c)) & 255u) / 255.0;
            tP[c * hw + p] = m[p] * t + (1 - m[p]) * x[c * hw + p];
        }
    }
    ctx.upload(gs.img, img.data(), img.size() * 4);
    gs.setTarget(tgt);
    double l1g = 0, ssg = 0;
    double Lg = gs.loss(&l1g, &ssg);
    float st0[4];
    ctx.download(gs.stats, st0, sizeof st0);
    double l1c = 0, ssc = 0;
    double Lc = ref::loss(x, tP, w, h, 0.3, &l1c, &ssc, true);
    addRow("loss (valid SSIM, soft mask)", "loss", "40x30", std::abs(Lg - Lc) / Lc, 1e-4);
    addRow("loss (valid SSIM, soft mask)", "mean SSIM", "40x30", std::abs(ssg - ssc), 1e-5);
    auto gimg = gs.read<float>(gs.gradImg, 3 * hw);
    std::vector<double> ga, fd;
    for (int s = 0; s < 60; s++) {
        size_t at = size_t(r.below(int(3 * hw)));
        if (std::abs(x[at] - tP[at]) < 1e-3) continue;   // |x − y| kink
        double v = x[at], hstep = 1e-6;
        x[at] = v + hstep; double lp = ref::loss(x, tP, w, h, 0.3, nullptr, nullptr, true);
        x[at] = v - hstep; double lm = ref::loss(x, tP, w, h, 0.3, nullptr, nullptr, true);
        x[at] = v;
        ga.push_back(gimg[at]);
        fd.push_back((lp - lm) / (2 * hstep));
    }
    addRow("loss (valid SSIM, soft mask)", "dL/dimage", "40x30", relErr(ga, fd, 5e-2), 2e-3);
    double se = 0;
    for (size_t i = 0; i < 3 * hw; i++) { double xc = std::clamp(x[i], 0.0, 1.0); se += (xc - tP[i]) * (xc - tP[i]); }
    float* st = st0;
    addRow("loss (valid SSIM, soft mask)", "PSNR statistic sum (clamp(x)-y)^2", "40x30", std::abs(st[2] - se) / se, 1e-4);
}

double regLoss(const ref::ParamsD& P, double wo, double ws, double wf, double wa, double maxRatio) {
    double L = 0;
    for (int i = 0; i < P.n; i++) {
        double o = 1 / (1 + std::exp(-P.g[3][size_t(i)]));
        L += wo * o;
        double l[3] = {P.g[1][size_t(i) * 3], P.g[1][size_t(i) * 3 + 1], P.g[1][size_t(i) * 3 + 2]};
        for (double v : l) L += ws * std::exp(v);
        double e2[3] = {std::exp(2 * l[0]), std::exp(2 * l[1]), std::exp(2 * l[2])};
        L += wf * std::min({e2[0], e2[1], e2[2]}) / (e2[0] + e2[1] + e2[2]);
        double s[3] = {l[0], l[1], l[2]};
        std::sort(s, s + 3);
        L += wa * std::max(std::exp(s[2] - s[1]) - maxRatio, 0.0);   // the middle axis is detached: handled by FD below
    }
    return L;
}

void testTrainingKernels(Context& ctx, GpuSplat& gs) {
    Rng r(31);
    const int n = 300;
    Params P;
    P.resize(n);
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < 3; k++) P.pos[size_t(i) * 3 + k] = float(r.uni(-1, 1));
        for (int k = 0; k < 3; k++) P.logScale[size_t(i) * 3 + k] = float(r.uni(-4, -1));
        if (i % 3 == 0) P.logScale[size_t(i) * 3 + size_t(r.below(3))] += 3.0f;   // needles beyond the ratio limit
        for (int k = 0; k < 4; k++) P.quat[size_t(i) * 4 + k] = float(r.uni(-1, 1));
        P.opLogit[size_t(i)] = float(r.uni(-6, 6));
    }
    gs.setParams(P);
    // ---- regularisers: the GPU gradients against finite differences of the summed penalty
    {
        const double wo = 0.01 / n, ws = 0.01 / (3.0 * n), wf = 0.2 / n, wa = 0.1 / n, maxRatio = 10;
        struct { float wOp, wScale, wFlat, wAniso, maxRatio; uint32_t n; } pc{float(wo), float(ws), float(wf), float(wa), float(maxRatio), n};
        ctx.begin();
        ctx.fill(gs.grads[1], 0);
        ctx.fill(gs.grads[3], 0);
        ctx.dispatch("reg_grad", {&gs.params[1], &gs.params[3], &gs.grads[1], &gs.grads[3]}, &pc, sizeof pc, (n + 255) / 256);
        ctx.submitWait();
        auto gS = gs.read<float>(gs.grads[1], size_t(n) * 3), gO = gs.read<float>(gs.grads[3], size_t(n));
        ref::ParamsD D = ref::toDouble(P);
        std::vector<double> ga, fd;
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < 4; k++) {
                int grp = k < 3 ? 1 : 3;
                size_t at = k < 3 ? size_t(i) * 3 + size_t(k) : size_t(i);
                double v = D.g[grp][at], hs = 1e-6;
                // the anisotropy term detaches the middle axis: freeze it by evaluating the FD on the longest axis only
                double l[3] = {D.g[1][size_t(i) * 3], D.g[1][size_t(i) * 3 + 1], D.g[1][size_t(i) * 3 + 2]};
                int imax = int(std::max_element(l, l + 3) - l);
                double s[3] = {l[0], l[1], l[2]};
                std::sort(s, s + 3);
                bool anisoActive = std::exp(s[2] - s[1]) > maxRatio;
                if (grp == 1 && anisoActive && k != imax) continue;   // d/d(middle) differs by design (detached)
                D.g[grp][at] = v + hs; double lp = regLoss(D, wo, ws, wf, wa, maxRatio);
                D.g[grp][at] = v - hs; double lm = regLoss(D, wo, ws, wf, wa, maxRatio);
                D.g[grp][at] = v;
                ga.push_back(grp == 1 ? gS[at] : gO[at]);
                fd.push_back((lp - lm) / (2 * hs));
            }
        }
        addRow("regularisers (reg_grad)", "d/dlog-scale, d/dopacity", "300 splats", relErr(ga, fd), 1e-3);
    }
    // ---- MCMC noise: the kernel against the host implementation (the same counter-based normals)
    {
        float scaler = 40.0f;
        struct { float scaler; uint32_t seed, step, n; } pc{scaler, 42u, 1234u, n};
        ctx.begin();
        ctx.dispatch("mcmc_noise", {&gs.params[0], &gs.params[1], &gs.params[2], &gs.params[3]}, &pc, sizeof pc, (n + 255) / 256);
        ctx.submitWait();
        auto pos = gs.read<float>(gs.params[0], size_t(n) * 3);
        SplatState s;
        s.p = P;
        mcmcNoise(s, scaler, 42u, 1234u);
        std::vector<double> ga, cd;
        for (size_t i = 0; i < pos.size(); i++) { ga.push_back(double(pos[i]) - P.pos[i]); cd.push_back(double(s.p.pos[i]) - P.pos[i]); }
        double moved = 0;
        for (double v : cd) moved += v != 0;
        addRow("mcmc noise", "displacement vs host version", "300 splats", relErr(ga, cd, 1e-2), 1e-3);
        addRow("mcmc noise", "displaced coordinates (info)", "300 splats", moved, 0, true);
        // mean / variance of the hash normals
        double s1 = 0, s2 = 0;
        const int cnt = 200000;
        for (int i = 0; i < cnt; i++) { double z = hashNormal(7u, 3u, uint32_t(i), uint32_t(i % 3)); s1 += z; s2 += z * z; }
        double mean = s1 / cnt, var = s2 / cnt - mean * mean;
        addRow("hash normals", "|mean| + |var - 1|", "200k samples", std::abs(mean) + std::abs(var - 1), 0.02);
    }
    // ---- densification statistics (adaptive strategy)
    {
        std::vector<float> acc(static_cast<size_t>(n) * 12);
        std::vector<uint32_t> touched(static_cast<size_t>(n));
        for (float& v : acc) v = float(r.uni(-1e-3, 1e-3));
        for (auto& t : touched) t = r.below(3) ? uint32_t(r.below(5) + 1) : 0u;
        ctx.upload(gs.acc, acc.data(), acc.size() * 4);
        ctx.upload(gs.tilesTouched, touched.data(), touched.size() * 4);
        Buffer g2 = ctx.createBuffer(size_t(n) * 4), cnt = ctx.createBuffer(size_t(n) * 4);
        std::vector<double> ga, ca;
        for (int absg = 0; absg < 2; absg++) {
            struct { float hw, hh; uint32_t absGrad, n; } pc{64.0f, 48.0f, uint32_t(absg), n};
            ctx.begin();
            ctx.fill(g2, 0);
            ctx.fill(cnt, 0);
            ctx.dispatch("grad_stats", {&gs.acc, &gs.tilesTouched, &g2, &cnt}, &pc, sizeof pc, (n + 255) / 256);
            ctx.dispatch("grad_stats", {&gs.acc, &gs.tilesTouched, &g2, &cnt}, &pc, sizeof pc, (n + 255) / 256);
            ctx.submitWait();
            auto G = gs.read<float>(g2, size_t(n)), C = gs.read<float>(cnt, size_t(n));
            for (int i = 0; i < n; i++) {
                size_t o = size_t(i) * 12 + (absg ? 10 : 0);
                double e = touched[size_t(i)] ? 2 * std::hypot(acc[o] * 64.0, acc[o + 1] * 48.0) : 0.0;
                ga.push_back(G[size_t(i)]); ca.push_back(e);
                ga.push_back(C[size_t(i)]); ca.push_back(touched[size_t(i)] ? 2.0 : 0.0);
            }
        }
        addRow("densification statistics", "grad2d, count (grad and absgrad)", "300 splats", relErr(ga, ca), 1e-5);
        ctx.destroy(g2);
        ctx.destroy(cnt);
    }
}

// ------------------------------------------------------------------ stage 2b: the image-space loss pipeline
// Sky composite, per-photo colour correction (affine / bilateral grid), decoupled SSIM, depth supervision and the
// depth-normal loss on ED = D / alpha, the sky coverage penalty: the GPU loss and its gradients w.r.t. the rendered
// colour, depth plane and T_final, the sky panorama and the correction parameters, against central finite
// differences of a double-precision reference written from trainer.py's formulas (render detached in the target,
// the bilateral grid's luminance guide and the normal-validity decisions frozen at the base point, as in torch).

struct PipeCase {
    std::string name;
    int w = 48, h = 40, viewW = 48, viewH = 40;
    bool sky = false, skyMask = false;
    int corr = 0;   // 0 none, 1 affine, 2 bilateral grid
    bool split = false;
    bool depth = false, normal = false, edge = true;
    int sensorW = 24, sensorH = 20;
    uint64_t seed = 1;
};

struct Pipe {
    PipeCase pc;
    Camera cam;
    double bg[3] = {0.2, 0.5, 0.7};
    double lam = 0.25;
    std::vector<double> C, T, D;   // render: rgb planes over black, T_final, Σ α·T·z
    std::vector<uint32_t> target;
    std::vector<uint8_t> skym;
    std::vector<double> tex;       // sky logits [sH][sW][3]
    int sW = 16, sH = 8;
    float rot[9] = {};
    std::vector<double> corr;
    std::vector<float> sensor;
    double skyW = 0.05, dw = 0.3, nw = 0.2;
    int k = 1;
    double count = 0;              // valid depth pixels
    std::vector<double> y, guide, ntgt, cor;
    std::vector<uint8_t> nvalid;
};

const double* ssimWindowD() {
    static double w[11];
    static bool init = false;
    if (!init) {
        double s = 0;
        for (int k = 0; k < 11; k++) s += w[k] = double(float(std::exp(-double((k - 5) * (k - 5)) / (2.0 * 1.5 * 1.5))));
        for (double& v : w) v = double(float(v / s));
        init = true;
    }
    return w;
}

double sensorAtRef(const Pipe& P, int x, int y) {
    const PipeCase& c = P.pc;
    if (x < 0 || y < 0 || x >= c.w || y >= c.h) return 0;
    uint32_t xv = uint32_t(x), yv = uint32_t(y);
    if (c.w != c.viewW) xv = std::min(uint32_t(std::floor(float(x) * (float(c.viewW) / float(c.w)))), uint32_t(c.viewW - 1));
    if (c.h != c.viewH) yv = std::min(uint32_t(std::floor(float(y) * (float(c.viewH) / float(c.h)))), uint32_t(c.viewH - 1));
    uint32_t xs = ((2 * xv + 1) * uint32_t(c.sensorW)) / (2 * uint32_t(c.viewW));
    uint32_t ys = ((2 * yv + 1) * uint32_t(c.sensorH)) / (2 * uint32_t(c.viewH));
    return P.sensor[size_t(ys) * size_t(c.sensorW) + xs];
}

double lumRef(const Pipe& P, size_t q) {
    uint32_t t = P.target[q];
    return (0.299 * double(t & 255u) + 0.587 * double((t >> 8) & 255u) + 0.114 * double((t >> 16) & 255u)) / 255.0;
}

double ewRef(const Pipe& P, int x, int y) {
    if (!P.pc.edge) return 1.0;
    int W = P.pc.w, H = P.pc.h;
    size_t q = size_t(y) * W + x;
    double gx = (x >= 1 && x + 1 < W) ? (lumRef(P, q + 1) - lumRef(P, q - 1)) * 0.5 : 0.0;
    double gy = (y >= 1 && y + 1 < H) ? (lumRef(P, q + size_t(W)) - lumRef(P, q - size_t(W))) * 0.5 : 0.0;
    return std::exp(-10.0 * std::sqrt(gx * gx + gy * gy));
}

void skyColRef(const Pipe& P, const std::vector<double>& tex, int x, int y, double out[3]) {
    const Camera& c = P.cam;
    double dc[3] = {(x + 0.5 - c.cx) / c.fx, (y + 0.5 - c.cy) / c.fy, 1.0};
    double s[3];
    for (int i = 0; i < 3; i++) s[i] = P.rot[i * 3] * dc[0] + P.rot[i * 3 + 1] * dc[1] + P.rot[i * 3 + 2] * dc[2];
    double l = std::sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
    for (double& v : s) v /= l;
    double lon = std::atan2(s[0], s[2]), lat = std::asin(std::clamp(s[1], -1.0, 1.0));
    double gx = lon / M_PI, gy = -lat / (M_PI / 2);
    double ix = std::clamp(((gx + 1) * P.sW - 1) * 0.5, 0.0, double(P.sW - 1));
    double iy = std::clamp(((gy + 1) * P.sH - 1) * 0.5, 0.0, double(P.sH - 1));
    int x0 = int(std::floor(ix)), y0 = int(std::floor(iy));
    int x1 = std::min(x0 + 1, P.sW - 1), y1 = std::min(y0 + 1, P.sH - 1);
    double ax = ix - x0, ay = iy - y0;
    int idx[4] = {y0 * P.sW + x0, y0 * P.sW + x1, y1 * P.sW + x0, y1 * P.sW + x1};
    double w[4] = {(1 - ax) * (1 - ay), ax * (1 - ay), (1 - ax) * ay, ax * ay};
    for (int ch = 0; ch < 3; ch++) {
        out[ch] = 0;
        for (int k = 0; k < 4; k++) out[ch] += w[k] / (1 + std::exp(-tex[size_t(idx[k]) * 3 + size_t(ch)]));
    }
}

// trainer.py's loss on the given render; freeze = true records the detached / decided quantities.
double pipeLoss(Pipe& P, const std::vector<double>& C, const std::vector<double>& T, const std::vector<double>& D,
                const std::vector<double>& tex, const std::vector<double>& corr, bool freeze, double parts[4] = nullptr) {
    const PipeCase& c = P.pc;
    int W = c.w, H = c.h;
    size_t hw = size_t(W) * H;
    std::vector<double> raw(3 * hw), cor(3 * hw);
    if (freeze) { P.guide.assign(hw, 0); P.y.assign(3 * hw, 0); }
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            size_t p = size_t(y) * W + x;
            double b[3] = {P.bg[0], P.bg[1], P.bg[2]};
            if (c.sky) skyColRef(P, tex, x, y, b);
            for (int ch = 0; ch < 3; ch++) raw[ch * hw + p] = C[ch * hw + p] + T[p] * b[ch];
            double xr[3] = {raw[p], raw[hw + p], raw[2 * hw + p]};
            if (freeze) P.guide[p] = std::clamp(0.299 * xr[0] + 0.587 * xr[1] + 0.114 * xr[2], 0.0, 1.0);
            double cf[12];
            if (c.corr == 1) {
                for (int r = 0; r < 3; r++) { for (int k = 0; k < 3; k++) cf[r * 4 + k] = corr[size_t(r * 3 + k)]; cf[r * 4 + 3] = corr[size_t(9 + r)]; }
            } else if (c.corr == 2) {
                double g[3] = {(x + 0.5) / W * 15.0, (y + 0.5) / H * 15.0, P.guide[p] * 7.0};
                int i0[3];
                double f[3];
                const int lim[3] = {14, 14, 6};
                for (int a = 0; a < 3; a++) { i0[a] = std::clamp(int(std::floor(g[a])), 0, lim[a]); f[a] = std::clamp(g[a] - i0[a], 0.0, 1.0); }
                for (int k = 0; k < 12; k++) {
                    double v = 0;
                    for (int dz = 0; dz < 2; dz++) for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) {
                        double wt = (dx ? f[0] : 1 - f[0]) * (dy ? f[1] : 1 - f[1]) * (dz ? f[2] : 1 - f[2]);
                        v += wt * corr[((size_t(k) * 8 + size_t(i0[2] + dz)) * 16 + size_t(i0[1] + dy)) * 16 + size_t(i0[0] + dx)];
                    }
                    cf[k] = v;
                }
            }
            for (int r = 0; r < 3; r++)
                cor[r * hw + p] = c.corr ? cf[r * 4] * xr[0] + cf[r * 4 + 1] * xr[1] + cf[r * 4 + 2] * xr[2] + cf[r * 4 + 3] : xr[r];
            if (freeze) {
                uint32_t t = P.target[p];
                double m = double(t >> 24) / 255.0;
                for (int ch = 0; ch < 3; ch++) P.y[ch * hw + p] = m * double((t >> (8 * ch)) & 255u) / 255.0 + (1 - m) * cor[ch * hw + p];
            }
        }
    if (freeze) P.cor = cor;
    double l1 = 0;
    for (size_t i = 0; i < 3 * hw; i++) l1 += std::abs(cor[i] - P.y[i]);
    l1 /= double(3 * hw);
    const double* wn = ssimWindowD();
    const double C1 = 0.0001, C2 = 0.0009;
    double sumS = 0;
    for (int ch = 0; ch < 3; ch++) {
        const double* xc = &cor[ch * hw];
        const double* xr = &raw[ch * hw];
        const double* yy = &P.y[ch * hw];
        for (int y = 5; y + 5 < H; y++)
            for (int x = 5; x + 5 < W; x++) {
                double s[6] = {0, 0, 0, 0, 0, 0};   // E[c], E[y], E[r²] / E[c²], E[y²], E[r·y] / E[c·y], E[r]
                for (int dy = -5; dy <= 5; dy++)
                    for (int dx = -5; dx <= 5; dx++) {
                        size_t q = size_t(y + dy) * W + size_t(x + dx);
                        double wt = wn[dx + 5] * wn[dy + 5];
                        double a = xc[q], b = yy[q], rr = c.split ? xr[q] : xc[q];
                        s[0] += wt * a; s[1] += wt * b; s[2] += wt * rr * rr; s[3] += wt * b * b; s[4] += wt * rr * b; s[5] += wt * rr;
                    }
                double lum = (2 * s[0] * s[1] + C1) / (s[0] * s[0] + s[1] * s[1] + C1);
                double sr = s[2] - s[5] * s[5], sy = s[3] - s[1] * s[1], sry = s[4] - s[5] * s[1];
                double cs = (2 * sry + C2) / (sr + sy + C2);
                sumS += lum * cs;
            }
    }
    double meanS = sumS / double(3 * (W - 10) * (H - 10));
    double photo = (1 - P.lam) * l1 + P.lam * (1 - meanS);
    std::vector<double> ed(hw);
    for (size_t p = 0; p < hw; p++) ed[p] = D[p] / std::max(1 - T[p], 1e-10);
    double dsum = 0, nsum = 0, pen = 0;
    if (c.depth)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                double d = sensorAtRef(P, x, y);
                if (d > 0) dsum += ewRef(P, x, y) * std::abs(ed[size_t(y) * W + x] - d);
            }
    if (c.normal) {
        if (freeze) { P.nvalid.assign(hw, 0); P.ntgt.assign(3 * hw, 0); }
        auto ray = [&](double u, double v, double out[3]) { out[0] = (u - P.cam.cx) / P.cam.fx; out[1] = (v - P.cam.cy) / P.cam.fy; out[2] = 1; };
        auto cross = [](const double a[3], const double b[3], double o[3]) {
            o[0] = a[1] * b[2] - a[2] * b[1]; o[1] = a[2] * b[0] - a[0] * b[2]; o[2] = a[0] * b[1] - a[1] * b[0];
        };
        int k = P.k;
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                size_t p = size_t(y) * W + x;
                if (freeze) {
                    bool ok = x >= k && y >= k && x + k < W && y + k < H && 1 - T[p] > 0.9;
                    if (ok) {
                        double e = ed[p], tol = 0.10 * std::max(e, 1e-9);
                        double nb[4] = {ed[p - 1], ed[p + 1], ed[p - size_t(W)], ed[p + size_t(W)]};
                        ok = e > 0;
                        for (double v : nb) ok = ok && v > 0 && std::abs(v - e) < tol;
                    }
                    if (ok) {
                        double c0 = sensorAtRef(P, x, y), sl = sensorAtRef(P, x - k, y), sr = sensorAtRef(P, x + k, y);
                        double su = sensorAtRef(P, x, y - k), sd = sensorAtRef(P, x, y + k), tol = 0.05 * std::max(c0, 1e-9);
                        ok = c0 > 0;
                        for (double v : {sl, sr, su, sd}) ok = ok && v > 0 && std::abs(v - c0) < tol;
                        if (ok) {
                            double a1[3], a2[3], a3[3], a4[3], e1[3], e2[3], m[3];
                            ray(x + k, y, a1); ray(x - k, y, a2); ray(x, y + k, a3); ray(x, y - k, a4);
                            for (int i = 0; i < 3; i++) { e1[i] = a1[i] * sr - a2[i] * sl; e2[i] = a3[i] * sd - a4[i] * su; }
                            cross(e1, e2, m);
                            double lm = std::sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]);
                            ok = lm > 1e-12;
                            if (ok) for (int i = 0; i < 3; i++) P.ntgt[p * 3 + size_t(i)] = m[i] / lm;
                        }
                    }
                    P.nvalid[p] = ok;
                }
                if (!P.nvalid[p]) continue;
                double al[3], ar[3], au[3], ad[3], e1[3], e2[3], m[3];
                ray(x - 1, y, al); ray(x + 1, y, ar); ray(x, y - 1, au); ray(x, y + 1, ad);
                for (int i = 0; i < 3; i++) {
                    e1[i] = ar[i] * ed[p + 1] - al[i] * ed[p - 1];
                    e2[i] = ad[i] * ed[p + size_t(W)] - au[i] * ed[p - size_t(W)];
                }
                cross(e1, e2, m);
                double lm = std::sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]);
                double cc = 0;
                for (int i = 0; i < 3; i++) cc += m[i] / lm * P.ntgt[p * 3 + size_t(i)];
                cc = std::clamp(cc, -1.0, 1.0);
                double hh = std::max(0.5 * (1 - cc), 0.0);
                nsum += ewRef(P, x, y) * (hh + std::sqrt(hh + 1e-4));
            }
    }
    if (c.sky && c.skyMask) {
        double ms = 0, s = 0;
        for (size_t p = 0; p < hw; p++) { double m = P.skym[p] / 255.0; ms += m; s += (1 - T[p]) * m; }
        pen = P.skyW * s / std::max(ms, 1.0);
    }
    if (parts) { parts[0] = photo; parts[1] = dsum; parts[2] = nsum; parts[3] = pen; }
    double total = photo + pen;
    if (c.depth && P.count > 0) total += P.dw / P.count * dsum;
    if (c.normal) total += P.nw / double(hw) * nsum;
    return total;
}

void testImagePipeline(Context& ctx, GpuSplat& gs, bool quick) {
    std::vector<PipeCase> cases;
    {
        PipeCase a; a.name = "sky+affine+split+depth+normal"; a.sky = a.skyMask = true; a.corr = 1; a.split = true;
        a.depth = a.normal = true; a.seed = 3; cases.push_back(a);
        PipeCase b; b.name = "grid+depth(scaled view)"; b.corr = 2; b.depth = true; b.viewW = 60; b.viewH = 50;
        b.sensorW = 30; b.sensorH = 25; b.seed = 4; cases.push_back(b);
        PipeCase c; c.name = "sky+grid+split+normal"; c.sky = true; c.corr = 2; c.split = true; c.normal = true; c.edge = false;
        c.seed = 5; cases.push_back(c);
    }
    for (const PipeCase& pc : cases) {
        Rng r(pc.seed);
        Pipe P;
        P.pc = pc;
        int W = pc.w, H = pc.h;
        size_t hw = size_t(W) * H;
        const double eye[3] = {0.3, -0.2, -3.0}, origin[3] = {0, 0, 0};
        P.cam = lookAt(eye, origin, 0.2, 40, 42, 24, 20, W, H);
        P.C.resize(3 * hw); P.T.resize(hw); P.D.resize(hw); P.target.resize(hw); P.skym.assign(hw, 0);
        double ph[3] = {r.uni(0, 6), r.uni(0, 6), r.uni(0, 6)};
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                size_t p = size_t(y) * W + x;
                bool open = x >= W * 6 / 10;
                double T = open ? r.uni(0.3, 0.9) : r.uni(0.01, 0.07);
                P.T[p] = T;
                double z = 2.0 + 0.004 * x + 0.003 * y + 0.02 * std::sin(0.2 * x + 0.1 * y);
                P.D[p] = (1 - T) * z;
                uint32_t t = 0;
                for (int ch = 0; ch < 3; ch++) {
                    P.C[ch * hw + p] = (1 - T) * (0.45 + 0.3 * std::sin(0.3 * x + 0.2 * y + ph[ch]));
                    double v = 0.5 + 0.3 * std::sin(0.25 * x - 0.15 * y + ph[ch] + 1) + r.uni(-0.1, 0.1);
                    t |= uint32_t(std::clamp(v, 0.0, 1.0) * 255 + 0.5) << (8 * ch);
                }
                uint32_t m = r.below(10) == 0 ? uint32_t(r.below(256)) : 255u;
                P.target[p] = t | (m << 24);
                if (open) P.skym[p] = r.below(5) == 0 ? uint8_t(r.below(256)) : 255;
            }
        P.tex.resize(size_t(P.sW) * P.sH * 3);
        for (double& v : P.tex) v = r.normal();
        double up[3] = {0, -1, 0};
        skyRotation(skyBasis(up), P.cam.w2c, P.rot);
        if (pc.corr == 1) {
            P.corr.assign(12, 0.0);
            for (int k = 0; k < 9; k++) P.corr[size_t(k)] = (k % 4 == 0 ? 1.0 : 0.0) + 0.1 * r.normal();
            for (int k = 9; k < 12; k++) P.corr[size_t(k)] = 0.05 * r.normal();
        } else if (pc.corr == 2) {
            std::vector<float> id = BilateralGridModel::identity();
            P.corr.resize(id.size());
            for (size_t k = 0; k < id.size(); k++) P.corr[k] = id[k] + 0.05 * r.normal();
        }
        P.sensor.resize(size_t(pc.sensorW) * pc.sensorH);
        for (int y = 0; y < pc.sensorH; y++)
            for (int x = 0; x < pc.sensorW; x++) {
                double vx = (x + 0.5) * pc.viewW / pc.sensorW * double(W) / pc.viewW, vy = (y + 0.5) * pc.viewH / pc.sensorH * double(H) / pc.viewH;
                P.sensor[size_t(y) * pc.sensorW + x] = r.below(20) == 0 ? 0.0f : float(1.9 + 0.006 * vx + 0.001 * vy);
            }
        P.k = std::max(1, int(std::nearbyint(double(W) / pc.sensorW)));
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) P.count += sensorAtRef(P, x, y) > 0;

        double parts[4];
        double Lc = pipeLoss(P, P.C, P.T, P.D, P.tex, P.corr, true, parts);

        // ---- GPU
        gs.feat = LossFeatures();
        Buffer texB, texG;
        double msum = 0;
        for (uint8_t m : P.skym) msum += m / 255.0;
        if (pc.sky) {
            std::vector<float> tf(P.tex.begin(), P.tex.end());
            texB = ctx.createBuffer(tf.size() * 4);
            texG = ctx.createBuffer(tf.size() * 4);
            ctx.upload(texB, tf.data(), tf.size() * 4);
            gs.feat.sky = true; gs.feat.skyTex = &texB; gs.feat.skyGrad = &texG;
            gs.feat.skyW = uint32_t(P.sW); gs.feat.skyH = uint32_t(P.sH);
            std::copy(P.rot, P.rot + 9, gs.feat.skyRot);
            gs.feat.skyMask = pc.skyMask;
            gs.feat.skyPenalty = float(-P.skyW / std::max(msum, 1.0));
        }
        gs.feat.corr = LossFeatures::Corr(pc.corr);
        gs.feat.decoupledSSIM = pc.split;
        gs.feat.depthScale = pc.depth ? float(P.dw / P.count) : 0.0f;
        gs.feat.normalScale = pc.normal ? float(P.nw / double(hw)) : 0.0f;
        gs.feat.edgeAware = pc.edge;
        gs.feat.normalK = P.k;
        RenderOpts opt;
        opt.ssimValid = true;
        opt.ssimWeight = float(P.lam);
        opt.hasDepth = pc.depth || pc.normal;
        for (int k = 0; k < 3; k++) opt.bg[k] = pc.sky ? 0.0f : float(P.bg[k]);
        gs.setCamera(P.cam, opt);
        gs.setTarget(P.target);
        std::vector<float> im(4 * hw), tf(hw);
        for (size_t i = 0; i < 3 * hw; i++) im[i] = float(P.C[i] + (pc.sky ? 0.0 : P.T[i % hw] * P.bg[i / hw]));   // raster_fwd's C + T·bg
        for (size_t p = 0; p < hw; p++) { im[3 * hw + p] = float(P.D[p]); tf[p] = float(P.T[p]); }
        ctx.upload(gs.img, im.data(), im.size() * 4);
        ctx.upload(gs.finalT, tf.data(), tf.size() * 4);
        if (pc.sky) {
            std::vector<uint32_t> aux(hw);
            for (size_t p = 0; p < hw; p++) aux[p] = P.skym[p];
            gs.setAux(aux);
        }
        if (pc.corr) {
            std::vector<float> cf(P.corr.begin(), P.corr.end());
            gs.setCorrection(cf.data(), cf.size());
        }
        if (pc.depth || pc.normal) gs.setDepthTarget(P.sensor.data(), pc.sensorW, pc.sensorH, pc.viewW, pc.viewH);
        double Lg = gs.imageLoss();
        double dsumG = gs.depthLossSum(), nsumG = gs.normalLossSum();
        gs.imageBackward();
        std::vector<float> G = gs.read<float>(gs.gradImg, 5 * hw);
        std::vector<float> texGrad = pc.sky ? gs.read<float>(texG, P.tex.size()) : std::vector<float>();
        std::vector<float> corrGrad = pc.corr ? gs.readCorrGrad() : std::vector<float>();
        const std::string sc = pc.name;
        addRow("image loss pipeline", "photometric loss value", sc, std::abs(Lg - parts[0]) / parts[0], 2e-4);
        if (pc.depth) addRow("image loss pipeline", "depth loss sum", sc, std::abs(dsumG - parts[1]) / parts[1], 1e-4);
        if (pc.normal) {
            addRow("image loss pipeline", "normal loss sum", sc, std::abs(nsumG - parts[2]) / std::max(parts[2], 1e-12), 1e-3);
            addRow("image loss pipeline", "normal-valid pixels (info)", sc, gs.normalValid(), 0, true);
        }
        std::printf("  [%s] loss %.6f (photometric %.6f, depth sum %.4f, normal sum %.4f, sky penalty %.5f)\n", sc.c_str(), Lc,
                    parts[0], parts[1], parts[2], parts[3]);

        // ---- finite differences
        auto L = [&]() { return pipeLoss(P, P.C, P.T, P.D, P.tex, P.corr, false); };
        auto fdOf = [&](std::vector<double>& v, size_t i, auto&& lossFn) {
            double x0 = v[i], h = 1e-6 * std::max(1.0, std::abs(x0));
            v[i] = x0 + h; double lp = lossFn();
            v[i] = x0 - h; double lm = lossFn();
            v[i] = x0;
            return (lp - lm) / (2 * h);
        };
        auto nearKink = [&](size_t p) {   // |corrected − target| or |ED − d| near (but not exactly) 0: the L1 kinks
            for (int ch = 0; ch < 3; ch++) {
                double dlt = std::abs(P.cor[size_t(ch) * hw + p] - P.y[size_t(ch) * hw + p]);
                if (dlt > 0 && dlt < 1e-4) return true;
            }
            double e = P.D[p] / std::max(1 - P.T[p], 1e-10);
            int x = int(p % size_t(W)), y = int(p / size_t(W));
            double d = sensorAtRef(P, x, y);
            return pc.depth && d > 0 && std::abs(e - d) < 1e-4;
        };
        int samples = quick ? 24 : 60;
        std::vector<double> ga[5], fd[5];   // colour, depth plane, T, sky, correction
        for (int s = 0; s < samples; s++) {
            size_t p = size_t(r.below(int(hw)));
            if (nearKink(p)) continue;
            int ch = r.below(3);
            size_t i = size_t(ch) * hw + p;
            ga[0].push_back(G[i]);
            fd[0].push_back(fdOf(P.C, i, L));
        }
        for (int s = 0; s < samples; s++) {
            size_t p = size_t(r.below(int(hw)));
            if (nearKink(p)) continue;
            ga[1].push_back(G[3 * hw + p]);
            fd[1].push_back(fdOf(P.D, p, L));
            double gT = G[4 * hw + p];
            if (!pc.sky) for (int ch = 0; ch < 3; ch++) gT += P.bg[ch] * G[size_t(ch) * hw + p];
            ga[2].push_back(gT);
            fd[2].push_back(fdOf(P.T, p, L));
        }
        if (pc.sky) {
            std::vector<size_t> used;
            for (size_t i = 0; i < texGrad.size(); i++) if (texGrad[i] != 0.0f) used.push_back(i);
            std::shuffle(used.begin(), used.end(), r.g);
            used.resize(std::min<size_t>(used.size(), size_t(samples)));
            for (size_t i : used) { ga[3].push_back(texGrad[i]); fd[3].push_back(fdOf(P.tex, i, L)); }
            addRow("image loss pipeline", "sky texels with gradient (info)", sc, double(used.size()), 0, true);
        }
        if (pc.corr) {
            std::vector<size_t> idx;
            if (pc.corr == 1) for (size_t i = 0; i < 12; i++) idx.push_back(i);
            else {
                for (size_t i = 0; i < corrGrad.size(); i++) if (corrGrad[i] != 0.0f) idx.push_back(i);
                std::shuffle(idx.begin(), idx.end(), r.g);
                idx.resize(std::min<size_t>(idx.size(), size_t(samples)));
                idx.push_back(0);   // a cell no pixel uses: zero
            }
            for (size_t i : idx) { ga[4].push_back(corrGrad[i]); fd[4].push_back(fdOf(P.corr, i, L)); }
        }
        const char* names[5] = {"dL/drender rgb", "dL/ddepth plane", "dL/dT_final", "dL/dsky logits", "dL/dcorrection"};
        for (int k = 0; k < 5; k++)
            if (!ga[k].empty()) addRow("image loss pipeline", names[k], sc, relErr(ga[k], fd[k], 5e-2), 2e-3);
        if (texB.buf) { ctx.destroy(texB); ctx.destroy(texG); }
        gs.feat = LossFeatures();
    }
}

// Transient masking: the GPU threshold (histogram median) against the exact median of the per-pixel errors, and the
// masks against trainer.py's rule (solid 3×3 regions, 16×16 patch vote) evaluated on the CPU at that threshold.
void testTransient(Context& ctx, GpuSplat& gs) {
    Rng r(91);
    const int W = 64, H = 48;
    size_t hw = size_t(W) * H;
    Camera cam = lookAt(std::array<double, 3>{0, 0, -3}.data(), std::array<double, 3>{0, 0, 0}.data(), 0, 50, 50, 32, 24, W, H);
    std::vector<uint32_t> tgt(hw);
    std::vector<float> im(4 * hw, 0.0f), T(hw);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            size_t p = size_t(y) * W + x;
            bool blob = x >= 4 && x < 26 && y >= 8 && y < 30;              // a moving object: solid, mostly one tile
            bool smallBlob = x >= 40 && x < 46 && y >= 34 && y < 40;       // solid but inside a mostly good patch
            bool speck = r.below(40) == 0;                                 // scattered outliers
            double err = (blob || smallBlob || speck) ? 0.6 : r.uni(0.0, 0.12);
            uint32_t t = 255u << 24;
            for (int c = 0; c < 3; c++) {
                int v = 40 + r.below(170);
                t |= uint32_t(v) << (8 * c);
                im[size_t(c) * hw + p] = float(v / 255.0 + (r.below(2) ? err : -err));
            }
            tgt[p] = t;
            T[p] = (x >= 18 && x < 24 && y >= 10 && y < 16) ? 0.5f : 0.01f;   // a thinly covered corner of the blob
        }
    gs.feat = LossFeatures();
    gs.feat.transient = gs.feat.transientActive = true;
    gs.feat.transientSensitivity = 3.0f;
    RenderOpts opt;
    opt.ssimValid = true;
    gs.setCamera(cam, opt);
    gs.setTarget(tgt);
    gs.setTransientState(0.0f, false);
    ctx.upload(gs.img, im.data(), im.size() * 4);
    ctx.upload(gs.finalT, T.data(), T.size() * 4);
    gs.imageLoss();
    GpuSplat::TransientState st = gs.transientState();
    std::vector<float> wgt = gs.read<float>(gs.weight, hw);
    std::vector<double> res(hw);
    for (size_t p = 0; p < hw; p++) {
        float s = 0;
        for (int c = 0; c < 3; c++) s += std::abs(im[size_t(c) * hw + p] - float((tgt[p] >> (8 * c)) & 255u) / 255.0f);
        res[p] = double(s / 3.0f);
    }
    std::vector<double> sorted = res;
    std::sort(sorted.begin(), sorted.end());
    double med = sorted[(hw - 1) / 2];
    double tCpu = std::max(med * 3.0, 0.10);
    addRow("transient masking", "threshold vs exact median rule", "64x48", std::abs(st.threshold - tCpu), 3.0 / 2048 + 1e-6);
    double thr = st.threshold;
    std::vector<int> out(hw), keep(hw);
    for (size_t p = 0; p < hw; p++) out[p] = T[p] < 0.05f && res[p] > thr;
    size_t dropped = 0, mismatch = 0;
    for (int ty = 0; ty < (H + 15) / 16; ty++)
        for (int tx = 0; tx < (W + 15) / 16; tx++) {
            int inl = 0;
            std::vector<size_t> px;
            for (int y = ty * 16; y < ty * 16 + 16; y++)
                for (int x = tx * 16; x < tx * 16 + 16; x++) {
                    if (x >= W || y >= H) { inl++; continue; }
                    int s = 0;
                    for (int dy = -1; dy <= 1; dy++)
                        for (int dx = -1; dx <= 1; dx++) {
                            int xx = x + dx, yy = y + dy;
                            if (xx >= 0 && yy >= 0 && xx < W && yy < H) s += out[size_t(yy) * W + size_t(xx)];
                        }
                    size_t p = size_t(y) * W + x;
                    keep[p] = s < 7;
                    inl += keep[p];
                    px.push_back(p);
                }
            bool patchOk = inl / 256.0 >= 0.6;
            for (size_t p : px) {
                if (patchOk) keep[p] = 1;
                dropped += !keep[p];
                mismatch += (keep[p] != 0) != (wgt[p] != 0.0f);
            }
        }
    addRow("transient masking", "mask pixels differing from the CPU rule", "64x48", double(mismatch), 0);
    addRow("transient masking", "outlier pixels dropped (info)", "64x48", double(dropped), 0, true);
    addRow("transient masking", "GPU outlier count vs dropped", "64x48", std::abs(double(st.outliers) - double(dropped)), 0);
    gs.feat = LossFeatures();
}

// Pose refinement: dL/d(the 9 correction numbers) from the GPU's camera gradient and the host chain rule, against
// finite differences of the double-precision reference render + loss with the corrected camera (frozen structure).
void testPose(Context& ctx, GpuSplat& gs, bool quick) {
    (void)ctx;
    Rng r(57);
    const int W = 64, H = 48;
    const double eye[3] = {1.5, -1.0, -3.2}, origin[3] = {0, 0, 0};
    Camera base = lookAt(eye, origin, 0.15, 58, 60, 31, 25, W, H);
    RenderOpts opt;
    opt.shDegree = 2;
    opt.antialiased = true;
    opt.bg[0] = 0.3f; opt.bg[1] = 0.2f; opt.bg[2] = 0.1f;
    Params P = randomScene(r, quick ? 150 : 300, eye, 2);
    std::vector<uint32_t> tgt = randomTarget(r, W, H);
    double d0[9];
    for (double& v : d0) v = 0.03 * r.normal();
    Camera cam = base;
    refinedW2c(base.w2c, d0, cam.w2c);
    gs.feat = LossFeatures();
    gs.feat.pose = true;
    gs.setParams(P);
    gs.setCamera(cam, opt);
    gs.setTarget(tgt);
    gs.forward();
    gs.loss();
    gs.rasterBackward();
    gs.projectBackward();
    double g15[15], gd[9];
    gs.readPoseGrad(g15);
    refinedW2c(base.w2c, d0, cam.w2c);
    poseChain(base.w2c, d0, g15, gd);
    // reference with the structure decided at d0
    ref::Cam rc = ref::makeCam(cam, opt);
    ref::ParamsD PD = ref::toDouble(P);
    std::vector<ref::Proj> cpuP = ref::projectAll(PD, rc, true);
    ref::Structure stC;
    std::vector<double> im, fT;
    std::vector<uint32_t> nC;
    ref::rasterDecide(cpuP, rc, im, fT, nC, &stC);
    std::vector<double> tP = targetPlanes(tgt, W, H);
    auto L = [&](const double* d) {
        Camera c2 = base;
        refinedW2c(base.w2c, d, c2.w2c);
        ref::Cam rc2 = ref::makeCam(c2, opt);
        std::vector<ref::Proj> pr = cpuP;
        for (int i = 0; i < PD.n; i++) if (pr[size_t(i)].valid) ref::project(PD, rc2, i, pr[size_t(i)], false);
        std::vector<double> img;
        ref::rasterFrozen(pr, stC, rc2, img);
        return ref::loss(img, tP, W, H, opt.ssimWeight);
    };
    std::vector<double> ga, fd;
    for (int j = 0; j < 9; j++) {
        double d[9];
        std::copy(d0, d0 + 9, d);
        double h = 1e-6;
        d[j] = d0[j] + h; double lp = L(d);
        d[j] = d0[j] - h; double lm = L(d);
        ga.push_back(gd[j]);
        fd.push_back((lp - lm) / (2 * h));
    }
    addRow("pose refinement (preprocess_bwd)", "dL/d(translation, 6D rotation)", "64x48", relErr(ga, fd, 5e-2), 2e-2);
    gs.feat = LossFeatures();
}

// optimizerPrecision "compact": the GPU kernels against the host reference (same hash random numbers, so the same
// codes up to float rounding at a decision boundary), the carrier round trip through getMoments / setState, the
// unbiasedness of the stochastic rounding, and the update against exact fp32 Adam.
void testCompactAdam(Context& ctx, GpuSplat& gs) {
    (void)ctx;
    Rng r(73);
    const int n = 64, steps = 5;
    Params P;
    P.resize(n);
    for (int k = 0; k < 5; k++) for (float& v : P.group(k)) v = float(r.uni(-1, 1));
    std::array<AdamGroup, 5> groups;
    for (int k = 0; k < 4; k++) groups[size_t(k)] = {float(1e-3 * (k + 1)), float(1e-3 * (k + 1)), 1, 1};
    groups[4] = {2.5e-3f, 1.25e-4f, 48, 3};
    gs.compactMoments = true;
    gs.compactSeed = 99;
    gs.setParams(P);
    Params hp = P;   // host reference
    std::vector<std::vector<uint32_t>> hw(5);
    for (int k = 0; k < 4; k++) hw[size_t(k)].assign(P.group(k).size(), 0u);
    hw[4].assign(size_t(n) * compact::SH_WORDS, 0u);
    std::vector<double> exactP[5], em[5], ev[5];   // exact fp32-moment Adam, for comparison
    for (int k = 0; k < 5; k++) { exactP[k].assign(P.group(k).begin(), P.group(k).end()); em[k].assign(exactP[k].size(), 0); ev[k].assign(exactP[k].size(), 0); }
    float bound = float(compact::ratioBound(0.9, 0.999));
    for (int s = 1; s <= steps; s++) {
        compact::StepParams sp;
        sp.bc1 = 1.0f - std::pow(0.9f, float(s));
        sp.bc2 = 1.0f - std::pow(0.999f, float(s));
        sp.step = uint32_t(s);
        for (int k = 0; k < 5; k++) {
            std::vector<float> g(P.group(k).size());
            for (size_t i = 0; i < g.size(); i++) g[i] = float(r.normal() * std::pow(10.0, -r.uni(0, 4)));   // many magnitudes
            ctx.upload(gs.grads[k], g.data(), g.size() * 4);
            sp.seed = 99u ^ (0x9E3779B9u * uint32_t(k + 1));
            sp.lr = groups[size_t(k)].lr;
            sp.lr2 = groups[size_t(k)].lr2;
            if (k < 4) compact::refBf16(hp.group(k).data(), g.data(), hw[size_t(k)].data(), g.size(), sp);
            else for (int i = 0; i < n; i++)
                compact::refSHRow(&hp.sh[size_t(i) * 48], &g[size_t(i) * 48], &hw[4][size_t(i) * compact::SH_WORDS], uint32_t(i), sp, bound);
            double bc1 = 1 - std::pow(0.9, s), bc2 = 1 - std::pow(0.999, s);
            for (size_t i = 0; i < g.size(); i++) {
                em[k][i] = 0.9 * em[k][i] + 0.1 * g[i];
                ev[k][i] = 0.999 * ev[k][i] + 0.001 * double(g[i]) * g[i];
                double lr = k == 4 ? (i % 48 < 3 ? groups[4].lr : groups[4].lr2) : groups[size_t(k)].lr;
                exactP[k][i] -= lr * (em[k][i] / bc1) / (std::sqrt(ev[k][i] / bc2) + 1e-15);
            }
        }
        gs.adamStep(groups);
    }
    Params gp, c1, c2;
    gs.getParams(gp);
    std::vector<double> ga, ha, ea;
    for (int k = 0; k < 5; k++)
        for (size_t i = 0; i < gp.group(k).size(); i++) {
            ga.push_back(double(gp.group(k)[i]) - P.group(k)[i]);
            ha.push_back(double(hp.group(k)[i]) - P.group(k)[i]);
            ea.push_back(exactP[k][i] - P.group(k)[i]);
        }
    // The GPU's exp2 / log2 / FMA differ from the host's in the last bits, so a stochastic rounding decision right at
    // its threshold can go the other way (one code level, then a slightly different trajectory for that value).
    {
        double mx = 0;
        for (double v : ha) mx = std::max(mx, std::abs(v));
        std::vector<double> e;
        for (size_t i = 0; i < ga.size(); i++) e.push_back(std::abs(ga[i] - ha[i]) / std::max(std::abs(ha[i]), 1e-2 * mx));
        std::vector<double> s = e;
        std::sort(s.begin(), s.end());
        double above = double(std::count_if(e.begin(), e.end(), [](double v) { return v > 1e-3; })) / double(e.size());
        addRow("compact adam", "param change vs host: median rel. error", "64 splats, 5 steps", s[s.size() / 2], 1e-5);
        addRow("compact adam", "param change vs host: fraction > 1e-3", "64 splats, 5 steps", above, 0.01);
        addRow("compact adam", "param change vs host: max (info)", "64 splats, 5 steps", s.back(), 0, true);
    }
    addRow("compact adam", "param change vs fp32 Adam: max (info)", "64 splats, 5 steps", relErr(ga, ea), 0, true);
    size_t words = 0, diff = 0;
    for (int k = 0; k < 5; k++) {
        size_t cnt = k < 4 ? P.group(k).size() : size_t(n) * compact::SH_WORDS;
        std::vector<uint32_t> w = gs.read<uint32_t>(gs.m1[k], cnt);
        for (size_t i = 0; i < cnt; i++) {
            if (k == 4 && i % compact::SH_WORDS == 26) continue;   // the fp32 row scale: compared by its codes' effect
            words++;
            diff += w[i] != hw[size_t(k)][i];
        }
    }
    addRow("compact adam", "code words differing from host", "64 splats, 5 steps", double(diff) / double(words), 0.005);
    // carriers: GPU → host → GPU → host is exact
    gs.getMoments(c1, c2);
    gs.setState(gp, &c1, &c2, gs.adamSteps);
    Params d1, d2;
    gs.getMoments(d1, d2);
    double rt = 0;
    for (int k = 0; k < 5; k++)
        for (size_t i = 0; i < c1.group(k).size(); i++) rt += (c1.group(k)[i] != d1.group(k)[i]) + (c2.group(k)[i] != d2.group(k)[i]);
    addRow("compact adam", "carrier round trip mismatches", "64 splats", rt, 0);
    // stochastic rounding is unbiased: mean decoded v and m over many seeds of one quantisation
    {
        double sumV = 0, sumM = 0;
        const int trials = 4000;
        float pr[48], g[48];
        for (int j = 0; j < 48; j++) { g[j] = 0; }
        g[3 + 7] = 3e-3f; g[3 + 8] = -2e-5f; g[3 + 9] = 1.0f;   // one row: very different magnitudes
        for (int t = 0; t < trials; t++) {
            uint32_t w[compact::SH_WORDS] = {};
            compact::StepParams sp;
            sp.bc1 = 0.1f; sp.bc2 = 0.001f; sp.seed = uint32_t(t); sp.step = 1;
            for (float& v : pr) v = 0;
            compact::refSHRow(pr, g, w, 0, sp, bound);
            double m[48], v[48];
            compact::decodedSHMoments(w, bound, m, v);
            sumV += v[3 + 8];
            sumM += m[3 + 7];
        }
        double vTrue = 0.001 * 2e-5 * 2e-5, mTrue = 0.1 * 3e-3;
        addRow("compact adam", "stochastic rounding bias, small v (relative)", "4000 trials", std::abs(sumV / trials / vTrue - 1), 0.03);
        addRow("compact adam", "stochastic rounding bias, m (relative)", "4000 trials", std::abs(sumM / trials / mTrue - 1), 0.03);
    }
    gs.compactMoments = false;
    gs.setParams(P);
}

// raster_bwd's extra dL/dT_final plane: a background b is the same as bg = 0 with plane 4 = b · dL/drgb.
void testExtraTPlane(Context& ctx, GpuSplat& gs) {
    Rng r(17);
    const int W = 64, H = 48;
    size_t hw = size_t(W) * H;
    const double eye[3] = {0.4, 0.3, -3.6}, origin[3] = {0, 0, 0};
    Camera cam = lookAt(eye, origin, -0.1, 55, 55, 32, 24, W, H);
    Params P = randomScene(r, 200, eye, 1);
    RenderOpts opt;
    opt.shDegree = 1;
    opt.bg[0] = 0.7f; opt.bg[1] = 0.4f; opt.bg[2] = 0.9f;
    gs.feat = LossFeatures();
    gs.setParams(P);
    gs.setCamera(cam, opt);
    gs.forward();
    std::vector<float> G(5 * hw, 0.0f);
    for (size_t i = 0; i < 4 * hw; i++) G[i] = float(r.uni(-1, 1));
    ctx.upload(gs.gradImg, G.data(), G.size() * 4);
    gs.rasterBackward();
    std::vector<float> a1 = gs.read<float>(gs.acc, size_t(gs.n) * 12);
    RenderOpts o2 = opt;
    o2.bg[0] = o2.bg[1] = o2.bg[2] = 0.0f;
    gs.setCamera(cam, o2);
    for (size_t p = 0; p < hw; p++) G[4 * hw + p] = opt.bg[0] * G[p] + opt.bg[1] * G[hw + p] + opt.bg[2] * G[2 * hw + p];
    ctx.upload(gs.gradImg, G.data(), G.size() * 4);
    gs.rasterBackward();
    std::vector<float> a2 = gs.read<float>(gs.acc, size_t(gs.n) * 12);
    std::vector<double> x(a2.begin(), a2.end()), y(a1.begin(), a1.end());
    addRow("rasterize bwd", "extra dL/dT plane = background term", "64x48", relErr(x, y), 1e-4);
}

// Host-side strategies: counts and invariants (they run on the CPU, once per test run).
void testStrategies() {
    g_path = "host";
    HostRng rng(5);
    std::vector<float> xyz, rgb;
    Rng r(77);
    for (int i = 0; i < 1000; i++) {
        for (int k = 0; k < 3; k++) xyz.push_back(float(r.uni(-1, 1)));
        for (int k = 0; k < 3; k++) rgb.push_back(float(r.uni(0, 1)));
    }
    SplatState s = initSplats(xyz, rgb, 0.1f, rng);
    // kNN: brute force on a few points
    std::vector<float> kd = knnMeanDist(xyz);
    double kerr = 0;
    for (int i = 0; i < 1000; i += 97) {
        std::vector<double> d;
        for (int j = 0; j < 1000; j++) {
            if (j == i) continue;
            double s2 = 0;
            for (int k = 0; k < 3; k++) s2 += double(xyz[size_t(i) * 3 + k] - xyz[size_t(j) * 3 + k]) * (xyz[size_t(i) * 3 + k] - xyz[size_t(j) * 3 + k]);
            d.push_back(std::sqrt(s2));
        }
        std::sort(d.begin(), d.end());
        kerr = std::max(kerr, std::abs((d[0] + d[1] + d[2]) / 3 - kd[size_t(i)]) / kd[size_t(i)]);
    }
    addRow("init (kd-tree kNN)", "mean 3-NN distance vs brute force", "1000 points", kerr, 1e-5);
    // MCMC: a third dead → all relocated, 5 % added, no dead left, moments of the new rows zero
    for (int i = 0; i < s.n(); i++) {
        s.p.opLogit[size_t(i)] = i % 3 == 0 ? -8.0f : float(r.uni(-2, 3));
        for (float* m : {&s.m1.pos[size_t(i) * 3], &s.m2.pos[size_t(i) * 3]}) *m = 1.0f;
    }
    McmcParams mp;
    mp.capMax = 100000;
    McmcResult mr = mcmcRefine(s, mp, rng);
    int deadLeft = 0, newMomentsNonzero = 0;
    for (int i = 0; i < s.n(); i++) if (1 / (1 + std::exp(-s.p.opLogit[size_t(i)])) <= 0.005) deadLeft++;
    for (int i = 1000; i < s.n(); i++) if (s.m1.pos[size_t(i) * 3] != 0) newMomentsNonzero++;
    addRow("mcmc refine (host)", "relocated - dead, added - 50", "1000 splats", std::abs(double(mr.relocated) - 334) + std::abs(double(mr.added) - 50), 0);
    addRow("mcmc refine (host)", "dead left + new rows with moments", "1000 splats", deadLeft + newMomentsNonzero, 0);
    // relocation keeps the opacity of a group: 20 splats grow by one copy, 1 − (1 − o_new)² = o_old
    {
        std::vector<float> pts, col;
        for (int i = 0; i < 20; i++) {
            for (int k = 0; k < 3; k++) pts.push_back(float(r.uni(-1, 1)));
            for (int k = 0; k < 3; k++) col.push_back(0.5f);
        }
        SplatState t = initSplats(pts, col, 0.6f, rng);
        McmcParams cap;
        cap.capMax = 1000;
        mcmcRefine(t, cap, rng);
        double o = 1 / (1 + std::exp(-double(t.p.opLogit[20])));
        addRow("mcmc relocation (eq. 9)", "1-(1-o_new)^2 - o_old", "20 splats", t.n() == 21 ? std::abs(1 - (1 - o) * (1 - o) - 0.6) : 1.0, 1e-6);
        size_t moved = spreadCloneGroups(t);
        bool apart = true;
        for (int i = 0; i < 20; i++)
            if (t.p.pos[size_t(i) * 3] == t.p.pos[60] && t.p.pos[size_t(i) * 3 + 1] == t.p.pos[61]) apart = false;
        addRow("long-axis split", "clone pair spread apart", "20 splats", moved == 2 && apart ? 0.0 : 1.0, 0);
    }
    // adaptive: high gradient → clone small, split large; prune transparent
    {
        SplatState a = initSplats(xyz, rgb, 0.1f, rng);
        std::vector<float> g(1000, 0.0f), c(1000, 1.0f);
        int wantDup = 0, wantSplit = 0, wantPrune = 0;
        for (int i = 0; i < 1000; i++) {
            for (int k = 0; k < 3; k++) a.p.logScale[size_t(i) * 3 + k] = -4.0f;
            if (i % 4 == 0) { g[size_t(i)] = 1e-3f; if (i % 8 == 0) { for (int k = 0; k < 3; k++) a.p.logScale[size_t(i) * 3 + k] = -1.0f; wantSplit++; } else wantDup++; }
            if (i % 10 == 5) { a.p.opLogit[size_t(i)] = -8.0f; wantPrune++; }
        }
        AdcParams ap;
        ap.growGrad2d = 2e-4;
        ap.sceneScale = 10.0;   // grow_scale3d · scene = 0.1: exp(-1) is large, exp(-4) small
        int n0 = a.n();
        AdcResult ar = adcRefine(a, g, c, ap, 600, rng);
        double err = std::abs(double(ar.duplicated) - wantDup) + std::abs(double(ar.split) - wantSplit) + std::abs(double(ar.pruned) - wantPrune) +
                     std::abs(double(a.n()) - (n0 + wantDup + wantSplit - wantPrune));
        addRow("adaptive refine (host)", "duplicated, split, pruned, count", "1000 splats", err, 0);
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string path = "both";
    int device = -1;
    bool quick = false;
    int fitSteps = 300;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); } return argv[++i]; };
        if (a == "--path") path = next();
        else if (a == "--device") device = std::stoi(next());
        else if (a == "--quick") quick = true;
        else if (a == "--fit-steps") fitSteps = std::stoi(next());
        else if (a == "-h" || a == "--help") {
            std::printf("usage: vktrain-selftest [--path auto|native|cas|both] [--device N] [--quick] [--fit-steps N]\n");
            return 0;
        } else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::pair<std::string, ContextOptions::Atomics>> paths;
    bool deviceHasNative = false, nativeBroken = false;
    try {
        ContextOptions probe;
        probe.deviceIndex = device;
        Context ctx(probe);   // auto mode: probes native float atomics
        deviceHasNative = ctx.info.floatAtomicAdd;
        nativeBroken = ctx.info.nativeAtomicsBroken;
        std::printf("device: %s (%s), Vulkan %u.%u, subgroup size %u, shaderBufferFloat32AtomicAdd %s%s, host-visible device memory %s\n",
                    ctx.info.name.c_str(), ctx.info.driver.c_str(), VK_API_VERSION_MAJOR(ctx.info.apiVersion),
                    VK_API_VERSION_MINOR(ctx.info.apiVersion), ctx.info.subgroupSize, deviceHasNative ? "yes" : "no",
                    nativeBroken ? " (but the probe got wrong sums: the trainer uses the CAS kernels)" : "",
                    ctx.info.hostVisibleDeviceMemory ? "yes" : "no");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "vktrain-selftest: no usable Vulkan device: %s\n", e.what());
        return 3;
    }
    if (path == "native" && !deviceHasNative) {
        std::fprintf(stderr, "vktrain-selftest: the device has no float atomic add; --path native is not possible\n");
        return 3;
    }
    bool nativeUsable = deviceHasNative && !nativeBroken;
    if (path == "native" || ((path == "both" || path == "auto") && nativeUsable)) paths.push_back({"native", ContextOptions::Atomics::Native});
    if (path == "cas" || path == "both" || (path == "auto" && !nativeUsable)) paths.push_back({"cas", ContextOptions::Atomics::Cas});
    if (path == "both" && !deviceHasNative) std::printf("note: the device has no float atomic add: only the CAS path is tested\n");
    if (path != "native" && path != "cas" && nativeBroken) {
        std::printf("note: the driver advertises float atomic add but loses updates (probe); the native path is skipped\n"
                    "      (force it with --path native). The trainer falls back to the CAS kernels on this driver.\n");
        g_path = "native";
        addRow("float atomic add (startup probe)", "driver advertises it, sums wrong", "-", 1, 0, true);
    }

    std::vector<Case> cases = {
        {"A:n50,sh0,64x48", 50, 0, false, 64, 48, {0, 0, -4}, 0.0, 60, 60, 32, 24, {0, 0, 0}, 1},
        {"B:n200,sh3,aa,roll,offc", 200, 3, true, 64, 48, {2.5, -1.5, -2.8}, 0.4, 58, 62, 29.5, 26, {0.2f, 0.4f, 0.6f}, 2},
        {"C:n500,sh2,70x45", 500, 2, false, 70, 45, {-1.0, 0.5, 3.5}, -0.25, 66, 60, 36, 21, {0.5f, 0.5f, 0.5f}, 3},
    };
    int rc = 0;
    try {
        for (auto& [name, atomics] : paths) {
            g_path = name;
            auto tp = std::chrono::steady_clock::now();
            ContextOptions o;
            o.deviceIndex = device;
            o.atomics = atomics;
            Context ctx(o);
            GpuSplat gs(ctx);
            std::printf("path %s:\n", name.c_str());
            testPrimitives(ctx, gs, quick);
            for (const Case& c : cases) runCase(ctx, gs, c, quick);
            testAdam(ctx, gs);
            testLossVariants(ctx, gs);
            testTrainingKernels(ctx, gs);
            testExtraTPlane(ctx, gs);
            testImagePipeline(ctx, gs, quick);
            testTransient(ctx, gs);
            testPose(ctx, gs, quick);
            testCompactAdam(ctx, gs);
            testFit(gs, quick ? std::min(fitSteps, 150) : fitSteps);
            std::printf("  path %s took %.1f s\n", name.c_str(),
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - tp).count());
        }
        testStrategies();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "vktrain-selftest: error: %s\n", e.what());
        rc = 4;
    }

    std::printf("\n%-36s %-34s %-26s %-6s %11s %9s  %s\n", "kernel", "quantity", "scene", "path", "max err", "tol", "status");
    int failed = 0;
    for (const Row& row : g_rows) {
        const char* st = row.infoOnly ? "info" : row.pass() ? "ok" : "FAIL";
        if (!row.pass()) failed++;
        std::printf("%-36s %-34s %-26s %-6s %11.3e %9.1e  %s\n", row.kernel.c_str(), row.quantity.c_str(), row.scene.c_str(),
                    row.path.c_str(), row.err, row.tol, st);
    }
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("\n%zu checks, %d failed, %.1f s\n", g_rows.size(), failed, secs);
    if (rc) return rc;
    return failed ? 1 : 0;
}
