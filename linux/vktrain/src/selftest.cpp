// vktrain-selftest: checks every stage-1 kernel of the Vulkan trainer against a double-precision CPU reference
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
            testFit(gs, quick ? std::min(fitSteps, 150) : fitSteps);
            std::printf("  path %s took %.1f s\n", name.c_str(),
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - tp).count());
        }
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
