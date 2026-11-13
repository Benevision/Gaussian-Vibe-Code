#include "gpu_splat.h"

#include "compact.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace vkt {

namespace {
constexpr uint32_t RADIX_BLOCK = 256 * 8;   // keys per radix workgroup (shaders: 256 threads × RADIX_CHUNKS)
constexpr uint32_t SCAN_BLOCK = 1024;       // values per scan workgroup
uint32_t ceilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
}  // namespace

GpuSplat::GpuSplat(Context& ctx) : ctx_(ctx) {
    double s = 0;
    for (int k = 0; k < 11; k++) s += ssimWindow_[k] = float(std::exp(-double((k - 5) * (k - 5)) / (2.0 * 1.5 * 1.5)));
    for (float& w : ssimWindow_) w = float(w / s);
    frame = ctx_.createBuffer(sizeof(Frame));
    stats = ctx_.createBuffer(16 * sizeof(float));
    hist = ctx_.createBuffer(2048 * 4);
    tstate = ctx_.createBuffer(4 * 4);
    corrBuf = ctx_.createBuffer(size_t(BILAGRID_SIZE) * 4);
    corrGrad = ctx_.createBuffer(size_t(std::max<int>(BILAGRID_SIZE, int(CORR_STRIPES) * 12)) * 4);
    poseAcc = ctx_.createBuffer(size_t(POSE_STRIPES) * 16 * 4);
    dummy = ctx_.createBuffer(16);
    ctx_.begin();
    for (Buffer* b : {&hist, &tstate, &corrBuf, &corrGrad, &poseAcc, &dummy}) ctx_.fill(*b, 0);
    ctx_.submitWait();
}

GpuSplat::~GpuSplat() {
    Buffer* all[] = {&frame, &p2d, &conicO, &rgbd, &rects, &tilesTouched, &depthKeys, &sortVals, &offsets, &tileKeys,
                     &tileVals, &ranges, &img, &finalT, &nContrib, &target, &planes, &mu, &tmp, &dmaps, &stats,
                     &gradImg, &acc, &sortKeysB_, &sortValsB_, &hist_, &raw, &gradRaw, &aux, &weight, &resid, &dED, &nbuf,
                     &hist, &tstate, &corrBuf, &corrGrad, &poseAcc, &depthT, &dummy};
    for (Buffer* b : all) ctx_.destroy(*b);
    for (int k = 0; k < 5; k++) {
        ctx_.destroy(params[k]); ctx_.destroy(grads[k]); ctx_.destroy(m1[k]); ctx_.destroy(m2[k]);
    }
    for (Buffer& b : scanLevels_) ctx_.destroy(b);
}

void GpuSplat::dispatch1D(const char* kernel, const std::vector<const Buffer*>& bufs, const void* pc, uint32_t pcSize, uint32_t groups) {
    if (groups == 0) return;
    uint32_t gx = std::min<uint32_t>(groups, 65535u);
    uint32_t gy = ceilDiv(groups, gx);
    ctx_.dispatch(kernel, bufs, pc, pcSize, gx, gy, 1);
}

void GpuSplat::setParams(const Params& p) {
    if (p.n != n || !params[0].buf || allocCompact_ != compactMoments) {
        n = p.n;
        allocCompact_ = compactMoments;
        for (int k = 0; k < 5; k++) {
            size_t bytes = size_t(n) * Params::width(k) * sizeof(float);
            for (Buffer* b : {&params[k], &grads[k], &m1[k], &m2[k]}) ctx_.destroy(*b);
            params[k] = ctx_.createBuffer(bytes);
            grads[k] = ctx_.createBuffer(bytes);
            if (!compactMoments) {
                m1[k] = ctx_.createBuffer(bytes);
                m2[k] = ctx_.createBuffer(bytes);
            } else {   // compact.h: one word per value (bf16 pair), SH rows of 27 words; m2 unused
                m1[k] = ctx_.createBuffer(size_t(std::max(n, 1)) * (k == 4 ? compact::SH_WORDS : Params::width(k)) * 4);
                m2[k] = ctx_.createBuffer(16);
            }
        }
        for (Buffer* b : {&p2d, &conicO, &rgbd, &rects, &tilesTouched, &depthKeys, &sortVals, &offsets, &acc}) ctx_.destroy(*b);
        size_t nn = size_t(std::max(n, 1));
        p2d = ctx_.createBuffer(nn * 8);
        conicO = ctx_.createBuffer(nn * 16);
        rgbd = ctx_.createBuffer(nn * 16);
        rects = ctx_.createBuffer(nn * 16);
        tilesTouched = ctx_.createBuffer(nn * 4);
        depthKeys = ctx_.createBuffer(nn * 4);
        sortVals = ctx_.createBuffer(nn * 4);
        offsets = ctx_.createBuffer((nn + 1) * 4);
        acc = ctx_.createBuffer(nn * 12 * 4);
    }
    for (int k = 0; k < 5; k++) ctx_.upload(params[k], p.group(k).data(), p.group(k).size() * sizeof(float));
    ctx_.begin();
    for (int k = 0; k < 5; k++) { ctx_.fill(m1[k], 0); ctx_.fill(m2[k], 0); ctx_.fill(grads[k], 0); }
    ctx_.submitWait();
    adamSteps = 0;
    frameData.count = uint32_t(n);
    ctx_.upload(frame, &frameData, sizeof(Frame));
}

void GpuSplat::getParams(Params& p) const {
    p.resize(n);
    for (int k = 0; k < 5; k++) ctx_.download(params[k], p.group(k).data(), p.group(k).size() * sizeof(float));
}

void GpuSplat::getGrads(Params& g) const {
    g.resize(n);
    for (int k = 0; k < 5; k++) ctx_.download(grads[k], g.group(k).data(), g.group(k).size() * sizeof(float));
}

void GpuSplat::setCamera(const Camera& cam, const RenderOpts& opt) {
    frameData = makeFrame(cam, opt, n);
    if (cam.width != width || cam.height != height || !img.buf) {
        width = cam.width; height = cam.height;
        tilesX = int(frameData.tilesX); tilesY = int(frameData.tilesY);
        size_t hw = size_t(width) * height;
        for (Buffer* b : {&img, &finalT, &nContrib, &target, &planes, &mu, &tmp, &dmaps, &gradImg, &ranges, &raw, &gradRaw, &aux,
                          &weight, &resid, &dED, &nbuf})
            ctx_.destroy(*b);
        img = ctx_.createBuffer(hw * 4 * 4);
        finalT = ctx_.createBuffer(hw * 4);
        nContrib = ctx_.createBuffer(hw * 4);
        target = ctx_.createBuffer(hw * 4);
        gradImg = ctx_.createBuffer(hw * 5 * 4);
        ranges = ctx_.createBuffer(size_t(tilesX) * tilesY * 2 * 4);
        ctx_.begin();
        ctx_.fill(gradImg, 0);   // plane 4 (extra dL/dT) stays zero for callers that upload only planes 0–3
        ctx_.submitWait();
    }
    ensureFeatureBuffers();
    ctx_.upload(frame, &frameData, sizeof(Frame));
}

void GpuSplat::ensureFeatureBuffers() {
    if (width == 0 || height == 0) return;
    size_t hw = size_t(width) * height;
    bool split = feat.decoupledSSIM && feat.corr != LossFeatures::None;
    size_t np = split ? 18 : 15;
    ctx_.ensure(planes, hw * np * 4);
    ctx_.ensure(mu, hw * np * 4);
    ctx_.ensure(tmp, hw * np * 4);
    ctx_.ensure(dmaps, hw * (split ? 12 : 9) * 4);
    if (feat.corr != LossFeatures::None) { ctx_.ensure(raw, hw * 3 * 4); ctx_.ensure(gradRaw, hw * 3 * 4); }
    if (feat.sky) ctx_.ensure(aux, hw * 4);
    if (feat.transient) { ctx_.ensure(weight, hw * 4); ctx_.ensure(resid, hw * 4); }
    if (feat.anyDepth()) { ctx_.ensure(dED, hw * 4); ctx_.ensure(nbuf, hw * 4 * 4); }
}

void GpuSplat::setCorrection(const float* params, size_t count) {
    if (count != 12 && count != size_t(BILAGRID_SIZE)) throw Error("colour correction: 12 or 24576 parameters expected");
    ctx_.upload(corrBuf, params, count * 4);
    corrCount_ = int(count);
}

void GpuSplat::setAux(const std::vector<uint32_t>& a) {
    if (a.size() != size_t(width) * height) throw Error("aux size does not match the camera");
    ctx_.ensure(aux, a.size() * 4);
    ctx_.upload(aux, a.data(), a.size() * 4);
}

void GpuSplat::setDepthTarget(const float* d, int w, int h, int viewW, int viewH) {
    ctx_.ensure(depthT, size_t(w) * h * 4);
    ctx_.upload(depthT, d, size_t(w) * h * 4);
    depthW_ = uint32_t(w); depthH_ = uint32_t(h); viewW_ = uint32_t(viewW); viewH_ = uint32_t(viewH);
}

GpuSplat::SkyPC GpuSplat::skyPC() const {
    SkyPC pc{};
    for (int k = 0; k < 3; k++) { pc.r0[k] = feat.skyRot[k]; pc.r1[k] = feat.skyRot[3 + k]; pc.r2[k] = feat.skyRot[6 + k]; }
    pc.W = feat.skyW; pc.H = feat.skyH;
    pc.penalty = feat.skyPenalty;
    pc.hasMask = feat.skyMask ? 1u : 0u;
    return pc;
}

void GpuSplat::recordComposite() {
    uint32_t gx = ceilDiv(uint32_t(width), 16), gy = ceilDiv(uint32_t(height), 16);
    if (feat.sky) {
        SkyPC pc = skyPC();
        ctx_.dispatch("sky_composite", {&frame, &img, &finalT, feat.skyTex}, &pc, sizeof pc, gx, gy);
    }
    if (feat.corr != LossFeatures::None) {
        uint32_t mode = uint32_t(feat.corr);
        ctx_.dispatch("corr_apply", {&frame, &img, &raw, &corrBuf}, &mode, 4, gx, gy);
    }
}

uint32_t GpuSplat::forwardComposite() {
    ensureFeatureBuffers();
    uint32_t m = recordForwardPhase1();
    ctx_.begin();
    recordForwardPhase2(m);
    recordComposite();
    ctx_.submitWait();
    return m;
}

void GpuSplat::recordImageLoss() {
    uint32_t gx = ceilDiv(uint32_t(width), 16), gy = ceilDiv(uint32_t(height), 16);
    recordComposite();
    bool weights = false;
    if (feat.transient) {
        ctx_.fill(hist, 0);
        ctx_.dispatch("transient_hist", {&frame, &img, &target, &finalT, &resid, &hist}, nullptr, 0, gx, gy);
        float sens = feat.transientSensitivity;
        ctx_.dispatch("transient_thresh", {&frame, &hist, &tstate}, &sens, 4, 1, 1);
        if (feat.transientActive) {
            ctx_.dispatch("transient_weights", {&frame, &resid, &weight, &tstate}, nullptr, 0, gx, gy);
            weights = true;
        }
    }
    bool split = feat.decoupledSSIM && feat.corr != LossFeatures::None;
    uint32_t flags = (weights ? 1u : 0u) | (split ? 2u : 0u);
    ctx_.dispatch("ssim_prep", {&frame, &img, &target, &planes, weights ? &weight : &dummy, split ? &raw : &dummy}, &flags, 4, gx, gy);
    recordBlur(planes, mu, split ? 18 : 15);
    ctx_.fill(stats, 0);
    uint32_t sp = split ? 1u : 0u;
    ctx_.dispatch("ssim_terms", {&frame, &mu, &dmaps, &stats, &planes}, &sp, 4, gx, gy);
    recordBlur(dmaps, mu, split ? 12 : 9);   // blur adjoint of the SSIM derivative maps (mu reused)
    ctx_.dispatch("loss_grad", {&frame, &planes, &mu, &gradImg, split ? &gradRaw : &dummy}, &sp, 4, gx, gy);
    if (feat.anyDepth()) {
        if (!depthT.buf || depthW_ == 0) throw Error("depth losses without a depth map");
        struct { uint32_t sw, sh, vw, vh; float sx, sy, depthScale, normalScale; uint32_t edgeAware; int32_t k; uint32_t pad0, pad1; } pc{};
        pc.sw = depthW_; pc.sh = depthH_; pc.vw = viewW_; pc.vh = viewH_;
        // torch F.interpolate(nearest) from the photo's size to this render's: source = floor(x · in/out) (float)
        pc.sx = uint32_t(width) == viewW_ ? 0.0f : float(viewW_) / float(width);
        pc.sy = uint32_t(height) == viewH_ ? 0.0f : float(viewH_) / float(height);
        pc.depthScale = feat.depthScale;
        pc.normalScale = feat.normalScale;
        pc.edgeAware = feat.edgeAware ? 1u : 0u;
        pc.k = feat.normalK;
        ctx_.dispatch("depth_normal", {&frame, &img, &finalT, &target, &depthT, &dED, &nbuf, &stats}, &pc, sizeof pc, gx, gy);
        ctx_.dispatch("depth_gather", {&frame, &img, &finalT, &dED, &nbuf, &gradImg}, nullptr, 0, gx, gy);
    }
}

void GpuSplat::recordImageBackward() {
    uint32_t gx = ceilDiv(uint32_t(width), 16), gy = ceilDiv(uint32_t(height), 16);
    if (feat.corr != LossFeatures::None) {
        bool split = feat.decoupledSSIM;
        ctx_.fill(corrGrad, 0);
        struct { uint32_t mode, direct, stripes; } pc{uint32_t(feat.corr), split ? 1u : 0u, CORR_STRIPES};
        ctx_.dispatch("corr_backward", {&frame, &gradImg, &raw, &corrBuf, split ? &gradRaw : &dummy, &corrGrad}, &pc, sizeof pc, gx, gy);
    }
    if (feat.sky) {
        if (feat.skyGrad) ctx_.fill(*feat.skyGrad, 0);
        SkyPC pc = skyPC();
        ctx_.dispatch("sky_backward", {&frame, &gradImg, &finalT, feat.skyTex, feat.skyMask ? &aux : &dummy,
                                       feat.skyGrad ? feat.skyGrad : &dummy}, &pc, sizeof pc, gx, gy);
    }
}

double GpuSplat::imageLoss(double* l1, double* ssim) {
    ensureFeatureBuffers();
    ctx_.begin();
    recordImageLoss();
    ctx_.submitWait();
    return lossFromStats(l1, ssim, nullptr);
}

void GpuSplat::imageBackward() {
    ctx_.begin();
    recordImageBackward();
    ctx_.submitWait();
}

std::vector<float> GpuSplat::readCorrGrad() const {
    if (feat.corr == LossFeatures::Grid) return read<float>(corrGrad, size_t(BILAGRID_SIZE));
    std::vector<float> s = read<float>(corrGrad, size_t(CORR_STRIPES) * 12), out(12, 0.0f);
    for (uint32_t k = 0; k < CORR_STRIPES; k++)
        for (int j = 0; j < 12; j++) out[size_t(j)] += s[size_t(k) * 12 + size_t(j)];
    return out;
}

void GpuSplat::readPoseGrad(double out[15]) const {
    std::vector<float> s = read<float>(poseAcc, size_t(POSE_STRIPES) * 16);
    for (int j = 0; j < 15; j++) out[j] = 0;
    for (uint32_t k = 0; k < POSE_STRIPES; k++)
        for (int j = 0; j < 15; j++) out[j] += s[size_t(k) * 16 + size_t(j)];
}

GpuSplat::TransientState GpuSplat::transientState() const {
    std::vector<uint32_t> s = read<uint32_t>(tstate, 3);
    TransientState t;
    std::memcpy(&t.threshold, &s[0], 4);
    t.hasThreshold = s[1] != 0;
    t.outliers = s[2];
    return t;
}

void GpuSplat::setTransientState(float threshold, bool has) {
    uint32_t s[3] = {0, has ? 1u : 0u, 0};
    std::memcpy(&s[0], &threshold, 4);
    ctx_.upload(tstate, s, sizeof s);
}

void GpuSplat::setTarget(const std::vector<uint32_t>& rgba) {
    if (rgba.size() != size_t(width) * height) throw Error("target size does not match the camera");
    ctx_.upload(target, rgba.data(), rgba.size() * 4);
}

// ------------------------------------------------------------------ scan and sort

void GpuSplat::reserveScan(uint32_t count) {
    size_t level = 0;
    uint32_t c = count;
    while (c > SCAN_BLOCK) {
        uint32_t nb = ceilDiv(c, SCAN_BLOCK);
        if (scanLevels_.size() <= level) scanLevels_.emplace_back();
        ctx_.ensure(scanLevels_[level], size_t(nb) * 4);
        c = nb;
        level++;
    }
}

void GpuSplat::recordScan(Buffer& data, uint32_t count, size_t level) {
    if (count == 0) return;
    uint32_t nb = ceilDiv(count, SCAN_BLOCK);
    if (nb == 1) {
        uint32_t pc[2] = {count, 0};
        dispatch1D("scan_block", {&data, &data}, pc, 8, 1);
        return;
    }
    Buffer& sums = scanLevels_.at(level);
    uint32_t pc[2] = {count, 1};
    dispatch1D("scan_block", {&data, &sums}, pc, 8, nb);
    recordScan(sums, nb, level + 1);
    uint32_t pa[1] = {count};
    dispatch1D("scan_add", {&data, &sums}, pa, 4, ceilDiv(count, 256));
}

void GpuSplat::reserveSort(uint32_t count) {
    uint32_t c = std::max<uint32_t>(count, 1);
    ctx_.ensure(sortKeysB_, size_t(c) * 4);
    ctx_.ensure(sortValsB_, size_t(c) * 4);
    uint32_t nb = ceilDiv(c, RADIX_BLOCK);
    ctx_.ensure(hist_, size_t(256) * nb * 4);
    reserveScan(256 * nb);
}

void GpuSplat::recordSort(Buffer& keys, Buffer& vals, uint32_t count, int bits) {
    if (count <= 1) return;
    int passes = (bits + 7) / 8;
    uint32_t nb = ceilDiv(count, RADIX_BLOCK);
    Buffer *inK = &keys, *inV = &vals, *outK = &sortKeysB_, *outV = &sortValsB_;
    for (int p = 0; p < passes; p++) {
        uint32_t pc[3] = {count, uint32_t(8 * p), nb};
        dispatch1D("radix_hist", {inK, &hist_}, pc, 12, nb);
        recordScan(hist_, 256 * nb);
        dispatch1D("radix_scatter", {inK, inV, outK, outV, &hist_}, pc, 12, nb);
        std::swap(inK, outK);
        std::swap(inV, outV);
    }
    if (passes % 2) {
        ctx_.copy(sortKeysB_, keys, size_t(count) * 4);
        ctx_.copy(sortValsB_, vals, size_t(count) * 4);
    }
}

void GpuSplat::sortPairs(Buffer& keys, Buffer& vals, uint32_t count, int bits) {
    reserveSort(count);
    ctx_.begin();
    recordSort(keys, vals, count, bits);
    ctx_.submitWait();
}

void GpuSplat::exclusiveScan(Buffer& data, uint32_t count) {
    reserveScan(count);
    ctx_.begin();
    recordScan(data, count);
    ctx_.submitWait();
}

int GpuSplat::atomicProbe(uint32_t count) { return ctx_.probeFloatAtomics(count); }

// ------------------------------------------------------------------ forward

uint32_t GpuSplat::recordForwardPhase1() {
    uint32_t un = uint32_t(n);
    reserveSort(un);
    reserveScan(un + 1);
    // Phase 1: projection, depth sort, tile counts and offsets (the total is read back to size the tile lists).
    ctx_.begin();
    dispatch1D("preprocess", {&frame, &params[0], &params[1], &params[2], &params[3], &params[4], &p2d, &conicO, &rgbd,
                              &rects, &tilesTouched, &depthKeys, &sortVals}, nullptr, 0, ceilDiv(un, 256));
    recordSort(depthKeys, sortVals, un, 32);
    dispatch1D("gather_counts", {&sortVals, &tilesTouched, &offsets}, &un, 4, ceilDiv(un + 1, 256));
    recordScan(offsets, un + 1);
    ctx_.submitWait();
    uint32_t m = 0;
    ctx_.download(offsets, &m, 4, size_t(un) * 4);
    tileCount = m;
    ctx_.ensure(tileKeys, size_t(std::max<uint32_t>(m, 1)) * 4);
    ctx_.ensure(tileVals, size_t(std::max<uint32_t>(m, 1)) * 4);
    reserveSort(m);
    return m;
}

void GpuSplat::recordForwardPhase2(uint32_t m) {
    // Phase 2: tile instances, stable sort by tile, per-tile ranges, rasterisation.
    uint32_t un = uint32_t(n);
    uint32_t numTiles = uint32_t(tilesX * tilesY);
    int bits = 1;
    while ((1u << bits) < numTiles && bits < 32) bits++;
    if (m > 0) {
        dispatch1D("duplicate_keys", {&frame, &sortVals, &offsets, &rects, &tileKeys, &tileVals}, &un, 4, ceilDiv(un, 256));
        recordSort(tileKeys, tileVals, m, bits);
    }
    ctx_.fill(ranges, 0);
    if (m > 0) dispatch1D("tile_ranges", {&tileKeys, &ranges}, &m, 4, ceilDiv(m, 256));
    ctx_.dispatch("raster_fwd", {&frame, &ranges, &tileVals, &p2d, &conicO, &rgbd, &img, &finalT, &nContrib}, nullptr, 0,
                  uint32_t(tilesX), uint32_t(tilesY));
}

uint32_t GpuSplat::forward() {
    uint32_t m = recordForwardPhase1();
    ctx_.begin();
    recordForwardPhase2(m);
    ctx_.submitWait();
    return m;
}

double GpuSplat::forwardLoss(double* l1, double* ssim, double* sqErr) {
    ensureFeatureBuffers();
    uint32_t m = recordForwardPhase1();
    ctx_.begin();
    recordForwardPhase2(m);
    recordImageLoss();
    ctx_.submitWait();
    return lossFromStats(l1, ssim, sqErr);
}

// ------------------------------------------------------------------ loss

void GpuSplat::recordBlur(Buffer& src, Buffer& dst, uint32_t planesCount) {
    struct { uint32_t dir, pad[3]; float w[11]; } pc{};
    std::memcpy(pc.w, ssimWindow_, sizeof(pc.w));
    uint32_t gx = ceilDiv(uint32_t(width), 16), gy = ceilDiv(uint32_t(height), 16);
    pc.dir = 0;
    ctx_.dispatch("blur", {&frame, &src, &tmp}, &pc, sizeof(pc), gx, gy, planesCount);
    pc.dir = 1;
    ctx_.dispatch("blur", {&frame, &tmp, &dst}, &pc, sizeof(pc), gx, gy, planesCount);
}

double GpuSplat::lossFromStats(double* l1, double* ssim, double* sqErr) {
    float s[8];
    ctx_.download(stats, s, sizeof(s));
    for (int k = 0; k < 8; k++) lastStats_[k] = s[k];
    double numel = double(width) * height * 3;
    double snumel = 3.0 / double(frameData.invSsimPixels);
    double lam = frameData.ssimWeight;
    if (l1) *l1 = s[0] / numel;
    if (ssim) *ssim = s[1] / snumel;
    if (sqErr) *sqErr = s[2] / numel;
    return (1 - lam) * s[0] / numel + lam * (1 - s[1] / snumel);
}

double GpuSplat::loss(double* l1, double* ssim) { return imageLoss(l1, ssim); }

// ------------------------------------------------------------------ backward and optimiser

void GpuSplat::rasterBackward() {
    ctx_.begin();
    ctx_.fill(acc, 0);
    ctx_.dispatch("raster_bwd", {&frame, &ranges, &tileVals, &p2d, &conicO, &rgbd, &finalT, &nContrib, &gradImg, &acc},
                  nullptr, 0, uint32_t(tilesX), uint32_t(tilesY));
    ctx_.submitWait();
}

void GpuSplat::projectBackward() {
    ctx_.begin();
    uint32_t stripes = feat.pose ? POSE_STRIPES : 0u;
    if (feat.pose) ctx_.fill(poseAcc, 0);
    dispatch1D("preprocess_bwd", {&frame, &params[0], &params[1], &params[2], &params[3], &params[4], &acc, &tilesTouched,
                                  &grads[0], &grads[1], &grads[2], &grads[3], &grads[4], &poseAcc}, &stripes, 4,
               ceilDiv(uint32_t(n), 256));
    ctx_.submitWait();
}

void GpuSplat::recordAdam(const std::array<AdamGroup, 5>& groups, float beta1, float beta2, float eps) {
    float bc1 = 1.0f - std::pow(beta1, float(adamSteps)), bc2 = 1.0f - std::pow(beta2, float(adamSteps));
    for (int k = 0; k < 5; k++) {
        if (!compactMoments) {
            struct { float lr, lr2; uint32_t period, split; float beta1, beta2, eps, bc1, bc2; uint32_t n; } pc{
                groups[k].lr, groups[k].lr2, groups[k].period, groups[k].split, beta1, beta2, eps, bc1, bc2,
                uint32_t(n * Params::width(k))};
            dispatch1D("adam", {&params[k], &grads[k], &m1[k], &m2[k]}, &pc, sizeof(pc), ceilDiv(pc.n, 256));
            continue;
        }
        uint32_t seed = compactSeed ^ (0x9E3779B9u * uint32_t(k + 1));
        if (k < 4) {
            struct { float lr, beta1, beta2, eps, bc1, bc2; uint32_t n, seed, step, pad[3]; } pc{
                groups[k].lr, beta1, beta2, eps, bc1, bc2, uint32_t(n * Params::width(k)), seed, uint32_t(adamSteps), {0, 0, 0}};
            dispatch1D("adam_bf16", {&params[k], &grads[k], &m1[k]}, &pc, sizeof(pc), ceilDiv(pc.n, 256));
        } else {
            struct { float lrDC, lrRest, beta1, beta2, eps, bc1, bc2; uint32_t n, seed, step; float bound; uint32_t pad; } pc{
                groups[k].lr, groups[k].lr2, beta1, beta2, eps, bc1, bc2, uint32_t(n), seed, uint32_t(adamSteps),
                float(compact::ratioBound(beta1, beta2)), 0};
            dispatch1D("adam_sh8", {&params[k], &grads[k], &m1[k]}, &pc, sizeof(pc), ceilDiv(uint32_t(n), 64));
        }
    }
}

void GpuSplat::adamStep(const std::array<AdamGroup, 5>& groups, float beta1, float beta2, float eps) {
    adamSteps++;
    ctx_.begin();
    recordAdam(groups, beta1, beta2, eps);
    ctx_.submitWait();
}

void GpuSplat::backwardStep(const std::array<AdamGroup, 5>& groups, const StepExtras& ex) {
    uint32_t un = uint32_t(n);
    adamSteps++;
    const float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-15f;
    ctx_.begin();
    recordImageBackward();
    ctx_.fill(acc, 0);
    ctx_.dispatch("raster_bwd", {&frame, &ranges, &tileVals, &p2d, &conicO, &rgbd, &finalT, &nContrib, &gradImg, &acc},
                  nullptr, 0, uint32_t(tilesX), uint32_t(tilesY));
    uint32_t stripes = feat.pose ? POSE_STRIPES : 0u;
    if (feat.pose) ctx_.fill(poseAcc, 0);
    dispatch1D("preprocess_bwd", {&frame, &params[0], &params[1], &params[2], &params[3], &params[4], &acc, &tilesTouched,
                                  &grads[0], &grads[1], &grads[2], &grads[3], &grads[4], &poseAcc}, &stripes, 4, ceilDiv(un, 256));
    if (ex.wOp != 0 || ex.wScale != 0 || ex.wFlat != 0 || ex.wAniso != 0) {
        struct { float wOp, wScale, wFlat, wAniso, maxRatio; uint32_t n; } pc{ex.wOp, ex.wScale, ex.wFlat, ex.wAniso, ex.maxRatio, un};
        dispatch1D("reg_grad", {&params[1], &params[3], &grads[1], &grads[3]}, &pc, sizeof(pc), ceilDiv(un, 256));
    }
    if (ex.grad2d && ex.count) {
        struct { float hw, hh; uint32_t absGrad, n; } pc{0.5f * float(width), 0.5f * float(height), ex.absGrad ? 1u : 0u, un};
        dispatch1D("grad_stats", {&acc, &tilesTouched, ex.grad2d, ex.count}, &pc, sizeof(pc), ceilDiv(un, 256));
    }
    recordAdam(groups, beta1, beta2, eps);
    if (feat.sky && ex.skyLr > 0 && feat.skyM && feat.skyV && feat.skyGrad) {
        float sb1 = 1.0f - std::pow(beta1, float(ex.skySteps)), sb2 = 1.0f - std::pow(beta2, float(ex.skySteps));
        struct { float lr, lr2; uint32_t period, split; float beta1, beta2, eps, bc1, bc2; uint32_t n; } pc{
            ex.skyLr, ex.skyLr, 1, 1, beta1, beta2, eps, sb1, sb2, feat.skyW * feat.skyH * 3};
        dispatch1D("adam", {feat.skyTex, feat.skyGrad, feat.skyM, feat.skyV}, &pc, sizeof(pc), ceilDiv(pc.n, 256));
    }
    if (ex.noiseScaler != 0) {
        struct { float scaler; uint32_t seed, step, n; } pc{ex.noiseScaler, ex.noiseSeed, ex.noiseStep, un};
        dispatch1D("mcmc_noise", {&params[0], &params[1], &params[2], &params[3]}, &pc, sizeof(pc), ceilDiv(un, 256));
    }
    ctx_.submitWait();
}

void GpuSplat::setState(const Params& p, const Params* mm1, const Params* mm2, int steps) {
    setParams(p);
    if (mm1 && mm2) {
        for (int k = 0; k < 5; k++) {
            if (!compactMoments) {
                ctx_.upload(m1[k], mm1->group(k).data(), mm1->group(k).size() * sizeof(float));
                ctx_.upload(m2[k], mm2->group(k).data(), mm2->group(k).size() * sizeof(float));
                continue;
            }
            // carriers (compact.h) → GPU words
            std::vector<uint32_t> w;
            if (k < 4) {
                w.resize(mm1->group(k).size());
                compact::encodeBf16(mm1->group(k).data(), mm2->group(k).data(), w.size(), w.data());
            } else {
                w.resize(size_t(n) * compact::SH_WORDS);
                for (int i = 0; i < n; i++)
                    compact::encodeSHRow(&mm1->sh[size_t(i) * 48], &mm2->sh[size_t(i) * 48], &w[size_t(i) * compact::SH_WORDS]);
            }
            if (!w.empty()) ctx_.upload(m1[k], w.data(), w.size() * 4);
        }
    }
    adamSteps = steps;
}

void GpuSplat::getMoments(Params& a, Params& b) const {
    a.resize(n);
    b.resize(n);
    for (int k = 0; k < 5; k++) {
        if (!compactMoments) {
            ctx_.download(m1[k], a.group(k).data(), a.group(k).size() * sizeof(float));
            ctx_.download(m2[k], b.group(k).data(), b.group(k).size() * sizeof(float));
            continue;
        }
        if (n == 0) continue;
        if (k < 4) {
            std::vector<uint32_t> w = read<uint32_t>(m1[k], a.group(k).size());
            compact::decodeBf16(w.data(), w.size(), a.group(k).data(), b.group(k).data());
        } else {
            std::vector<uint32_t> w = read<uint32_t>(m1[k], size_t(n) * compact::SH_WORDS);
            for (int i = 0; i < n; i++)
                compact::decodeSHRow(&w[size_t(i) * compact::SH_WORDS], &a.sh[size_t(i) * 48], &b.sh[size_t(i) * 48]);
        }
    }
}

std::vector<float> GpuSplat::image() const { return read<float>(img, size_t(width) * height * 3); }

double GpuSplat::trainStep(const std::array<AdamGroup, 5>& groups) {
    forward();
    double l = loss();
    rasterBackward();
    projectBackward();
    adamStep(groups);
    return l;
}

}  // namespace vkt
