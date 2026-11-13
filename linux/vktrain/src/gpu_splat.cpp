#include "gpu_splat.h"

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
}

GpuSplat::~GpuSplat() {
    Buffer* all[] = {&frame, &p2d, &conicO, &rgbd, &rects, &tilesTouched, &depthKeys, &sortVals, &offsets, &tileKeys,
                     &tileVals, &ranges, &img, &finalT, &nContrib, &target, &planes, &mu, &tmp, &dmaps, &stats,
                     &gradImg, &acc, &sortKeysB_, &sortValsB_, &hist_};
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
    if (p.n != n || !params[0].buf) {
        n = p.n;
        for (int k = 0; k < 5; k++) {
            size_t bytes = size_t(n) * Params::width(k) * sizeof(float);
            for (Buffer* b : {&params[k], &grads[k], &m1[k], &m2[k]}) { ctx_.destroy(*b); *b = ctx_.createBuffer(bytes); }
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
        for (Buffer* b : {&img, &finalT, &nContrib, &target, &planes, &mu, &tmp, &dmaps, &gradImg, &ranges}) ctx_.destroy(*b);
        img = ctx_.createBuffer(hw * 4 * 4);
        finalT = ctx_.createBuffer(hw * 4);
        nContrib = ctx_.createBuffer(hw * 4);
        target = ctx_.createBuffer(hw * 4);
        planes = ctx_.createBuffer(hw * 15 * 4);
        mu = ctx_.createBuffer(hw * 15 * 4);
        tmp = ctx_.createBuffer(hw * 15 * 4);
        dmaps = ctx_.createBuffer(hw * 9 * 4);
        gradImg = ctx_.createBuffer(hw * 4 * 4);
        ranges = ctx_.createBuffer(size_t(tilesX) * tilesY * 2 * 4);
    }
    ctx_.upload(frame, &frameData, sizeof(Frame));
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

uint32_t GpuSplat::forward() {
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

    // Phase 2: tile instances, stable sort by tile, per-tile ranges, rasterisation.
    ctx_.ensure(tileKeys, size_t(std::max<uint32_t>(m, 1)) * 4);
    ctx_.ensure(tileVals, size_t(std::max<uint32_t>(m, 1)) * 4);
    reserveSort(m);
    uint32_t numTiles = uint32_t(tilesX * tilesY);
    int bits = 1;
    while ((1u << bits) < numTiles && bits < 32) bits++;
    ctx_.begin();
    if (m > 0) {
        dispatch1D("duplicate_keys", {&frame, &sortVals, &offsets, &rects, &tileKeys, &tileVals}, &un, 4, ceilDiv(un, 256));
        recordSort(tileKeys, tileVals, m, bits);
    }
    ctx_.fill(ranges, 0);
    if (m > 0) dispatch1D("tile_ranges", {&tileKeys, &ranges}, &m, 4, ceilDiv(m, 256));
    ctx_.dispatch("raster_fwd", {&frame, &ranges, &tileVals, &p2d, &conicO, &rgbd, &img, &finalT, &nContrib}, nullptr, 0,
                  uint32_t(tilesX), uint32_t(tilesY));
    ctx_.submitWait();
    return m;
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

double GpuSplat::loss(double* l1, double* ssim) {
    uint32_t gx = ceilDiv(uint32_t(width), 16), gy = ceilDiv(uint32_t(height), 16);
    ctx_.begin();
    ctx_.dispatch("ssim_prep", {&frame, &img, &target, &planes}, nullptr, 0, gx, gy);
    recordBlur(planes, mu, 15);
    ctx_.fill(stats, 0);
    ctx_.dispatch("ssim_terms", {&frame, &mu, &dmaps, &stats, &planes}, nullptr, 0, gx, gy);
    recordBlur(dmaps, mu, 9);   // blur adjoint of the SSIM derivative maps (mu reused)
    ctx_.dispatch("loss_grad", {&frame, &planes, &mu, &gradImg}, nullptr, 0, gx, gy);
    ctx_.submitWait();
    float s[4];
    ctx_.download(stats, s, sizeof(s));
    double numel = double(width) * height * 3;
    double lam = frameData.ssimWeight;
    if (l1) *l1 = s[0] / numel;
    if (ssim) *ssim = s[1] / numel;
    return (1 - lam) * s[0] / numel + lam * (1 - s[1] / numel);
}

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
    dispatch1D("preprocess_bwd", {&frame, &params[0], &params[1], &params[2], &params[3], &params[4], &acc, &tilesTouched,
                                  &grads[0], &grads[1], &grads[2], &grads[3], &grads[4]}, nullptr, 0, ceilDiv(uint32_t(n), 256));
    ctx_.submitWait();
}

void GpuSplat::adamStep(const std::array<AdamGroup, 5>& groups, float beta1, float beta2, float eps) {
    adamSteps++;
    float bc1 = 1.0f - std::pow(beta1, float(adamSteps)), bc2 = 1.0f - std::pow(beta2, float(adamSteps));
    ctx_.begin();
    for (int k = 0; k < 5; k++) {
        struct { float lr, lr2; uint32_t period, split; float beta1, beta2, eps, bc1, bc2; uint32_t n; } pc{
            groups[k].lr, groups[k].lr2, groups[k].period, groups[k].split, beta1, beta2, eps, bc1, bc2,
            uint32_t(n * Params::width(k))};
        dispatch1D("adam", {&params[k], &grads[k], &m1[k], &m2[k]}, &pc, sizeof(pc), ceilDiv(pc.n, 256));
    }
    ctx_.submitWait();
}

double GpuSplat::trainStep(const std::array<AdamGroup, 5>& groups) {
    forward();
    double l = loss();
    rasterBackward();
    projectBackward();
    adamStep(groups);
    return l;
}

}  // namespace vkt
