// GPU side of one training step: projection, depth sort, tile binning, rasterisation, L1 + SSIM loss,
// backward passes and Adam, encoded like MetalTrainer.swift (encodeForward / encodeRaster / encodeLoss /
// forwardBackward), on the kernels in shaders/.
#pragma once
#include <array>

#include "scene.h"
#include "vk.h"

namespace vkt {

struct AdamGroup {
    float lr = 1e-3f, lr2 = 1e-3f;   // element i uses lr when (i % period) < split, else lr2
    uint32_t period = 1, split = 1;
};

class GpuSplat {
public:
    explicit GpuSplat(Context& ctx);
    ~GpuSplat();

    void setParams(const Params& p);          // (re)allocates per-splat buffers when n changes; resets Adam
    void getParams(Params& p) const;
    void getGrads(Params& g) const;           // gradients in the Params layout
    void setCamera(const Camera& cam, const RenderOpts& opt);
    void setTarget(const std::vector<uint32_t>& rgba);   // width × height RGBA8 (r | g<<8 | b<<16 | a<<24)

    uint32_t forward();                       // returns the number of tile instances (0: nothing visible)
    double loss(double* l1 = nullptr, double* ssim = nullptr);   // needs forward(); also writes gradImg
    void rasterBackward();                    // gradImg → acc (zeroed first)
    void projectBackward();                   // acc → grads
    void adamStep(const std::array<AdamGroup, 5>& groups, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-15f);
    // One full step on the current camera/target: forward, loss, backward, Adam. Returns the loss.
    double trainStep(const std::array<AdamGroup, 5>& groups);

    // Standalone primitives (own submission), used by the tests.
    void sortPairs(Buffer& keys, Buffer& vals, uint32_t count, int bits);
    void exclusiveScan(Buffer& data, uint32_t count);
    // Float-atomic probe (shaders/atomic_probe.comp) with `count` invocations: returns the number of wrong
    // sums (0 = the selected atomic path adds correctly, per subgroup and contended per lane).
    int atomicProbe(uint32_t count);

    template <class T> std::vector<T> read(const Buffer& b, size_t count) const {
        std::vector<T> v(count);
        ctx_.download(b, v.data(), count * sizeof(T));
        return v;
    }

    Context& ctx_;
    int n = 0, width = 0, height = 0, tilesX = 0, tilesY = 0;
    uint32_t tileCount = 0;
    Frame frameData{};
    int adamSteps = 0;

    Buffer frame;
    Buffer params[5], grads[5], m1[5], m2[5];
    Buffer p2d, conicO, rgbd, rects, tilesTouched, depthKeys, sortVals, offsets;
    Buffer tileKeys, tileVals, ranges;
    Buffer img, finalT, nContrib, target, planes, mu, tmp, dmaps, stats, gradImg, acc;

private:
    void reserveSort(uint32_t count);
    void reserveScan(uint32_t count);
    void recordSort(Buffer& keys, Buffer& vals, uint32_t count, int bits);
    void recordScan(Buffer& data, uint32_t count, size_t level = 0);
    void recordBlur(Buffer& src, Buffer& dst, uint32_t planesCount);
    void dispatch1D(const char* kernel, const std::vector<const Buffer*>& bufs, const void* pc, uint32_t pcSize, uint32_t groups);
    Buffer sortKeysB_, sortValsB_, hist_;
    std::vector<Buffer> scanLevels_;
    float ssimWindow_[11];
};

}  // namespace vkt
