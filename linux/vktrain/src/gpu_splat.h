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

// The optional parts of the loss (stage 2b), set per step by the training loop (trainer.cpp) before forwardLoss /
// backwardStep. Everything off by default: the loss is then the stage-2a L1 + SSIM.
struct LossFeatures {
    // Learned skybox (trainer.py Skybox): the backdrop is sigmoid(panorama) at the pixel's direction; raster_fwd
    // then runs over a black background (RenderOpts.bg must be 0). skyRot = B · Rᵀ (camera → sky coordinates).
    bool sky = false;
    Buffer* skyTex = nullptr;     // logits [H][W][3]
    Buffer* skyGrad = nullptr;    // same layout, accumulated (zeroed by backwardStep)
    Buffer* skyM = nullptr;       // its Adam moments (backwardStep with StepExtras.skyLr > 0)
    Buffer* skyV = nullptr;
    uint32_t skyW = 0, skyH = 0;
    float skyRot[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    bool skyMask = false;         // aux holds a sky mask (setAux): coverage penalty
    float skyPenalty = 0;         // −skyAlphaWeight / max(Σ mask, 1)
    // Per-photo colour correction (trainer.py AppearanceCorrection / BilateralGrid), parameters from setCorrection.
    enum Corr { None = 0, Affine = 1, Grid = 2 } corr = None;
    bool decoupledSSIM = false;   // ssim_split (only with a correction)
    // Transient masking (trainer.py TransientMasker): the threshold is updated every step; weights applied when active.
    bool transient = false, transientActive = false;
    float transientSensitivity = 3.0f;
    // Depth supervision / depth-normal loss on ED = D / alpha against the map from setDepthTarget.
    float depthScale = 0;         // depth weight / #valid pixels (0: off)
    float normalScale = 0;        // normalWeight / #pixels (0: off)
    bool edgeAware = true;
    int normalK = 1;              // sensor normal spacing (trainer.py: round(W / sensor width), ≥ 1)
    // Pose refinement: accumulate dL/d(world → camera) in backwardStep (readPoseGrad).
    bool pose = false;
    bool anyDepth() const { return depthScale > 0 || normalScale > 0; }
};
constexpr uint32_t CORR_STRIPES = 32, POSE_STRIPES = 32;
constexpr int BILAGRID_SIZE = 12 * 8 * 16 * 16;

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

    // ---- training-loop pieces (trainer.cpp)
    // Projection, sort, rasterisation and the loss in two submissions; returns the loss (stats read back).
    // statsOut (optional, 3 values): Σ|x−y|, Σ SSIM, Σ(clamp(x)−y)².
    double forwardLoss(double* l1 = nullptr, double* ssim = nullptr, double* sqErr = nullptr);
    struct StepExtras {
        float wOp = 0, wScale = 0, wFlat = 0, wAniso = 0, maxRatio = 10;   // reg_grad.comp weights
        Buffer* grad2d = nullptr;  // grad_stats.comp target (adaptive strategy), or null
        Buffer* count = nullptr;
        bool absGrad = false;
        float noiseScaler = 0;     // mcmc_noise.comp (0: off)
        uint32_t noiseSeed = 0, noiseStep = 0;
        float skyLr = 0;           // > 0 with feat.sky: Adam step of the panorama (its own count, skySteps ≥ 1)
        int skySteps = 0;
    };
    // Backward passes, regularisers, densification statistics, Adam and MCMC noise in one submission.
    void backwardStep(const std::array<AdamGroup, 5>& groups, const StepExtras& ex);
    // Parameters with Adam moments (m1, m2 null: zero moments) and the Adam step count.
    void setState(const Params& p, const Params* m1, const Params* m2, int steps);
    void getMoments(Params& m1, Params& m2) const;
    std::vector<float> image() const;          // the rendered rgb planes [3][h][w] of the last forward

    // ---- stage 2b loss features (see LossFeatures)
    LossFeatures feat;
    void setCorrection(const float* params, size_t count);   // 12 (affine) or BILAGRID_SIZE (grid) floats
    void setAux(const std::vector<uint32_t>& aux);           // per pixel of the current camera: sky mask in bits 0–7
    void setDepthTarget(const float* d, int w, int h, int viewW, int viewH);   // sensor / prior map, photo size
    // forward() plus the skybox and the colour correction: the image the loss (or evaluation) sees.
    uint32_t forwardComposite();
    // The image-space loss pipeline alone on the current img / finalT (tests): sky composite, correction, transient
    // masks, L1 + SSIM, depth losses → gradImg; then imageBackward() (correction and sky backward → dL/d raw image).
    double imageLoss(double* l1 = nullptr, double* ssim = nullptr);
    void imageBackward();
    // Reductions read back after backwardStep.
    std::vector<float> readCorrGrad() const;   // 12 (summed over the stripes) or BILAGRID_SIZE
    void readPoseGrad(double out[15]) const;   // dL/dW (row-major), dL/dt, dL/dcampos
    struct TransientState { float threshold = 0; bool hasThreshold = false; uint32_t outliers = 0; };
    TransientState transientState() const;
    void setTransientState(float threshold, bool has);
    double depthLossSum() const { return lastStats_[3]; }    // Σ ew·|ED − d| of the last forwardLoss
    double normalLossSum() const { return lastStats_[4]; }
    double normalValid() const { return lastStats_[5]; }

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
    // optimizerPrecision "compact" (compact.h): set before setParams / setState; moments then live in m1 as words.
    bool compactMoments = false;
    uint32_t compactSeed = 0;
    int n = 0, width = 0, height = 0, tilesX = 0, tilesY = 0;
    uint32_t tileCount = 0;
    Frame frameData{};
    int adamSteps = 0;

    Buffer frame;
    Buffer params[5], grads[5], m1[5], m2[5];
    Buffer p2d, conicO, rgbd, rects, tilesTouched, depthKeys, sortVals, offsets;
    Buffer tileKeys, tileVals, ranges;
    Buffer img, finalT, nContrib, target, planes, mu, tmp, dmaps, stats, gradImg, acc;
    // stage 2b (allocated when a feature first needs them)
    Buffer raw, gradRaw, aux, weight, resid, dED, nbuf, hist, tstate, corrBuf, corrGrad, poseAcc, depthT, dummy;

private:
    void ensureFeatureBuffers();
    void recordAdam(const std::array<AdamGroup, 5>& groups, float beta1, float beta2, float eps);
    bool allocCompact_ = false;
    void recordImageLoss();             // after raster_fwd: sky, correction, transient, SSIM, loss gradient, depth
    void recordImageBackward();         // correction backward, sky backward (before raster_bwd)
    void recordComposite();             // sky composite + correction apply
    struct SkyPC { float r0[4], r1[4], r2[4]; uint32_t W, H; float penalty; uint32_t hasMask; };
    SkyPC skyPC() const;
    uint32_t depthW_ = 0, depthH_ = 0, viewW_ = 0, viewH_ = 0;
    int corrCount_ = 0;
    double lastStats_[8] = {};
    void reserveSort(uint32_t count);
    void reserveScan(uint32_t count);
    void recordSort(Buffer& keys, Buffer& vals, uint32_t count, int bits);
    void recordScan(Buffer& data, uint32_t count, size_t level = 0);
    void recordBlur(Buffer& src, Buffer& dst, uint32_t planesCount);
    uint32_t recordForwardPhase1();      // records phase 1, submits, returns the tile instance count
    void recordForwardPhase2(uint32_t m);
    double lossFromStats(double* l1, double* ssim, double* sqErr);
    void dispatch1D(const char* kernel, const std::vector<const Buffer*>& bufs, const void* pc, uint32_t pcSize, uint32_t groups);
    Buffer sortKeysB_, sortValsB_, hist_;
    std::vector<Buffer> scanLevels_;
    float ssimWindow_[11];
};

}  // namespace vkt
