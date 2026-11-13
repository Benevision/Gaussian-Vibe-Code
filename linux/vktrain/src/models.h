// Host side of the stage-2b training features, ported by behaviour from linux/splatworker/trainer.py and skybox.py:
// the small per-photo models whose optimisers run on the CPU (colour correction, bilateral grids, camera pose
// corrections — a few dozen to a few ten thousand numbers per photo, updated only when their photo is drawn), the
// skybox's orientation and its sky.png / sky.json sidecars, sensor / prior depth maps, and helpers.
// The per-pixel work of all of them runs on the GPU (gpu_splat.*, shaders/).
#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "dataset.h"

namespace vkt {

// ---------------------------------------------------------------- optimisers

// torch.optim.Adam (no weight decay) for one small parameter vector: m, v and the step count of its own.
struct HostAdam {
    std::vector<double> m, v;
    int64_t steps = 0;
    void step(std::vector<float>& p, const std::vector<double>& g, double lr, double eps, double b1 = 0.9, double b2 = 0.999);
};

// ---------------------------------------------------------------- per-photo colour correction

// trainer.py AppearanceCorrection: corrected = M · rendered + b per photo (12 numbers), seeded from the recorded
// exposure (gain (E / E_median)^(1/2.2) clamped to [1/3, 3]), Adam (eps 1e-8) per photo with an L2 pull towards
// the seed, only stepped when the photo is drawn. Covers the training photos and then the holdout photos.
struct AppearanceModel {
    std::vector<std::vector<float>> p, init;
    std::vector<HostAdam> opt;
    double lr = 0.02, reg = 0.001;
    void setup(const std::vector<const CameraInfo*>& cams, double lr, double reg);
    // grad: dL/dp of photo i from the GPU (12). Skipped when not finite (trainer.py).
    void step(size_t i, const std::vector<float>& grad);
};

// trainer.py BilateralGrid: per training photo a 12 × 8 × 16 × 16 grid of 3×4 colour transforms (identity at the
// start), total-variation smoothed, Adam (eps 1e-15) per photo, and every 50 steps the mean over photos is pinned
// back to identity (pinMean).
struct BilateralGridModel {
    std::vector<std::vector<float>> grids;
    std::vector<HostAdam> opt;
    double lr = 2e-3, tv = 10.0;
    void setup(size_t views, double lr, double tvWeight);
    static std::vector<float> identity();
    // Adds the gradient of tv · Σ_dims mean((g[1:] − g[:−1])²) of photo i's grid to grad.
    void addTvGrad(size_t i, std::vector<double>& grad) const;
    double tvLoss(size_t i) const;
    void step(size_t i, const std::vector<float>& gpuGrad);
    void pinMean();
};

// ---------------------------------------------------------------- camera pose refinement

// trainer.py PoseRefiner: per training photo 9 numbers d (translation, 6D rotation residual), zero = no change; the
// corrected world→camera is inv(c2w · ΔT(d)) = ΔT(d)⁻¹ · w2c, ΔT = [R6(d[3:] + (1,0,0,0,1,0)) | d[:3]] with the
// Gram-Schmidt 6D rotation (rows b1, b2, b1 × b2).
void rot6d(const double a[6], double R[3][3]);
void refinedW2c(const double w2c[4][4], const double d[9], double out[4][4]);
// dL/dd from the GPU's dL/dW (3×3 row-major), dL/dt and dL/dcampos of the corrected camera (15 numbers), by
// central differences of the (smooth, 9 → 15) parametrisation in double precision.
void poseChain(const double w2c[4][4], const double d[9], const double g15[15], double out[9]);

// The embedding with torch.optim.SparseAdam (lr = poseRefinementLR, eps 1e-8): one step count for all rows, the
// moments of the drawn row only; L2 pull reg · |d|².
struct PoseRefinerModel {
    std::vector<double> e, m, v;   // rows of 9
    int64_t steps = 0;
    double lr = 1e-4, reg = 1e-6;
    void setup(size_t views, double lr, double reg);
    void step(size_t row, const double grad[9]);   // grad without the L2 term (added here)
    const double* row(size_t i) const { return &e[i * 9]; }
};

// ---------------------------------------------------------------- skybox

// skybox.py basis(): up, forward (⟂ up, nearest world +Z, else +X), right = up × forward.
struct SkyBasis { double up[3], forward[3], right[3]; };
SkyBasis skyBasis(const double up[3]);
// M = B · Rᵀ (B rows: right, up, forward; R the world→camera rotation): camera direction → sky coordinates.
void skyRotation(const SkyBasis& b, const double w2c[4][4], float out[9]);
// trainer.py Skybox.save: <out>.sky.png (sigmoid of the logits, 8-bit, 2:1) and <out>.sky.json next to the PLY
// (Path.with_suffix). logits: [H][W][3]. Returns the PNG path.
std::string writeSkySidecars(const std::string& plyPath, const std::vector<float>& logits, int W, int H, const SkyBasis& b);
void writePng(const std::string& path, const std::vector<uint8_t>& rgb, int w, int h);

// ---------------------------------------------------------------- depth maps

// dataset.py load_depth: float16 little-endian, row-major, metres; non-finite and negative values → 0.
std::vector<float> loadDepthF16(const std::string& path, int w, int h);
// depthprior.py cache: DepthPriors(ds.root / "depth-prior") — meta.json (model, version) and <id>-<sha1[:12]>.npy.
bool priorCacheValid(const std::string& dir);
std::string priorCacheKey(const CameraInfo& cam);
bool readNpyFloat2D(const std::string& path, std::vector<float>& out, int& w, int& h);
std::string sha1Hex(const std::string& data);

// ---------------------------------------------------------------- misc

// trainer.py median_nearest_view_gap (degrees): ~5 dense capture, > 15 sparse.
double medianNearestViewGap(const std::vector<const CameraInfo*>& cams, double sceneExtent);

}  // namespace vkt
