// Double-precision CPU reference of the forward pass (projection, tile rasterisation, L1 + SSIM loss), used by
// the self-test as ground truth for the GPU forward kernels and, through central finite differences, for the
// backward kernels. Written from the same maths as the GPU kernels, independently of their code.
#pragma once
#include <cstdint>
#include <vector>

#include "scene.h"

namespace vkt::ref {

struct ParamsD {
    int n = 0;
    std::vector<double> g[5];   // pos 3, logScale 3, quat 4, opLogit 1, sh 48 per splat
};
ParamsD toDouble(const Params& p);

struct Cam {
    double W[3][3], t0[3];
    double fx, fy, cx, cy;
    int width, height, tilesX, tilesY;
    double limx, limy, dil;
    bool aa;
    int shDeg;
    double bg[3];
    double campos[3];
    double ssimWeight;
};
Cam makeCam(const Camera& c, const RenderOpts& o);

struct Proj {
    bool valid = false;
    double mean[2] = {0, 0}, conic[3] = {0, 0, 0}, op = 0, rgb[3] = {0, 0, 0}, depth = 0;
    int rect[4] = {0, 0, 0, 0};
    uint32_t tiles = 0;
    bool clamped = false;   // x/z or y/z outside the EWA Jacobian's clamp
    double q(int k) const;  // the 10 screen-space quantities in raster_bwd's order: mean xy, conic abc, op, rgb, depth
    double& q(int k);
};

// Projects splat i. decide = true applies the culling rules (valid, rect, tiles); false only evaluates the
// continuous outputs (for finite differences with frozen visibility).
void project(const ParamsD& p, const Cam& cam, int i, Proj& out, bool decide);
std::vector<Proj> projectAll(const ParamsD& p, const Cam& cam, bool decide);

// Splat indices in rasterisation order: valid splats sorted by float(depth), stable by index (like the GPU).
std::vector<int> depthOrder(const std::vector<Proj>& pr);

// Per-pixel list of contributing splats, front to back (frozen rasterisation structure).
struct Structure { std::vector<std::vector<int>> lists; };

// Tile rasterisation with the GPU's tests and early stop. img: 4 planes [r, g, b, depth].
void rasterDecide(const std::vector<Proj>& pr, const Cam& cam, std::vector<double>& img, std::vector<double>& finalT,
                  std::vector<uint32_t>& nContrib, Structure* st);
// Same blending over a frozen structure (smooth in the projected quantities).
void rasterFrozen(const std::vector<Proj>& pr, const Structure& st, const Cam& cam, std::vector<double>& img);

// L = (1−λ)·mean|x−y| + λ·(1 − mean SSIM) over 3 channels; img has ≥ 3 planes, target 3 planes in [0, 1].
double loss(const std::vector<double>& img, const std::vector<double>& target, int w, int h, double lam,
            double* l1 = nullptr, double* ssim = nullptr);

}  // namespace vkt::ref
