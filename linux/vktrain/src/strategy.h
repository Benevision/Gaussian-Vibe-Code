// Host-side splat operations of the training loop: initialisation (trainer.py init_points / background_shell,
// kNN scale init) and the two densification strategies, ported by algorithm from gsplat 1.5 (Apache-2.0):
//   MCMCStrategy  — relocate dead splats onto live ones, add 5 % per refinement up to the cap, position noise
//                   (gsplat/strategy/mcmc.py, ops.py relocate / sample_add / inject_noise_to_position, and the
//                   relocation formula of cuda/csrc/RelocationCUDA.cu, eq. 9 of the 3DGS-MCMC paper);
//   DefaultStrategy — clone small / split large splats with high screen-space gradient, prune transparent and
//                   (after the first reset interval) huge ones (gsplat/strategy/default.py, ops.py).
// plus trainer.py's spread_clone_groups (long-axis split of MCMC clone groups).
//
// Densification runs on the CPU on a read-back copy of the parameters and Adam moments (the Mac does it on
// unified memory). That costs one full download + upload per refinement (every densifyInterval steps inside
// the densification window) — see docs/dev/GPU_SUPPORT.md for the numbers.
#pragma once
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "scene.h"

namespace vkt {

// Parameters and the two Adam moment sets, in the same row layout (Params).
struct SplatState {
    Params p, m1, m2;
    int n() const { return p.n; }
};

// Random numbers of the host side of the loop (view order, backgrounds, initialisation, strategies).
struct HostRng {
    std::mt19937_64 g;
    explicit HostRng(uint64_t seed = 42) : g(seed) {}
    double uniform() { return std::uniform_real_distribution<double>(0.0, 1.0)(g); }
    double normal() { return std::normal_distribution<double>(0.0, 1.0)(g); }
    std::string save() const;
    void load(const std::string& s);
};

// The counter-based normal of shaders/common.glsl hash_normal (same function of seed, step, index, k).
float hashNormal(uint32_t seed, uint32_t step, uint32_t idx, uint32_t k);

// Mean distance to the 3 nearest neighbours of each point (kd-tree; trainer.py _knn_mean_dist).
std::vector<float> knnMeanDist(const std::vector<float>& xyz, int k = 3);

// Fibonacci-lattice points on a sphere of radius r around centre (trainer.py background_shell).
std::vector<float> backgroundShell(const double centre[3], double r, int64_t n);

// New splats at xyz with colours rgb (trainer.py: log kNN scales, random quaternions in [0,1)^4, logit of
// initialOpacity, SH DC from the colour, higher bands zero). Moments zero.
SplatState initSplats(const std::vector<float>& xyz, const std::vector<float>& rgb, float initialOpacity, HostRng& rng);

struct McmcParams {
    int64_t capMax = 1500000;
    double minOpacity = 0.005;
};
struct McmcResult { size_t relocated = 0, added = 0; };
// gsplat MCMCStrategy refinement (relocate, then add) on the host copy.
McmcResult mcmcRefine(SplatState& s, const McmcParams& prm, HostRng& rng);
// gsplat inject_noise_to_position, host version of shaders/mcmc_noise.comp.
void mcmcNoise(SplatState& s, float scaler, uint32_t seed, uint32_t step);
// trainer.py spread_clone_groups: returns the number of splats moved.
size_t spreadCloneGroups(SplatState& s);

struct AdcParams {
    double growGrad2d = 0.0002;    // +inf at the splat cap
    double growScale3d = 0.01;
    double pruneScale3d = 0.1;
    double pruneOpa = 0.005;
    double sceneScale = 1.0;
    int64_t resetEvery = 3000;
};
struct AdcResult { size_t duplicated = 0, split = 0, pruned = 0; };
// gsplat DefaultStrategy _grow_gs + _prune_gs at `step` with the accumulated statistics.
AdcResult adcRefine(SplatState& s, const std::vector<float>& grad2d, const std::vector<float>& count, const AdcParams& prm,
                    int64_t step, HostRng& rng);
// Textbook opacity reset (clamp to 2 × pruneOpa, zero the opacity moments). See trainer.cpp for why it is off.
void resetOpacity(SplatState& s, double value);

}  // namespace vkt
