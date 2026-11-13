// TrainingConfig (docs/PROTOCOL.md §3) for the Vulkan trainer: every key the protocol defines, with the same
// defaults, coercion and validation as the CUDA worker's linux/splatworker/config.py, so one JSON config means
// the same thing to both trainers. Keys this trainer does not implement yet are accepted and reported once
// each by unsupportedNotes().
#pragma once
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace vkt {

struct ConfigError : std::runtime_error { using std::runtime_error::runtime_error; };

struct Region {
    double min[3] = {0, 0, 0}, max[3] = {0, 0, 0};
    double margin = 0;
    // The margin-expanded box (each side pushed out by margin × size along its axis), config.py Region.expanded.
    void expanded(double lo[3], double hi[3]) const;
};

struct TrainConfig {
    int64_t iterations = 30000;
    int64_t maxImageSize = 1600;
    int64_t shDegree = 3;
    int64_t shDegreeInterval = 1000;
    std::string strategy = "mcmc";
    int64_t maxGaussians = 1500000;
    int64_t densifyStart = 500;
    int64_t densifyStop = 15000;
    int64_t densifyInterval = 100;
    double densifyGradThreshold = 0.0002;
    bool absGrad = false;
    int64_t opacityResetInterval = 3000;
    double pruneOpacity = 0.005;
    double lrPosition = 0.00016;
    double lrPositionFinal = 0.0000016;
    double lrColor = 0.0025;
    double lrSHRest = 0.000125;
    double lrOpacity = 0.05;
    double lrScale = 0.005;
    double lrRotation = 0.001;
    double ssimWeight = 0.2;
    double opacityReg = 0.01;
    double scaleReg = 0.01;
    double noiseLR = 500000.0;
    bool antialiased = true;
    double depthWeight = 0.0;
    std::string background = "random";
    double initialOpacity = 0.1;
    int64_t randomInitCount = 100000;
    int64_t seed = 42;
    int64_t previewInterval = 500;
    int64_t evalInterval = 2000;
    bool transientMasking = false;
    double transientSensitivity = 3.0;
    int64_t transientStart = 1000;
    double transientMaxViewGap = 15.0;
    bool depthEdgeAware = true;
    double anisotropyReg = 0.1;
    double anisotropyMaxRatio = 10.0;
    double normalWeight = 0.05;
    int64_t normalStart = 1000;
    double flatnessReg = 0.0;
    std::string depthPriors = "auto";
    double monoDepthWeight = 0.0;
    int64_t priorInitPoints = 200000;
    bool longAxisSplit = true;
    bool poseRefinement = false;
    double poseRefinementLR = 1e-4;
    double poseRefinementReg = 1e-6;
    int64_t poseRefinementStart = 500;
    bool appearanceCorrection = true;
    double appearanceLR = 0.02;
    double appearanceReg = 0.001;
    bool bilateralGrid = false;
    double bilateralGridLR = 2e-3;
    double bilateralGridTV = 10.0;
    bool progressiveResolution = false;
    bool decoupledSSIM = false;
    int64_t backgroundShell = 0;
    std::string optimizerPrecision = "fp32";
    std::string skybox = "auto";
    int64_t skyboxResolution = 256;
    double skyAlphaWeight = 0.05;
    bool hasRegion = false;
    Region region;
    bool hasCameraIds = false;
    std::vector<int64_t> cameraIds;

    std::set<std::string> explicitKeys;   // keys the caller supplied (context-dependent defaults)

    static TrainConfig fromJson(const nlohmann::json& j);   // throws ConfigError
    void validate() const;
    nlohmann::json toJson() const;
    bool isSet(const std::string& k) const { return explicitKeys.count(k) != 0; }

    int64_t effectiveDensifyStop() const;
    double effectiveGradThreshold() const;
    double effectiveOpacityReg() const;
    double effectiveScaleReg() const;
    double effectiveDepthWeight(bool hasDepth) const;
};

// What the dataset offers that decides whether an unsupported feature would be active.
struct FeatureContext {
    bool hasDepth = false;          // any training photo has sensor depth
    bool hasSky = false;            // any training photo has a sky mask
    bool exposureLocked = false;    // all photos within 10 % exposure (appearance correction off in trainer.py)
};
// One line per enabled feature the Vulkan trainer does not implement yet (empty since stage 2b implemented them all;
// kept for features added to the protocol later).
std::vector<std::string> unsupportedNotes(const TrainConfig& c, const FeatureContext& f);

}  // namespace vkt
