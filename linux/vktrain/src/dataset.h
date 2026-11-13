// The prepared training dataset (docs/PROTOCOL.md §2): dataset.json parsing, image / mask loading with the
// CUDA worker's resizing rules (linux/splatworker/dataset.py), camera selection for block training, and the
// image operations the training loop needs (area downscaling, packing for the GPU).
#pragma once
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "scene.h"

namespace vkt {

struct DatasetError : std::runtime_error { using std::runtime_error::runtime_error; };

struct CameraInfo {
    int64_t id = 0;
    std::string image;
    int width = 0, height = 0;
    double fx = 0, fy = 0, cx = 0, cy = 0;
    double c2w[4][4] = {};   // row-major, OpenCV convention
    std::string depth, mask, sky;
    int depthWidth = 0, depthHeight = 0;   // the depth map's size (float16 values)
    bool holdout = false;
    double exposure = 0;     // 0 = not recorded
};

struct DatasetInfo {
    std::string root;
    std::vector<CameraInfo> cameras;
    double center[3] = {0, 0, 0};
    double extent = 1;
    double up[3] = {0, -1, 0};
    bool hasGravityUp = false;
    double gravityUp[3] = {0, 0, 0};
    std::string points;      // relative path, empty = none
    std::string resolve(const std::string& rel) const;   // throws on paths that escape the root
};

DatasetInfo parseDataset(const std::string& dir);

// Longest side ≤ maxSize, never upscaled (dataset.py target_size).
void targetSize(int w, int h, int maxSize, int& nw, int& nh);

// A training / evaluation view at its loaded resolution.
struct View {
    const CameraInfo* cam = nullptr;
    int width = 0, height = 0;
    double fx = 0, fy = 0, cx = 0, cy = 0;
    double w2c[4][4] = {};
    std::vector<uint8_t> rgb;     // width × height × 3
    std::vector<uint8_t> alpha;   // the image's own alpha (empty: opaque)
    std::vector<uint8_t> mask;    // 255 = train on the pixel (empty: no mask)
    std::vector<uint32_t> packed; // opaque photos after compact(): r | g<<8 | b<<16 | mask<<24 (rgb, mask freed)
    std::vector<uint8_t> sky;     // sky mask, 255 = sky (empty: none), resized bilinearly like the loss mask
    Camera camera() const;
    void compact();               // opaque photos keep one packed copy (4 bytes / pixel) instead of rgb + mask
    // Pixel p: colour in 0..1, the photo's alpha and the loss mask (1 = train on it).
    void pixel(size_t p, float c[3], float& alpha, float& m) const;
};

View loadView(const DatasetInfo& ds, const CameraInfo& cam, int maxSize);

// dataset.py camera_sees_box / select_cameras.
bool cameraSeesBox(const CameraInfo& cam, const double lo[3], const double hi[3], int samples = 6, double nearZ = 1e-3);
std::vector<const CameraInfo*> selectCameras(const DatasetInfo& ds, const std::vector<int64_t>* cameraIds,
                                             const double* lo, const double* hi);

// Every photo has a recorded exposure and they're all within 10 % of each other (trainer.py exposure_locked).
bool exposureLocked(const std::vector<const CameraInfo*>& cams);

// ---- image operations

// PIL-style separable resampling of an 8-bit image with `ch` channels (antialiased when shrinking).
enum class Filter { Lanczos, Bilinear };
std::vector<uint8_t> resample(const std::vector<uint8_t>& src, int w, int h, int ch, int nw, int nh, Filter f);

// torch F.interpolate(mode="area") (adaptive average pooling) of float planes [ch][h][w].
std::vector<float> areaResize(const std::vector<float>& src, int w, int h, int ch, int nw, int nh);

// The GPU target for one step: RGBA8 packed as r | g<<8 | b<<16 | m<<24 with m the loss mask (255 = train).
// Photos with alpha are composited over bg (trainer.py: gt = img·α + (1 − α)·bg).
std::vector<uint32_t> packTarget(const View& v, const float bg[3]);
// Same at a progressive-resolution scale (trainer.py downscale_view: area-averaged image, alpha and mask;
// K scaled about pixel centres). Returns the scaled camera through `cam`.
std::vector<uint32_t> packTargetScaled(const View& v, const float bg[3], double scale, Camera& cam);

}  // namespace vkt
