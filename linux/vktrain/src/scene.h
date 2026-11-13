// Splat parameters, cameras and per-view constants shared by the GPU trainer, the CPU reference and the tools.
#pragma once
#include <cmath>
#include <cstdint>
#include <vector>

namespace vkt {

// Per-splat parameters in the trainer's (Mac GaussianCloud) layout.
struct Params {
    int n = 0;
    std::vector<float> pos;       // 3 per splat
    std::vector<float> logScale;  // 3
    std::vector<float> quat;      // 4, (w, x, y, z), not necessarily normalised
    std::vector<float> opLogit;   // 1
    std::vector<float> sh;        // 48: 16 coefficients × rgb, sh[i*48 + k*3 + c]
    void resize(int count) {
        n = count;
        pos.assign(size_t(3) * n, 0.f);
        logScale.assign(size_t(3) * n, 0.f);
        quat.assign(size_t(4) * n, 0.f);
        opLogit.assign(size_t(n), 0.f);
        sh.assign(size_t(48) * n, 0.f);
    }
    std::vector<float>& group(int k) { return k == 0 ? pos : k == 1 ? logScale : k == 2 ? quat : k == 3 ? opLogit : sh; }
    const std::vector<float>& group(int k) const { return const_cast<Params*>(this)->group(k); }
    static int width(int k) { static const int w[5] = {3, 3, 4, 1, 48}; return w[k]; }
};

// Pinhole camera, OpenCV convention (x right, y down, z forward), like the Linux dataset's cameraToWorld⁻¹.
struct Camera {
    double w2c[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};   // row-major
    double fx = 1, fy = 1, cx = 0, cy = 0;
    int width = 1, height = 1;
    void center(double out[3]) const {   // camera position: −Rᵀ t
        for (int k = 0; k < 3; k++) out[k] = -(w2c[0][k] * w2c[0][3] + w2c[1][k] * w2c[1][3] + w2c[2][k] * w2c[2][3]);
    }
};

struct RenderOpts {
    int shDegree = 3;
    bool antialiased = false;   // Mip-Splatting opacity compensation (gsplat rasterize_mode="antialiased")
    float bg[3] = {0, 0, 0};
    float dilation = 0.3f;      // added to the 2D covariance diagonal (gsplat eps2d)
    float ssimWeight = 0.2f;
    bool absGrad = true;
    bool hasDepth = true;       // accumulate the depth-plane gradient in the backward pass
};

// Mirrors `Frame` in shaders/common.glsl (std430).
struct Frame {
    float w2c[16];   // column-major
    float intr[4];
    float lim[4];
    float campos[4];
    float bg[4];
    uint32_t width, height, tilesX, tilesY;
    uint32_t count, shDegree, flags, absGrad;
    uint32_t hasDepth, pad0, pad1, pad2;
    float ssimWeight, invNumPixels, depthWeight, pad3;
};
static_assert(sizeof(Frame) == 192, "Frame layout");

constexpr int TILE = 16;

inline double tanLimitX(const Camera& c) { return 1.3 * (0.5 * c.width / c.fx); }
inline double tanLimitY(const Camera& c) { return 1.3 * (0.5 * c.height / c.fy); }

inline Frame makeFrame(const Camera& c, const RenderOpts& o, int n) {
    Frame f{};
    for (int r = 0; r < 4; r++)
        for (int k = 0; k < 4; k++) f.w2c[k * 4 + r] = float(c.w2c[r][k]);
    f.intr[0] = float(c.fx); f.intr[1] = float(c.fy); f.intr[2] = float(c.cx); f.intr[3] = float(c.cy);
    f.lim[0] = float(tanLimitX(c)); f.lim[1] = float(tanLimitY(c)); f.lim[2] = o.dilation;
    double cp[3];
    c.center(cp);
    for (int k = 0; k < 3; k++) { f.campos[k] = float(cp[k]); f.bg[k] = o.bg[k]; }
    f.width = uint32_t(c.width); f.height = uint32_t(c.height);
    f.tilesX = uint32_t((c.width + TILE - 1) / TILE); f.tilesY = uint32_t((c.height + TILE - 1) / TILE);
    f.count = uint32_t(n);
    f.shDegree = uint32_t(o.shDegree);
    f.flags = o.antialiased ? 1u : 0u;
    f.absGrad = o.absGrad ? 1u : 0u;
    f.hasDepth = o.hasDepth ? 1u : 0u;
    f.ssimWeight = o.ssimWeight;
    f.invNumPixels = 1.0f / float(c.width * c.height);
    return f;
}

// OpenCV camera at `eye` looking at `target`, rolled by `roll` radians about the view axis.
inline Camera lookAt(const double eye[3], const double target[3], double roll, double fx, double fy, double cx, double cy, int w, int h) {
    double f[3] = {target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]};
    double fl = std::sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
    for (double& v : f) v /= fl;
    double up[3] = {0, -1, 0};
    double r[3] = {f[1] * up[2] - f[2] * up[1], f[2] * up[0] - f[0] * up[2], f[0] * up[1] - f[1] * up[0]};
    double rl = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    if (rl < 1e-9) { r[0] = 1; r[1] = 0; r[2] = 0; rl = 1; }
    for (double& v : r) v /= rl;
    double d[3] = {f[1] * r[2] - f[2] * r[1], f[2] * r[0] - f[0] * r[2], f[0] * r[1] - f[1] * r[0]};   // down = f × r
    double cr = std::cos(roll), sr = std::sin(roll);
    double x[3], y[3];
    for (int k = 0; k < 3; k++) { x[k] = cr * r[k] + sr * d[k]; y[k] = -sr * r[k] + cr * d[k]; }
    Camera c;
    for (int k = 0; k < 3; k++) { c.w2c[0][k] = x[k]; c.w2c[1][k] = y[k]; c.w2c[2][k] = f[k]; }
    for (int i = 0; i < 3; i++) c.w2c[i][3] = -(c.w2c[i][0] * eye[0] + c.w2c[i][1] * eye[1] + c.w2c[i][2] * eye[2]);
    c.w2c[3][0] = c.w2c[3][1] = c.w2c[3][2] = 0; c.w2c[3][3] = 1;
    c.fx = fx; c.fy = fy; c.cx = cx; c.cy = cy; c.width = w; c.height = h;
    return c;
}

}  // namespace vkt
