// Shared definitions for the SplatWork Vulkan trainer kernels (GLSL 4.6, compiled to SPIR-V at build time).
// Conventions follow the Mac trainer (mac/Sources/SplatStudio/Shaders/TrainShaders.swift), ported by design:
// OpenCV camera (x right, y down, z forward), quaternions (w, x, y, z), SH in the 3DGS basis order with
// 16 coefficients × 3 colours per splat (sh[i*48 + k*3 + c]), images as planes [plane][H][W].
#ifndef SPLAT_COMMON_GLSL
#define SPLAT_COMMON_GLSL

#define TILE 16

// Per-view constants (one std430 storage buffer, binding 0 in every kernel that needs it). Mirrors vkt::Frame.
struct Frame {
    mat4 w2c;          // world → camera, column-major (w2c[col][row])
    vec4 intr;         // fx, fy, cx, cy (pixels)
    vec4 lim;          // x, y: clamp of x/z, y/z for the EWA Jacobian (1.3·tan half-fov); z: 2D dilation (0.3 px)
    vec4 campos;       // xyz camera centre in world space
    vec4 bg;           // rgb background
    uint width; uint height; uint tilesX; uint tilesY;
    uint count; uint shDegree; uint flags; uint absGrad;   // flags bit0: Mip-Splatting opacity compensation,
                                                           // bit1: SSIM over the valid window positions only
    uint hasDepth; uint pad0; uint pad1; uint pad2;
    float ssimWeight; float invNumPixels; float depthWeight; float invSsimPixels;   // 1 / pixels the SSIM mean is over
};

const float SH_C0 = 0.28209479177387814;
const float SH_C1 = 0.4886025119029199;
const float SH_C2_0 = 1.0925484305920792;
const float SH_C2_1 = -1.0925484305920792;
const float SH_C2_2 = 0.31539156525252005;
const float SH_C2_3 = -1.0925484305920792;
const float SH_C2_4 = 0.5462742152960396;
const float SH_C3_0 = -0.5900435899266435;
const float SH_C3_1 = 2.890611442640554;
const float SH_C3_2 = -0.4570457994644658;
const float SH_C3_3 = 0.3731763325901154;
const float SH_C3_4 = -0.4570457994644658;
const float SH_C3_5 = 1.445305721320277;
const float SH_C3_6 = -0.5900435899266435;

// Real SH basis (3DGS order) at unit direction d, degrees 0..deg.
void sh_basis(vec3 d, int deg, out float b[16]) {
    for (int k = 0; k < 16; k++) b[k] = 0.0;
    b[0] = SH_C0;
    if (deg < 1) return;
    float x = d.x, y = d.y, z = d.z;
    b[1] = -SH_C1 * y; b[2] = SH_C1 * z; b[3] = -SH_C1 * x;
    if (deg < 2) return;
    float xx = x * x, yy = y * y, zz = z * z, xy = x * y, yz = y * z, xz = x * z;
    b[4] = SH_C2_0 * xy; b[5] = SH_C2_1 * yz; b[6] = SH_C2_2 * (2.0 * zz - xx - yy);
    b[7] = SH_C2_3 * xz; b[8] = SH_C2_4 * (xx - yy);
    if (deg < 3) return;
    b[9]  = SH_C3_0 * y * (3.0 * xx - yy);
    b[10] = SH_C3_1 * xy * z;
    b[11] = SH_C3_2 * y * (4.0 * zz - xx - yy);
    b[12] = SH_C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy);
    b[13] = SH_C3_4 * x * (4.0 * zz - xx - yy);
    b[14] = SH_C3_5 * z * (xx - yy);
    b[15] = SH_C3_6 * x * (xx - 3.0 * yy);
}

// d b_k / d(x, y, z) of the polynomial basis above (direction treated as free; the caller projects out the
// radial part through the normalisation).
void sh_basis_grad(vec3 d, int deg, out vec3 g[16]) {
    for (int k = 0; k < 16; k++) g[k] = vec3(0.0);
    if (deg < 1) return;
    float x = d.x, y = d.y, z = d.z;
    g[1] = vec3(0.0, -SH_C1, 0.0); g[2] = vec3(0.0, 0.0, SH_C1); g[3] = vec3(-SH_C1, 0.0, 0.0);
    if (deg < 2) return;
    float xx = x * x, yy = y * y, zz = z * z;
    g[4] = SH_C2_0 * vec3(y, x, 0.0);
    g[5] = SH_C2_1 * vec3(0.0, z, y);
    g[6] = SH_C2_2 * vec3(-2.0 * x, -2.0 * y, 4.0 * z);
    g[7] = SH_C2_3 * vec3(z, 0.0, x);
    g[8] = SH_C2_4 * vec3(2.0 * x, -2.0 * y, 0.0);
    if (deg < 3) return;
    g[9]  = SH_C3_0 * vec3(6.0 * x * y, 3.0 * xx - 3.0 * yy, 0.0);
    g[10] = SH_C3_1 * vec3(y * z, x * z, x * y);
    g[11] = SH_C3_2 * vec3(-2.0 * x * y, 4.0 * zz - xx - 3.0 * yy, 8.0 * y * z);
    g[12] = SH_C3_3 * vec3(-6.0 * x * z, -6.0 * y * z, 6.0 * zz - 3.0 * xx - 3.0 * yy);
    g[13] = SH_C3_4 * vec3(4.0 * zz - 3.0 * xx - yy, -2.0 * x * y, 8.0 * x * z);
    g[14] = SH_C3_5 * vec3(2.0 * x * z, -2.0 * y * z, xx - yy);
    g[15] = SH_C3_6 * vec3(3.0 * xx - 3.0 * yy, -6.0 * x * y, 0.0);
}

// Row-major rotation matrix R[r][c] of a unit quaternion q = (w, x, y, z).
void quat_rot(vec4 q, out float R[3][3]) {
    float w = q.x, x = q.y, y = q.z, z = q.w;
    R[0][0] = 1.0 - 2.0 * (y * y + z * z); R[0][1] = 2.0 * (x * y - w * z);       R[0][2] = 2.0 * (x * z + w * y);
    R[1][0] = 2.0 * (x * y + w * z);       R[1][1] = 1.0 - 2.0 * (x * x + z * z); R[1][2] = 2.0 * (y * z - w * x);
    R[2][0] = 2.0 * (x * z - w * y);       R[2][1] = 2.0 * (y * z + w * x);       R[2][2] = 1.0 - 2.0 * (x * x + y * y);
}

// Row-major world → camera rotation W[r][c] of the frame.
void frame_rot(Frame F, out float W[3][3]) {
    for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) W[r][c] = F.w2c[c][r];
}

// Counter-based random numbers (PCG hash), identical to vkt::hashNormal on the host: a random number is a
// function of (seed, step, index, k) only, so there is no generator state to save for pause / resume.
uint pcg_hash(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
uint rng_key(uint seed, uint step, uint idx, uint k) {
    return pcg_hash(seed ^ pcg_hash(step ^ pcg_hash(idx * 8u + k)));
}
// Standard normal by Box-Muller from two hashed uniforms.
float hash_normal(uint seed, uint step, uint idx, uint k) {
    uint a = rng_key(seed, step, idx, 2u * k), b = rng_key(seed, step, idx, 2u * k + 1u);
    float u1 = (float(a >> 8u) + 1.0) * (1.0 / 16777216.0);   // (0, 1]
    float u2 = float(b >> 8u) * (1.0 / 16777216.0);           // [0, 1)
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

// One float-accumulation buffer per kernel: binding B holds `atomicAcc`, added to with accAdd(i, v) — a native
// float atomic (GL_EXT_shader_atomic_float, NATIVE_FLOAT_ATOMICS builds) or a compare-and-swap loop on the bits.
#ifdef NATIVE_FLOAT_ATOMICS
#define DECLARE_ATOMIC_ACC(B) \
    layout(std430, binding = B) buffer AtomicAccBuf { float atomicAcc[]; }; \
    void accAdd(uint i, float v) { atomicAdd(atomicAcc[i], v); }
#else
#define DECLARE_ATOMIC_ACC(B) \
    layout(std430, binding = B) buffer AtomicAccBuf { uint atomicAcc[]; }; \
    void accAdd(uint i, float v) { \
        uint old = atomicAcc[i]; \
        for (;;) { \
            uint nv = floatBitsToUint(uintBitsToFloat(old) + v); \
            uint prev = atomicCompSwap(atomicAcc[i], old, nv); \
            if (prev == old) break; \
            old = prev; \
        } \
    }
#endif

const float LUMA_R = 0.299, LUMA_G = 0.587, LUMA_B = 0.114;

// ---- learned skybox (linux/splatworker/skybox.py): an equirectangular panorama of logits [H][W][rgb] (W = 2H),
// sampled like torch grid_sample(bilinear, padding "border", align_corners=False) at
// (longitude / π, −latitude / (π/2)) of the pixel's viewing direction in the sky basis (right, up, forward).
// rot rows: M = B · Rᵀ, camera direction → sky coordinates.
struct SkyTap { uint idx[4]; float w[4]; uint key; };
SkyTap sky_tap(Frame F, uvec2 p, vec3 r0, vec3 r1, vec3 r2, uint W, uint H) {
    vec3 dc = vec3((float(p.x) + 0.5 - F.intr.z) / F.intr.x, (float(p.y) + 0.5 - F.intr.w) / F.intr.y, 1.0);
    vec3 s = normalize(vec3(dot(r0, dc), dot(r1, dc), dot(r2, dc)));
    float lon = atan(s.x, s.z), lat = asin(clamp(s.y, -1.0, 1.0));
    float gx = lon / 3.141592653589793, gy = -lat / 1.5707963267948966;
    float ix = clamp(((gx + 1.0) * float(W) - 1.0) * 0.5, 0.0, float(W - 1u));
    float iy = clamp(((gy + 1.0) * float(H) - 1.0) * 0.5, 0.0, float(H - 1u));
    float fx0 = floor(ix), fy0 = floor(iy);
    uint x0 = uint(fx0), y0 = uint(fy0);
    uint x1 = min(x0 + 1u, W - 1u), y1 = min(y0 + 1u, H - 1u);
    float ax = ix - fx0, ay = iy - fy0;
    SkyTap t;
    t.idx[0] = y0 * W + x0; t.idx[1] = y0 * W + x1; t.idx[2] = y1 * W + x0; t.idx[3] = y1 * W + x1;
    t.w[0] = (1.0 - ax) * (1.0 - ay); t.w[1] = ax * (1.0 - ay); t.w[2] = (1.0 - ax) * ay; t.w[3] = ax * ay;
    t.key = y0 * W + x0;
    return t;
}
float sigmoid1(float x) { return 1.0 / (1.0 + exp(-x)); }

// ---- per-photo local colour correction (trainer.py BilateralGrid): 12 coefficient grids of 8 (luminance) × 16 (y)
// × 16 (x), index ((k·8 + z)·16 + y)·16 + x, k = row·4 + column of the 3×4 transform [A | b]. Sampled like torch
// grid_sample(trilinear, border, align_corners=True) at ((x+½)/W, (y+½)/H, clamp(luminance, 0, 1)).
const int BG_W = 16, BG_H = 16, BG_D = 8;
int bg_index(int k, int z, int y, int x) { return ((k * BG_D + z) * BG_H + y) * BG_W + x; }
void bg_coords(uvec2 p, uint W, uint H, float lum, out ivec3 i0, out vec3 f) {
    vec3 g = vec3((float(p.x) + 0.5) / float(W) * float(BG_W - 1), (float(p.y) + 0.5) / float(H) * float(BG_H - 1),
                  clamp(lum, 0.0, 1.0) * float(BG_D - 1));
    i0 = clamp(ivec3(floor(g)), ivec3(0), ivec3(BG_W - 2, BG_H - 2, BG_D - 2));
    f = clamp(g - vec3(i0), 0.0, 1.0);
}

#endif
