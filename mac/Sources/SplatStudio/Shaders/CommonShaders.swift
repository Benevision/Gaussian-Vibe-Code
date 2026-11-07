// Metal shader sources are embedded as Swift strings and compiled once at launch with
// `MTLDevice.makeLibrary(source:)`. This keeps the app buildable with the Command Line
// Tools alone (no offline `metal` compiler needed) and lets us specialise via #defines.

enum ShaderSource {
    static var all: String { [common, sort, render, train, coverage].joined(separator: "\n") }

    static let common = #"""
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

constant float SH_C0 = 0.28209479177387814f;
constant float SH_C1 = 0.4886025119029199f;
constant float SH_C2[5] = { 1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f, -1.0925484305920792f, 0.5462742152960396f };
constant float SH_C3[7] = { -0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f, 0.3731763325901154f, -0.4570457994644658f, 1.445305721320277f, -0.5900435899266435f };

// Real spherical-harmonic basis in the 3DGS ordering, evaluated for a unit direction.
inline void sh_basis(float3 d, int degree, thread float* b) {
    b[0] = SH_C0;
    if (degree < 1) return;
    float x = d.x, y = d.y, z = d.z;
    b[1] = -SH_C1 * y; b[2] = SH_C1 * z; b[3] = -SH_C1 * x;
    if (degree < 2) return;
    float xx = x*x, yy = y*y, zz = z*z, xy = x*y, yz = y*z, xz = x*z;
    b[4] = SH_C2[0] * xy; b[5] = SH_C2[1] * yz; b[6] = SH_C2[2] * (2.0f*zz - xx - yy);
    b[7] = SH_C2[3] * xz; b[8] = SH_C2[4] * (xx - yy);
    if (degree < 3) return;
    b[9]  = SH_C3[0] * y * (3.0f*xx - yy);
    b[10] = SH_C3[1] * xy * z;
    b[11] = SH_C3[2] * y * (4.0f*zz - xx - yy);
    b[12] = SH_C3[3] * z * (2.0f*zz - 3.0f*xx - 3.0f*yy);
    b[13] = SH_C3[4] * x * (4.0f*zz - xx - yy);
    b[14] = SH_C3[5] * z * (xx - yy);
    b[15] = SH_C3[6] * x * (xx - 3.0f*yy);
}

inline float3x3 quat_to_rot(float4 q) { // q = (w, x, y, z), normalised
    float w = q.x, x = q.y, y = q.z, z = q.w;
    // Columns of R
    return float3x3(
        float3(1.f - 2.f*(y*y + z*z), 2.f*(x*y + w*z),       2.f*(x*z - w*y)),
        float3(2.f*(x*y - w*z),       1.f - 2.f*(x*x + z*z), 2.f*(y*z + w*x)),
        float3(2.f*(x*z + w*y),       2.f*(y*z - w*x),       1.f - 2.f*(x*x + y*y)));
}

// Per-view camera description shared by viewer and trainer (OpenCV convention).
struct CameraUniforms {
    float4x4 worldToCam;   // rigid transform
    float4 intr;           // fx, fy, cx, cy (pixels)
    float4 viewport;       // width, height, tanHalfFovX*1.3 clamp, tanHalfFovY*1.3 clamp
    float4 camPos;         // xyz world position
    uint count;            // number of gaussians
    uint shDegree;         // active SH degree
    float splatScale;      // user scale multiplier (viewer)
    uint flags;            // bit0 antialiased, bit1 crop enabled, bits 8..15 display mode
    float4 cropMin;        // xyz, w = near plane
    float4 cropMax;        // xyz, w = far plane (depth colour mapping / sort range)
    float4 misc;           // x: alpha cutoff, y: highlight selection, z: dilation, w: unused
};

// Projects a 3D covariance to the screen (EWA splatting, 3DGS eq. 5) and returns the
// camera-space point via `t`.
inline float3 project_cov(float3 mean, float3 cA, float3 cB, constant CameraUniforms& cam,
                          thread float3& t, thread float3x3& Jout, thread float3x3& W) {
    float4 tc = cam.worldToCam * float4(mean, 1.0f);
    t = tc.xyz;
    float fx = cam.intr.x, fy = cam.intr.y;
    float limx = cam.viewport.z, limy = cam.viewport.w;
    float txz = clamp(t.x / t.z, -limx, limx) * t.z;
    float tyz = clamp(t.y / t.z, -limy, limy) * t.z;
    float iz = 1.0f / t.z;
    float3x3 J = float3x3(float3(fx * iz, 0, 0), float3(0, fy * iz, 0), float3(-fx * txz * iz * iz, -fy * tyz * iz * iz, 0));
    W = float3x3(cam.worldToCam[0].xyz, cam.worldToCam[1].xyz, cam.worldToCam[2].xyz);
    float3x3 S = float3x3(float3(cA.x, cA.y, cA.z), float3(cA.y, cB.x, cB.y), float3(cA.z, cB.y, cB.z));
    float3x3 T = J * W;
    float3x3 C = T * S * transpose(T);
    Jout = J;
    return float3(C[0][0], C[0][1], C[1][1]);
}

inline uint float_to_ordered(float f) {
    uint u = as_type<uint>(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
"""#
}
