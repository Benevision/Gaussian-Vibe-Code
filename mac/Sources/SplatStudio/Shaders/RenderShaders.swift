extension ShaderSource {
    /// Real-time viewer: compute-shader projection + GPU depth sort + hardware-rasterised,
    /// alpha-blended instanced quads. Apple GPUs (tile-based deferred) blend in on-chip tile
    /// memory, which makes this path much faster than a compute rasteriser for display.
    static let render = #"""
struct SplatVertex {
    float4 axes;     // major axis xy, minor axis xy (pixels, already scaled by extent)
    float2 center;   // pixels
    half4 color;     // rgb, alpha
};

// Converts trainer-format parameters into the compact render format.
kernel void prepare_splats(device const float* pos [[buffer(0)]],
                           device const float* logScale [[buffer(1)]],
                           device const float* quat [[buffer(2)]],
                           device const float* opacityLogit [[buffer(3)]],
                           device const float* sh [[buffer(4)]],
                           device float4* gpos [[buffer(5)]],
                           device float4* gcov [[buffer(6)]],
                           device half* gsh [[buffer(7)]],
                           constant uint& n [[buffer(8)]],
                           uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float3 p = float3(pos[i*3], pos[i*3+1], pos[i*3+2]);
    float3 s = exp(float3(logScale[i*3], logScale[i*3+1], logScale[i*3+2]));
    float4 q = float4(quat[i*4], quat[i*4+1], quat[i*4+2], quat[i*4+3]);
    float ql = length(q); q = ql > 0 ? q / ql : float4(1,0,0,0);
    float3x3 R = quat_to_rot(q);
    float3x3 M = float3x3(R[0] * s.x, R[1] * s.y, R[2] * s.z);
    float3x3 S = M * transpose(M);
    float o = 1.0f / (1.0f + exp(-opacityLogit[i]));
    gpos[i] = float4(p, o);
    gcov[i*2]   = float4(S[0][0], S[0][1], S[0][2], S[1][1]);
    gcov[i*2+1] = float4(S[1][2], S[2][2], 0, 0);
    for (uint k = 0; k < 48; k++) gsh[i*48 + k] = half(clamp(sh[i*48 + k], -65000.0f, 65000.0f));
}

inline float3 turbo(float t) {
    t = clamp(t, 0.0f, 1.0f);
    const float4 kR = float4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
    const float4 kG = float4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
    const float4 kB = float4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
    const float2 kR2 = float2(-152.94239396, 59.28637943);
    const float2 kG2 = float2(4.27729857, 2.82956604);
    const float2 kB2 = float2(-89.90310912, 27.34824973);
    float4 v4 = float4(1.0, t, t*t, t*t*t);
    float2 v2 = v4.zw * v4.z;
    return float3(dot(v4, kR) + dot(v2, kR2), dot(v4, kG) + dot(v2, kG2), dot(v4, kB) + dot(v2, kB2));
}

kernel void reset_draw_args(device uint* a [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i == 0) { a[0] = 4; a[1] = 0; a[2] = 0; a[3] = 0; }
}

struct DrawArgs { uint vertexCount; atomic_uint instanceCount; uint vertexStart; uint baseInstance; };

kernel void project_splats(device const float4* gpos [[buffer(0)]],
                           device const float4* gcov [[buffer(1)]],
                           device const half* gsh [[buffer(2)]],
                           device const uchar* gflags [[buffer(3)]],
                           constant CameraUniforms& cam [[buffer(4)]],
                           device SplatVertex* verts [[buffer(5)]],
                           device uint* keys [[buffer(6)]],
                           device uint* vals [[buffer(7)]],
                           device DrawArgs* args [[buffer(8)]],
                           constant uint& sixteenBit [[buffer(9)]],
                           device const float* coverage [[buffer(10)]],
                           constant uint& hasCoverage [[buffer(11)]],
                           uint i [[thread_position_in_grid]]) {
    if (i >= cam.count) return;
    vals[i] = i;
    uint invisible = sixteenBit ? 0xFFFFu : 0xFFFFFFFFu;
    keys[i] = invisible;
    float4 po = gpos[i];
    uchar fl = gflags[i];
    if (fl & 1) return; // deleted
    float3 mean = po.xyz;
    uint mode = (cam.flags >> 8) & 0xFF;
    if ((cam.flags & 2) && (any(mean < cam.cropMin.xyz) || any(mean > cam.cropMax.xyz))) return;
    float4 c0 = gcov[i*2], c1 = gcov[i*2+1];
    float sc2 = cam.splatScale * cam.splatScale;
    float3 t; float3x3 J, W;
    float3 cov = project_cov(mean, c0.xyz * sc2, float3(c0.w, c1.x, c1.y) * sc2, cam, t, J, W);
    float near = cam.cropMin.w, far = cam.cropMax.w;
    if (t.z < near) return;
    float dil = cam.misc.z;
    float det0 = cov.x * cov.z - cov.y * cov.y;
    cov.x += dil; cov.z += dil;
    float det = cov.x * cov.z - cov.y * cov.y;
    if (det <= 0) return;
    float opacity = po.w;
    if (cam.flags & 1) opacity *= sqrt(max(det0, 0.0f) / det);
    if (opacity < cam.misc.x) return;
    float2 center = float2(cam.intr.x * t.x / t.z + cam.intr.z, cam.intr.y * t.y / t.z + cam.intr.w);
    float mid = 0.5f * (cov.x + cov.z);
    float disc = sqrt(max(0.1f, mid * mid - det));
    float l1 = mid + disc, l2 = max(mid - disc, 0.1f);
    float2 v1 = abs(cov.y) > 1e-7f ? normalize(float2(cov.y, l1 - cov.x)) : (cov.x >= cov.z ? float2(1,0) : float2(0,1));
    float2 v2 = float2(-v1.y, v1.x);
    // Opacity-aware extent: beyond this the gaussian contributes < 1/255.
    float k = min(3.0f, sqrt(max(2.0f * log(max(255.0f * opacity, 1.0001f)), 0.0f)));
    float r1 = k * sqrt(l1), r2 = k * sqrt(l2);
    if (mode == 2) { r1 = 2.0f; r2 = 2.0f; v1 = float2(1,0); v2 = float2(0,1); }
    if (r1 < 0.3f) return;
    if (center.x + r1 < 0 || center.y + r1 < 0 || center.x - r1 > cam.viewport.x || center.y - r1 > cam.viewport.y) return;

    float3 rgb;
    if (mode == 1) {
        float d = (log(t.z) - log(near)) / (log(far) - log(near));
        rgb = turbo(1.0f - d);
    } else if (mode == 4) {
        rgb = turbo(po.w);
    } else if (mode == 5) {
        // coverage: red (poorly seen) → yellow → green (well seen); grey until computed
        if (hasCoverage) {
            float c = coverage[i];
            rgb = c < 0.5f ? mix(float3(0.86, 0.16, 0.12), float3(0.96, 0.78, 0.12), c * 2.0f)
                           : mix(float3(0.96, 0.78, 0.12), float3(0.20, 0.72, 0.32), (c - 0.5f) * 2.0f);
        } else rgb = float3(0.55);
    } else {
        float b[16];
        int deg = int(cam.shDegree);
        float3 dir = normalize(mean - cam.camPos.xyz);
        sh_basis(dir, deg, b);
        rgb = float3(0);
        int nc = (deg + 1) * (deg + 1);
        for (int k2 = 0; k2 < nc; k2++) {
            rgb += b[k2] * float3(gsh[i*48 + k2*3], gsh[i*48 + k2*3 + 1], gsh[i*48 + k2*3 + 2]);
        }
        rgb = max(rgb + 0.5f, 0.0f);
    }
    if ((fl & 2) && cam.misc.y > 0) rgb = mix(rgb, float3(1.0, 0.45, 0.1), 0.6f);
    if (mode == 2) opacity = 1.0f;

    SplatVertex v;
    v.axes = float4(v1 * r1, v2 * r2);
    v.center = center;
    v.color = half4(half3(rgb), half(opacity));
    verts[i] = v;
    if (sixteenBit) {
        float dn = clamp((log(t.z) - log(near)) / (log(far) - log(near)), 0.0f, 1.0f);
        keys[i] = 0xFFFEu - uint(dn * 65534.0f);
    } else {
        keys[i] = ~as_type<uint>(t.z);
    }
    atomic_fetch_add_explicit(&args->instanceCount, 1, memory_order_relaxed);
}

struct SplatVSOut {
    float4 position [[position]];
    float2 uv;          // position in units of sigma along the splat axes (scaled by extent)
    half4 color;
    float extent;
};

vertex SplatVSOut splat_vertex(uint vid [[vertex_id]], uint iid [[instance_id]],
                               device const SplatVertex* verts [[buffer(0)]],
                               device const uint* order [[buffer(1)]],
                               constant CameraUniforms& cam [[buffer(2)]]) {
    SplatVertex v = verts[order[iid]];
    float2 corner = float2((vid & 1) ? 1.0f : -1.0f, (vid & 2) ? 1.0f : -1.0f);
    float2 p = v.center + corner.x * v.axes.xy + corner.y * v.axes.zw;
    float2 ndc = float2(p.x / cam.viewport.x * 2.0f - 1.0f, 1.0f - p.y / cam.viewport.y * 2.0f);
    SplatVSOut o;
    o.position = float4(ndc, 0.5f, 1.0f);
    float r1 = length(v.axes.xy);
    float l1 = r1;
    // extent k is encoded implicitly: r = k*sigma, so uv in sigma units = corner*k.
    float a = float(v.color.a);
    float k = min(3.0f, sqrt(max(2.0f * log(max(255.0f * a, 1.0001f)), 0.0f)));
    o.uv = corner * k;
    o.extent = k;
    o.color = v.color;
    (void)l1;
    return o;
}

fragment half4 splat_fragment(SplatVSOut in [[stage_in]], constant CameraUniforms& cam [[buffer(0)]]) {
    uint mode = (cam.flags >> 8) & 0xFF;
    float r2 = dot(in.uv, in.uv);
    float a;
    if (mode == 2) {
        a = r2 < in.extent * in.extent ? 1.0f : 0.0f;
    } else if (mode == 3) {
        float r = sqrt(r2);
        a = (r > 0.85f * in.extent && r < in.extent) ? 1.0f : 0.0f;
    } else {
        a = float(in.color.a) * exp(-0.5f * r2);
    }
    if (a < 1.0f / 255.0f) discard_fragment();
    a = min(a, 0.99f);
    return half4(in.color.rgb * half(a), half(a));
}

// Simple coloured lines (camera frustums, crop box, grid, axes).
struct LineVertex { float3 pos; float4 color; };
struct LineVSOut { float4 position [[position]]; float4 color; };

vertex LineVSOut line_vertex(uint vid [[vertex_id]], device const LineVertex* v [[buffer(0)]],
                             constant float4x4& viewProj [[buffer(1)]]) {
    LineVSOut o;
    o.position = viewProj * float4(v[vid].pos, 1.0f);
    o.color = v[vid].color;
    return o;
}
fragment half4 line_fragment(LineVSOut in [[stage_in]]) { return half4(half3(in.color.rgb) * half(in.color.a), half(in.color.a)); }

// Learned skybox: a full-screen triangle that looks up each pixel's world direction in the equirectangular
// panorama (longitude atan2(right, forward), latitude asin(up); top row = straight up), drawn before the splats.
struct SkyVSOut { float4 pos [[position]]; };
vertex SkyVSOut sky_vertex(uint vid [[vertex_id]]) {
    float2 p = float2((vid << 1) & 2, vid & 2);
    SkyVSOut o; o.pos = float4(p * 2.0 - 1.0, 0.0, 1.0); return o;
}
fragment half4 sky_fragment(SkyVSOut in [[stage_in]], constant CameraUniforms& u [[buffer(0)]],
                            constant float3x3& basis [[buffer(1)]], texture2d<half> tex [[texture(0)]]) {
    constexpr sampler s(filter::linear, s_address::repeat, t_address::clamp_to_edge);
    float3 dc = float3((in.pos.x - u.intr.z) / u.intr.x, (in.pos.y - u.intr.w) / u.intr.y, 1.0);
    float3x3 R = float3x3(u.worldToCam[0].xyz, u.worldToCam[1].xyz, u.worldToCam[2].xyz);
    float3 d = normalize(transpose(R) * dc);
    float3 k = basis * d;                                   // (right, up, forward) components
    float lon = atan2(k.x, k.z), lat = asin(clamp(k.y, -1.0, 1.0));
    float2 uv = float2(lon / (2.0 * M_PI_F) + 0.5, 0.5 - lat / M_PI_F);
    return half4(tex.sample(s, uv).rgb, 1.0h);
}
"""#
}
