extension ShaderSource {
    /// Differentiable tile rasteriser (3DGS forward/backward), fused SSIM loss, Adam and
    /// 3DGS-MCMC noise. All per-gaussian parameters use the GaussianCloud layout.
    static let train = #"""
#define TILE 16

struct RasterParams {
    uint width; uint height; uint tilesX; uint tilesY;
    float4 bg;             // rgb background, w unused
    uint absGrad; uint hasDepth; uint numGaussians; uint step;
    float ssimWeight; float depthWeight; float invNumPixels; float pad;
};

// ---------------------------------------------------------------- preprocess (forward)
kernel void train_preprocess(device const float* pos [[buffer(0)]],
                             device const float* logScale [[buffer(1)]],
                             device const float* quat [[buffer(2)]],
                             device const float* opLogit [[buffer(3)]],
                             device const float* sh [[buffer(4)]],
                             constant CameraUniforms& cam [[buffer(5)]],
                             constant RasterParams& rp [[buffer(6)]],
                             device float2* p2d [[buffer(7)]],
                             device float4* conicO [[buffer(8)]],
                             device float4* rgbd [[buffer(9)]],
                             device uint4* rects [[buffer(10)]],
                             device uint* tilesTouched [[buffer(11)]],
                             device uint* depthKeys [[buffer(12)]],
                             device uint* sortVals [[buffer(13)]],
                             device const uchar* alive [[buffer(14)]],
                             uint i [[thread_position_in_grid]]) {
    if (i >= cam.count) return;
    sortVals[i] = i;
    tilesTouched[i] = 0;
    depthKeys[i] = 0xFFFFFFFFu;
    rects[i] = uint4(0);
    if (alive[i] == 0) return;
    float3 p = float3(pos[i*3], pos[i*3+1], pos[i*3+2]);
    float3 s = exp(float3(logScale[i*3], logScale[i*3+1], logScale[i*3+2]));
    float4 q = float4(quat[i*4], quat[i*4+1], quat[i*4+2], quat[i*4+3]);
    float ql = length(q); q = ql > 1e-12f ? q / ql : float4(1,0,0,0);
    float3x3 R = quat_to_rot(q);
    float3x3 M = float3x3(R[0] * s.x, R[1] * s.y, R[2] * s.z);
    float3x3 S = M * transpose(M);
    float3 t; float3x3 J, W;
    float3 cov = project_cov(p, float3(S[0][0], S[0][1], S[0][2]), float3(S[1][1], S[1][2], S[2][2]), cam, t, J, W);
    if (t.z < 0.01f) return;
    float dil = cam.misc.z;
    float det0 = cov.x * cov.z - cov.y * cov.y;
    float a = cov.x + dil, b = cov.y, c = cov.z + dil;
    float det = a * c - b * b;
    if (det <= 1e-12f) return;
    float o = 1.0f / (1.0f + exp(-opLogit[i]));
    float comp = (cam.flags & 1) ? sqrt(max(det0, 0.0f) / det) : 1.0f;
    float oe = o * comp;
    float idet = 1.0f / det;
    float3 conic = float3(c * idet, -b * idet, a * idet);
    float2 mean = float2(cam.intr.x * t.x / t.z + cam.intr.z, cam.intr.y * t.y / t.z + cam.intr.w);
    float mid = 0.5f * (a + c);
    float l1 = mid + sqrt(max(0.1f, mid * mid - det));
    float k = min(3.0f, sqrt(max(2.0f * log(max(255.0f * oe, 1.0001f)), 0.0f)));
    float radius = ceil(k * sqrt(l1));
    if (radius <= 0 || oe < 1.0f / 255.0f) return;
    // Tight axis-aligned bounds of the k-sigma ellipse (±k·sqrt(Σxx), ±k·sqrt(Σyy)) instead of
    // the circumscribed circle: far fewer tile instances for elongated splats.
    float2 ext = ceil(k * sqrt(float2(a, c)));
    int2 rmin = int2(clamp(int(floor((mean.x - ext.x) / TILE)), 0, int(rp.tilesX)), clamp(int(floor((mean.y - ext.y) / TILE)), 0, int(rp.tilesY)));
    int2 rmax = int2(clamp(int(floor((mean.x + ext.x) / TILE)) + 1, 0, int(rp.tilesX)), clamp(int(floor((mean.y + ext.y) / TILE)) + 1, 0, int(rp.tilesY)));
    uint n = uint((rmax.x - rmin.x) * (rmax.y - rmin.y));
    if (n == 0) return;
    // colour
    float bsh[16];
    int deg = int(cam.shDegree);
    float3 dir = normalize(p - cam.camPos.xyz);
    sh_basis(dir, deg, bsh);
    float3 col = float3(0);
    int nc = (deg + 1) * (deg + 1);
    for (int kk = 0; kk < nc; kk++) col += bsh[kk] * float3(sh[i*48 + kk*3], sh[i*48 + kk*3 + 1], sh[i*48 + kk*3 + 2]);
    col = max(col + 0.5f, 0.0f);
    p2d[i] = mean;
    conicO[i] = float4(conic, oe);
    rgbd[i] = float4(col, t.z);
    rects[i] = uint4(rmin.x, rmin.y, rmax.x, rmax.y);
    tilesTouched[i] = n;
    depthKeys[i] = as_type<uint>(t.z);
}

kernel void gather_counts(device const uint* sortedIdx [[buffer(0)]], device const uint* tilesTouched [[buffer(1)]],
                          device uint* out [[buffer(2)]], constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < n) out[i] = tilesTouched[sortedIdx[i]];
}

kernel void total_count(device const uint* offsets [[buffer(0)]], device const uint* counts [[buffer(1)]],
                        device uint* total [[buffer(2)]], constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i == 0) total[0] = n > 0 ? offsets[n-1] + counts[n-1] : 0;
}

kernel void duplicate_keys(device const uint* sortedIdx [[buffer(0)]], device const uint* offsets [[buffer(1)]],
                           device const uint4* rects [[buffer(2)]], device uint* tileKeys [[buffer(3)]],
                           device uint* tileVals [[buffer(4)]], constant RasterParams& rp [[buffer(5)]],
                           constant uint& n [[buffer(6)]], uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    uint g = sortedIdx[i];
    uint4 r = rects[g];
    uint off = offsets[i];
    for (uint y = r.y; y < r.w; y++)
        for (uint x = r.x; x < r.z; x++) {
            tileKeys[off] = y * rp.tilesX + x;
            tileVals[off] = g;
            off++;
        }
}

// NB: scalar stores — writing .x/.y of a shared uint2 from different threads can be
// compiled to a full-vector read-modify-write and race.
kernel void tile_ranges(device const uint* tileKeys [[buffer(0)]], device uint* ranges [[buffer(1)]],
                        constant uint& m [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= m) return;
    uint t = tileKeys[i];
    if (i == 0 || tileKeys[i-1] != t) ranges[t * 2] = i;
    if (i == m - 1 || tileKeys[i+1] != t) ranges[t * 2 + 1] = i + 1;
}

// ---------------------------------------------------------------- rasterise (forward)
kernel void rasterize_forward(device const uint2* ranges [[buffer(0)]],
                              device const uint* tileVals [[buffer(1)]],
                              device const float2* p2d [[buffer(2)]],
                              device const float4* conicO [[buffer(3)]],
                              device const float4* rgbd [[buffer(4)]],
                              constant RasterParams& rp [[buffer(5)]],
                              device float* outImg [[buffer(6)]],
                              device float* finalT [[buffer(7)]],
                              device uint* nContrib [[buffer(8)]],
                              uint2 tg [[threadgroup_position_in_grid]],
                              uint2 lt [[thread_position_in_threadgroup]],
                              uint tidx [[thread_index_in_threadgroup]]) {
    threadgroup float2 sP[256];
    threadgroup float4 sC[256];
    threadgroup float4 sF[256];
    threadgroup uint anyActive;
    uint2 pix = tg * TILE + lt;
    bool inside = pix.x < rp.width && pix.y < rp.height;
    float2 pf = float2(pix) + 0.5f;
    uint2 range = ranges[tg.y * rp.tilesX + tg.x];
    int toDo = int(range.y) - int(range.x);
    int rounds = (toDo + 255) / 256;
    float T = 1.0f;
    float4 C = float4(0);
    uint contributor = 0, last = 0;
    bool done = !inside;
    for (int r = 0; r < rounds; r++, toDo -= 256) {
        if (tidx == 0) anyActive = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!done) anyActive = 1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (anyActive == 0) break;
        uint idx = range.x + uint(r) * 256 + tidx;
        if (idx < range.y) {
            uint g = tileVals[idx];
            sP[tidx] = p2d[g]; sC[tidx] = conicO[g]; sF[tidx] = rgbd[g];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int cnt = min(256, toDo);
        for (int j = 0; j < cnt && !done; j++) {
            contributor++;
            float2 d = sP[j] - pf;
            float4 con = sC[j];
            float power = -0.5f * (con.x * d.x * d.x + con.z * d.y * d.y) - con.y * d.x * d.y;
            if (power > 0.0f) continue;
            float alpha = min(0.99f, con.w * exp(power));
            if (alpha < 1.0f / 255.0f) continue;
            float testT = T * (1.0f - alpha);
            if (testT < 0.0001f) { done = true; continue; }
            C += sF[j] * (alpha * T);
            T = testT;
            last = contributor;
        }
    }
    if (inside) {
        uint pid = pix.y * rp.width + pix.x;
        uint hw = rp.width * rp.height;
        outImg[pid] = C.x + T * rp.bg.x;
        outImg[hw + pid] = C.y + T * rp.bg.y;
        outImg[2*hw + pid] = C.z + T * rp.bg.z;
        outImg[3*hw + pid] = C.w;
        finalT[pid] = T;
        nContrib[pid] = last;
    }
}

// ---------------------------------------------------------------- per-splat peak contribution
// Same compositing as rasterize_forward, recording for every splat the most it adds to any pixel (alpha * T).
// Non-negative floats order like their bit patterns, so an atomic max on the bits is a float max.
kernel void rasterize_peak(device const uint2* ranges [[buffer(0)]],
                           device const uint* tileVals [[buffer(1)]],
                           device const float2* p2d [[buffer(2)]],
                           device const float4* conicO [[buffer(3)]],
                           constant RasterParams& rp [[buffer(5)]],
                           device atomic_uint* peak [[buffer(9)]],
                           uint2 tg [[threadgroup_position_in_grid]],
                           uint2 lt [[thread_position_in_threadgroup]],
                           uint tidx [[thread_index_in_threadgroup]]) {
    threadgroup float2 sP[256];
    threadgroup float4 sC[256];
    threadgroup uint sG[256];
    threadgroup uint anyActive;
    uint2 pix = tg * TILE + lt;
    bool inside = pix.x < rp.width && pix.y < rp.height;
    float2 pf = float2(pix) + 0.5f;
    uint2 range = ranges[tg.y * rp.tilesX + tg.x];
    int toDo = int(range.y) - int(range.x);
    int rounds = (toDo + 255) / 256;
    float T = 1.0f;
    bool done = !inside;
    for (int r = 0; r < rounds; r++, toDo -= 256) {
        if (tidx == 0) anyActive = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!done) anyActive = 1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (anyActive == 0) break;
        uint idx = range.x + uint(r) * 256 + tidx;
        if (idx < range.y) { uint g = tileVals[idx]; sP[tidx] = p2d[g]; sC[tidx] = conicO[g]; sG[tidx] = g; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int cnt = min(256, toDo);
        for (int j = 0; j < cnt && !done; j++) {
            float2 d = sP[j] - pf;
            float4 con = sC[j];
            float power = -0.5f * (con.x * d.x * d.x + con.z * d.y * d.y) - con.y * d.x * d.y;
            if (power > 0.0f) continue;
            float alpha = min(0.99f, con.w * exp(power));
            if (alpha < 1.0f / 255.0f) continue;
            float testT = T * (1.0f - alpha);
            if (testT < 0.0001f) { done = true; continue; }
            atomic_fetch_max_explicit(&peak[sG[j]], as_type<uint>(alpha * T), memory_order_relaxed);
            T = testT;
        }
    }
}

// Per-view pass of the contribution census: fold this view's peak (α·T bit patterns from rasterize_peak)
// into the overall peak, and count the views in which each splat contributes at least `thr`.
// Error-driven densification: per splat, the number of high-error pixels (`hot`) it visibly contributes to.
kernel void rasterize_errcount(device const uint2* ranges [[buffer(0)]],
                           device const uint* tileVals [[buffer(1)]],
                           device const float2* p2d [[buffer(2)]],
                           device const float4* conicO [[buffer(3)]],
                           constant RasterParams& rp [[buffer(5)]],
                           device atomic_uint* errCount [[buffer(9)]],
                           device const uchar* hot [[buffer(10)]],
                           uint2 tg [[threadgroup_position_in_grid]],
                           uint2 lt [[thread_position_in_threadgroup]],
                           uint tidx [[thread_index_in_threadgroup]]) {
    threadgroup float2 sP[256];
    threadgroup float4 sC[256];
    threadgroup uint sG[256];
    threadgroup uint anyActive;
    uint2 pix = tg * TILE + lt;
    bool inside = pix.x < rp.width && pix.y < rp.height;
    float2 pf = float2(pix) + 0.5f;
    uint2 range = ranges[tg.y * rp.tilesX + tg.x];
    int toDo = int(range.y) - int(range.x);
    int rounds = (toDo + 255) / 256;
    float T = 1.0f;
    bool done = !inside;
    bool isHot = inside && hot[pix.y * rp.width + pix.x] != 0;
    if (!isHot) done = true;   // only high-error pixels count (FastGS-style error-driven densification)
    for (int r = 0; r < rounds; r++, toDo -= 256) {
        if (tidx == 0) anyActive = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!done) anyActive = 1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (anyActive == 0) break;
        uint idx = range.x + uint(r) * 256 + tidx;
        if (idx < range.y) { uint g = tileVals[idx]; sP[tidx] = p2d[g]; sC[tidx] = conicO[g]; sG[tidx] = g; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int cnt = min(256, toDo);
        for (int j = 0; j < cnt && !done; j++) {
            float2 d = sP[j] - pf;
            float4 con = sC[j];
            float power = -0.5f * (con.x * d.x * d.x + con.z * d.y * d.y) - con.y * d.x * d.y;
            if (power > 0.0f) continue;
            float alpha = min(0.99f, con.w * exp(power));
            if (alpha < 1.0f / 255.0f) continue;
            float testT = T * (1.0f - alpha);
            if (testT < 0.0001f) { done = true; continue; }
            atomic_fetch_add_explicit(&errCount[sG[j]], 1u, memory_order_relaxed);
            T = testT;
        }
    }
}

// Per-view pass of the contribution census: fold this view's peak (α·T bit patterns from rasterize_peak)
// into the overall peak, and count the views in which each splat contributes at least `thr`.
kernel void peak_accumulate(device const float* viewPeak [[buffer(0)]],
                            device float* peak [[buffer(1)]],
                            device uint* views [[buffer(2)]],
                            constant float& thr [[buffer(3)]],
                            constant uint& n [[buffer(4)]],
                            uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float v = viewPeak[i];
    peak[i] = max(peak[i], v);
    if (v >= thr) views[i] += 1u;
}

// ---------------------------------------------------------------- rasterise (backward)
// Accumulator layout per gaussian (12 floats): dmean2D.xy, dconic.abc, dopacity, drgb, ddepth, |dmean2D|.xy
kernel void rasterize_backward(device const uint2* ranges [[buffer(0)]],
                               device const uint* tileVals [[buffer(1)]],
                               device const float2* p2d [[buffer(2)]],
                               device const float4* conicO [[buffer(3)]],
                               device const float4* rgbd [[buffer(4)]],
                               constant RasterParams& rp [[buffer(5)]],
                               device const float* finalT [[buffer(6)]],
                               device const uint* nContrib [[buffer(7)]],
                               device const float* gradImg [[buffer(8)]],
                               device atomic<float>* acc [[buffer(9)]],
                               uint2 tg [[threadgroup_position_in_grid]],
                               uint2 lt [[thread_position_in_threadgroup]],
                               uint tidx [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
    // Gradients are reduced across each SIMD-group with simd_sum and added with one atomic per
    // SIMD-group (measured faster on M4 than staging through threadgroup memory).
    threadgroup float2 sP[256];
    threadgroup float4 sC[256];
    threadgroup float4 sF[256];
    threadgroup uint sI[256];
    uint2 pix = tg * TILE + lt;
    bool inside = pix.x < rp.width && pix.y < rp.height;
    float2 pf = float2(pix) + 0.5f;
    uint2 range = ranges[tg.y * rp.tilesX + tg.x];
    int toDo = int(range.y) - int(range.x);
    int rounds = (toDo + 255) / 256;
    uint pid = pix.y * rp.width + pix.x;
    uint hw = rp.width * rp.height;
    float Tfinal = inside ? finalT[pid] : 1.0f;
    float T = Tfinal;
    uint last = inside ? nContrib[pid] : 0;
    float4 dpix = inside ? float4(gradImg[pid], gradImg[hw + pid], gradImg[2*hw + pid], gradImg[3*hw + pid]) : float4(0);
    float bgDot = dot(rp.bg.xyz, dpix.xyz);
    float4 accumRec = float4(0), lastColor = float4(0);
    float lastAlpha = 0.0f;
    uint contributor = uint(toDo);
    for (int r = 0; r < rounds; r++, toDo -= 256) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int progress = r * 256 + int(tidx);
        if (int(range.x) + progress < int(range.y)) {
            uint g = tileVals[range.y - uint(progress) - 1];
            sI[tidx] = g; sP[tidx] = p2d[g]; sC[tidx] = conicO[g]; sF[tidx] = rgbd[g];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int cnt = min(256, toDo);
        for (int j = 0; j < cnt; j++) {
            bool active = inside;
            if (active) { contributor--; active = contributor < last; }
            float2 d = sP[j] - pf;
            float4 con = sC[j];
            float power = -0.5f * (con.x * d.x * d.x + con.z * d.y * d.y) - con.y * d.x * d.y;
            float G = exp(power);
            float alpha = min(0.99f, con.w * G);
            active = active && power <= 0.0f && alpha >= 1.0f / 255.0f;
            if (!simd_any(active)) continue;
            float gm0 = 0, gm1 = 0, gc0 = 0, gc1 = 0, gc2 = 0, go = 0, gr = 0, gg = 0, gb = 0, gdep = 0;
            if (active) {
                T = T / (1.0f - alpha);
                float dchdc = alpha * T;
                float4 c = sF[j];
                accumRec = lastAlpha * lastColor + (1.0f - lastAlpha) * accumRec;
                lastColor = c;
                float dLda = dot(c - accumRec, dpix);
                gr = dchdc * dpix.x; gg = dchdc * dpix.y; gb = dchdc * dpix.z; gdep = dchdc * dpix.w;
                dLda *= T;
                lastAlpha = alpha;
                dLda += (-Tfinal / (1.0f - alpha)) * bgDot;
                float dLdG = con.w * dLda;
                float gdx = G * d.x, gdy = G * d.y;
                gm0 = dLdG * (-gdx * con.x - gdy * con.y);
                gm1 = dLdG * (-gdy * con.z - gdx * con.y);
                gc0 = -0.5f * gdx * d.x * dLdG;
                gc1 = -gdx * d.y * dLdG;
                gc2 = -0.5f * gdy * d.y * dLdG;
                go = G * dLda;
            }
            float s0 = simd_sum(gm0), s1 = simd_sum(gm1);
            float s2 = simd_sum(gc0), s3 = simd_sum(gc1), s4 = simd_sum(gc2);
            float s5 = simd_sum(go), s6 = simd_sum(gr), s7 = simd_sum(gg), s8 = simd_sum(gb);
            float s9 = rp.hasDepth ? simd_sum(gdep) : 0.0f;
            float a0 = 0, a1 = 0;
            if (rp.absGrad) { a0 = simd_sum(abs(gm0)); a1 = simd_sum(abs(gm1)); }
            if (lane == 0) {
                device atomic<float>* A = acc + sI[j] * 12;
                atomic_fetch_add_explicit(A + 0, s0, memory_order_relaxed);
                atomic_fetch_add_explicit(A + 1, s1, memory_order_relaxed);
                atomic_fetch_add_explicit(A + 2, s2, memory_order_relaxed);
                atomic_fetch_add_explicit(A + 3, s3, memory_order_relaxed);
                atomic_fetch_add_explicit(A + 4, s4, memory_order_relaxed);
                atomic_fetch_add_explicit(A + 5, s5, memory_order_relaxed);
                atomic_fetch_add_explicit(A + 6, s6, memory_order_relaxed);
                atomic_fetch_add_explicit(A + 7, s7, memory_order_relaxed);
                atomic_fetch_add_explicit(A + 8, s8, memory_order_relaxed);
                if (rp.hasDepth) atomic_fetch_add_explicit(A + 9, s9, memory_order_relaxed);
                if (rp.absGrad) {
                    atomic_fetch_add_explicit(A + 10, a0, memory_order_relaxed);
                    atomic_fetch_add_explicit(A + 11, a1, memory_order_relaxed);
                }
            }
        }
    }
}

// ---------------------------------------------------------------- loss: L1 + D-SSIM + depth
// Planes are [plane][H][W]. Gaussian window 11x11, sigma 1.5, zero padding (matches the
// reference PyTorch SSIM used by 3DGS).
kernel void ssim_prep(device const float* img [[buffer(0)]], device const uchar4* target [[buffer(1)]],
                      device float* planes [[buffer(2)]], constant RasterParams& rp [[buffer(3)]],
                      uint2 p [[thread_position_in_grid]]) {
    if (p.x >= rp.width || p.y >= rp.height) return;
    uint hw = rp.width * rp.height, pid = p.y * rp.width + p.x;
    uchar4 t = target[pid];
    bool keep = t.w > 127;
    for (uint c = 0; c < 3; c++) {
        float x = img[c*hw + pid];
        float y = keep ? float(t[c]) / 255.0f : x;   // masked-out pixels: target := render
        planes[(c*5 + 0)*hw + pid] = x;
        planes[(c*5 + 1)*hw + pid] = y;
        planes[(c*5 + 2)*hw + pid] = x * x;
        planes[(c*5 + 3)*hw + pid] = y * y;
        planes[(c*5 + 4)*hw + pid] = x * y;
    }
}

kernel void blur_h(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]],
                   constant RasterParams& rp [[buffer(2)]], constant float* w [[buffer(3)]],
                   uint3 p [[thread_position_in_grid]]) {
    if (p.x >= rp.width || p.y >= rp.height) return;
    uint hw = rp.width * rp.height;
    uint base = p.z * hw + p.y * rp.width;
    float s = 0;
    for (int k = -5; k <= 5; k++) {
        int x = int(p.x) + k;
        if (x >= 0 && x < int(rp.width)) s += w[k + 5] * src[base + uint(x)];
    }
    dst[base + p.x] = s;
}

kernel void blur_v(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]],
                   constant RasterParams& rp [[buffer(2)]], constant float* w [[buffer(3)]],
                   uint3 p [[thread_position_in_grid]]) {
    if (p.x >= rp.width || p.y >= rp.height) return;
    uint hw = rp.width * rp.height;
    uint base = p.z * hw;
    float s = 0;
    for (int k = -5; k <= 5; k++) {
        int y = int(p.y) + k;
        if (y >= 0 && y < int(rp.height)) s += w[k + 5] * src[base + uint(y) * rp.width + p.x];
    }
    dst[base + p.y * rp.width + p.x] = s;
}

// From blurred moments compute the SSIM map and the three derivative maps per channel.
// stats[0] += sum |x-y|, stats[1] += sum ssim, stats[2] += sum (x-y)^2
kernel void ssim_terms(device const float* mu [[buffer(0)]], device float* dmaps [[buffer(1)]],
                       device atomic<float>* stats [[buffer(2)]], constant RasterParams& rp [[buffer(3)]],
                       device const float* planes [[buffer(4)]],
                       uint2 p [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    bool inb = p.x < rp.width && p.y < rp.height;
    uint hw = rp.width * rp.height, pid = p.y * rp.width + p.x;
    const float C1 = 0.0001f, C2 = 0.0009f;
    float sumS = 0, sumL1 = 0, sumL2 = 0;
    if (inb) {
        for (uint c = 0; c < 3; c++) {
            float mx = mu[(c*5+0)*hw + pid], my = mu[(c*5+1)*hw + pid];
            float mxx = mu[(c*5+2)*hw + pid], myy = mu[(c*5+3)*hw + pid], mxy = mu[(c*5+4)*hw + pid];
            float sx = mxx - mx*mx, sy = myy - my*my, sxy = mxy - mx*my;
            float A1 = 2*mx*my + C1, A2 = 2*sxy + C2, B1 = mx*mx + my*my + C1, B2 = sx + sy + C2;
            float Q = 1.0f / (B1 * B2);
            float S = A1 * A2 * Q;
            // S/A1 and S/A2 written without dividing by A1, A2 (either can be exactly 0 → 0/0 = NaN).
            dmaps[(c*3+0)*hw + pid] = 2*my*A2*Q - 2*my*A1*Q - 2*mx*S/B1 + 2*mx*S/B2;
            dmaps[(c*3+1)*hw + pid] = -S / B2;
            dmaps[(c*3+2)*hw + pid] = 2*A1*Q;
            sumS += S;
            float x = planes[(c*5)*hw + pid], y = planes[(c*5+1)*hw + pid];
            sumL1 += abs(x - y); sumL2 += (x - y) * (x - y);
        }
    }
    sumS = simd_sum(sumS); sumL1 = simd_sum(sumL1); sumL2 = simd_sum(sumL2);
    if (lane == 0) {
        atomic_fetch_add_explicit(stats + 0, sumL1, memory_order_relaxed);
        atomic_fetch_add_explicit(stats + 1, sumS, memory_order_relaxed);
        atomic_fetch_add_explicit(stats + 2, sumL2, memory_order_relaxed);
    }
}

kernel void loss_grad(device const float* planes [[buffer(0)]], device const float* bd [[buffer(1)]],
                      device const float* img [[buffer(2)]], device float* gradImg [[buffer(3)]],
                      device const float* depthTarget [[buffer(4)]], constant uint2& depthSize [[buffer(5)]],
                      constant RasterParams& rp [[buffer(6)]], device const uchar4* target [[buffer(7)]],
                      device atomic<float>* stats [[buffer(8)]],
                      uint2 p [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    bool inb = p.x < rp.width && p.y < rp.height;
    float dl = 0;
    if (inb) {
        uint hw = rp.width * rp.height, pid = p.y * rp.width + p.x;
        float norm = rp.invNumPixels / 3.0f;
        float lam = rp.ssimWeight;
        for (uint c = 0; c < 3; c++) {
            float x = planes[(c*5)*hw + pid], y = planes[(c*5+1)*hw + pid];
            float dS = bd[(c*3+0)*hw + pid] + 2.0f * x * bd[(c*3+1)*hw + pid] + y * bd[(c*3+2)*hw + pid];
            float l1g = x > y ? 1.0f : (x < y ? -1.0f : 0.0f);
            gradImg[c*hw + pid] = norm * ((1.0f - lam) * l1g - lam * dS);
        }
        float g = 0;
        if (rp.hasDepth && target[pid].w > 127) {
            uint dx = min(depthSize.x - 1, p.x * depthSize.x / rp.width);
            uint dy = min(depthSize.y - 1, p.y * depthSize.y / rp.height);
            float dt = depthTarget[dy * depthSize.x + dx];
            if (dt > 0.0f && isfinite(dt)) {
                float d = img[3*hw + pid];
                // Edge-aware (DN-Splatter): LiDAR is least reliable at depth / colour edges, so its pull is
                // weighted by exp(-|∇I|) of the photo there (rp.pad = 1 enables it).
                float ew = 1.0f;
                if (rp.pad > 0.5f) {
                    uint xl = p.x > 0 ? pid - 1 : pid, xr = p.x + 1 < rp.width ? pid + 1 : pid;
                    uint yu = p.y > 0 ? pid - rp.width : pid, yd = p.y + 1 < rp.height ? pid + rp.width : pid;
                    float4 tl = float4(target[xl]), tr = float4(target[xr]), tu = float4(target[yu]), td = float4(target[yd]);
                    float gx = dot(tr.xyz - tl.xyz, float3(0.299f, 0.587f, 0.114f)) / 510.0f;
                    float gy = dot(td.xyz - tu.xyz, float3(0.299f, 0.587f, 0.114f)) / 510.0f;
                    ew = exp(-10.0f * sqrt(gx * gx + gy * gy));
                }
                g = ew * rp.depthWeight * rp.invNumPixels * (d > dt ? 1.0f : -1.0f);
                dl = abs(d - dt);
            }
        }
        gradImg[3*hw + pid] = g;
    }
    dl = simd_sum(dl);
    if (lane == 0) atomic_fetch_add_explicit(stats + 3, dl, memory_order_relaxed);
}

// ---------------------------------------------------------------- per-photo appearance (exposure / white balance)
// corrected = M · rendered + b, with M (3×3, row-major) and b (3) learned per training photo, so differing
// auto-exposure / white balance between photos doesn't get baked into the scene as haze or colour noise.
kernel void appearance_apply(device float* img [[buffer(0)]], device float* raw [[buffer(1)]],
                             device const float* ap [[buffer(2)]], constant RasterParams& rp [[buffer(3)]],
                             uint2 p [[thread_position_in_grid]]) {
    if (p.x >= rp.width || p.y >= rp.height) return;
    uint hw = rp.width * rp.height, pid = p.y * rp.width + p.x;
    float3 x = float3(img[pid], img[hw + pid], img[2*hw + pid]);
    raw[pid] = x.x; raw[hw + pid] = x.y; raw[2*hw + pid] = x.z;
    for (uint r = 0; r < 3; r++)
        img[r*hw + pid] = ap[r*3] * x.x + ap[r*3+1] * x.y + ap[r*3+2] * x.z + ap[9+r];
}

kernel void appearance_backward(device float* gradImg [[buffer(0)]], device const float* raw [[buffer(1)]],
                                device const float* ap [[buffer(2)]], constant RasterParams& rp [[buffer(3)]],
                                device atomic<float>* apGrad [[buffer(4)]],
                                uint2 p [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    bool inb = p.x < rp.width && p.y < rp.height;
    float3 g = 0, x = 0;
    if (inb) {
        uint hw = rp.width * rp.height, pid = p.y * rp.width + p.x;
        g = float3(gradImg[pid], gradImg[hw + pid], gradImg[2*hw + pid]);
        x = float3(raw[pid], raw[hw + pid], raw[2*hw + pid]);
        // d/d(raw) = Mᵀ g
        for (uint c = 0; c < 3; c++) gradImg[c*hw + pid] = ap[c] * g.x + ap[3+c] * g.y + ap[6+c] * g.z;
    }
    float v[12];
    for (uint r = 0; r < 3; r++) { v[r*3] = g[r] * x.x; v[r*3+1] = g[r] * x.y; v[r*3+2] = g[r] * x.z; v[9+r] = g[r]; }
    for (uint k = 0; k < 12; k++) {
        float s = simd_sum(v[k]);
        if (lane == 0) atomic_fetch_add_explicit(apGrad + k, s, memory_order_relaxed);
    }
}

// ---------------------------------------------------------------- transient (moving-object) robustness
// RobustNeRF / SpotLessSplats style: pixels whose error is far above what the rest of the image
// achieves, *in spatially coherent regions*, are treated as transient and removed from the loss.

kernel void residual_hist(device const float* planes [[buffer(0)]], device const uchar4* target [[buffer(1)]],
                          device float* resid [[buffer(2)]], device atomic_uint* hist [[buffer(3)]],
                          constant RasterParams& rp [[buffer(4)]], device const float* finalT [[buffer(5)]],
                          uint2 p [[thread_position_in_grid]]) {
    if (p.x >= rp.width || p.y >= rp.height) return;
    uint hw = rp.width * rp.height, pid = p.y * rp.width + p.x;
    if (target[pid].w <= 127) { resid[pid] = -1.0f; return; }      // dataset-masked: ignore
    float r = 0;
    for (uint c = 0; c < 3; c++) r += abs(planes[(c*5)*hw + pid] - planes[(c*5+1)*hw + pid]);
    r /= 3.0f;
    // Only well-covered pixels may be called transient: a moving object hides reconstructed
    // background, whereas a high error where the scene is still thin means "not learned yet"
    // (masking those would stop them ever being learned). -2 = exempt but counted in the histogram.
    resid[pid] = finalT[pid] > 0.05f ? -2.0f : r;
    uint bin = min(255u, uint(r * 512.0f));                        // 0 … 0.5 in 256 bins
    atomic_fetch_add_explicit(&hist[bin], 1u, memory_order_relaxed);
}

// One threadgroup per 16x16 tile (+1 px halo for the 3x3 smoothing).
kernel void transient_weights(device const float* resid [[buffer(0)]], device float* weight [[buffer(1)]],
                              constant RasterParams& rp [[buffer(2)]], constant float& threshold [[buffer(3)]],
                              device atomic_uint* outlierCount [[buffer(4)]],
                              uint2 tg [[threadgroup_position_in_grid]], uint2 lt [[thread_position_in_threadgroup]],
                              uint tidx [[thread_index_in_threadgroup]]) {
    threadgroup float inl[18][18];
    threadgroup atomic_uint tileInliers;
    if (tidx == 0) atomic_store_explicit(&tileInliers, 0u, memory_order_relaxed);
    // load 18x18 halo region
    for (uint i = tidx; i < 18 * 18; i += 256) {
        int lx = int(i % 18) - 1, ly = int(i / 18) - 1;
        int x = int(tg.x * 16) + lx, y = int(tg.y * 16) + ly;
        float v = 1.0f;
        if (x >= 0 && y >= 0 && x < int(rp.width) && y < int(rp.height)) {
            float r = resid[uint(y) * rp.width + uint(x)];
            v = (r < 0.0f || r <= threshold) ? 1.0f : 0.0f;   // -1 masked, -2 thin coverage: inlier
        }
        inl[i / 18][i % 18] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0;
    for (int dy = 0; dy < 3; dy++) for (int dx = 0; dx < 3; dx++) s += inl[lt.y + dy][lt.x + dx];
    // Outlier only inside *solid* high-error regions (≥ 7 of 9 neighbours): moving objects are solid
    // blobs, whereas not-yet-converged texture produces thin, broken error patterns.
    // (Tested: a looser 5-of-9 rule without the patch vote below masked 12-16 % of pixels on textured
    // scenes and lost 7 dB. This rule leaks ~1 px at object edges but never harmed quality.)
    bool smoothInlier = s > 2.5f;
    uint2 pix = tg * 16 + lt;
    bool inside = pix.x < rp.width && pix.y < rp.height;
    if (inside && smoothInlier) atomic_fetch_add_explicit(&tileInliers, 1u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!inside) return;
    uint tw = min(16u, rp.width - tg.x * 16), th = min(16u, rp.height - tg.y * 16);
    float frac = float(atomic_load_explicit(&tileInliers, memory_order_relaxed)) / float(tw * th);
    // Patch vote: outliers inside a mostly-good 16x16 patch are hard detail, not transients.
    bool keep = smoothInlier || frac >= 0.6f;
    weight[pix.y * rp.width + pix.x] = keep ? 1.0f : 0.0f;
    if (!keep) atomic_fetch_add_explicit(outlierCount, 1u, memory_order_relaxed);
}

kernel void apply_weights(device float* gradImg [[buffer(0)]], device const float* weight [[buffer(1)]],
                          constant RasterParams& rp [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
    if (p.x >= rp.width || p.y >= rp.height) return;
    uint hw = rp.width * rp.height, pid = p.y * rp.width + p.x;
    float w = weight[pid];
    if (w == 1.0f) return;
    for (uint c = 0; c < 4; c++) gradImg[c*hw + pid] *= w;
}

// ---------------------------------------------------------------- preprocess (backward)
kernel void preprocess_backward(device const float* pos [[buffer(0)]],
                                device const float* logScale [[buffer(1)]],
                                device const float* quat [[buffer(2)]],
                                device const float* opLogit [[buffer(3)]],
                                device const float* sh [[buffer(4)]],
                                constant CameraUniforms& cam [[buffer(5)]],
                                constant RasterParams& rp [[buffer(6)]],
                                device const float* acc [[buffer(7)]],
                                device const uint* tilesTouched [[buffer(8)]],
                                device float* gPos [[buffer(9)]],
                                device float* gScale [[buffer(10)]],
                                device float* gQuat [[buffer(11)]],
                                device float* gOp [[buffer(12)]],
                                device float* gSH [[buffer(13)]],
                                device float* gradStats [[buffer(14)]],  // per gaussian: accum, count, maxRadius
                                constant float4& reg [[buffer(15)]],     // x: opacityReg/N, y: scaleReg/(3N), z: anisotropyReg/N, w: max axis ratio
                                constant float2& big [[buffer(16)]],     // x: bigSplatReg/N, y: largest allowed tan(half-angle)
                                uint i [[thread_position_in_grid]]) {
    if (i >= cam.count) return;
    float o = 1.0f / (1.0f + exp(-opLogit[i]));
    float3 s = exp(float3(logScale[i*3], logScale[i*3+1], logScale[i*3+2]));
    // Regulariser gradients apply to every gaussian.
    float gop = reg.x * o * (1.0f - o);
    float3 gls = reg.y * s;
    // Needle penalty: r = longest / MIDDLE axis. Flat discs (two long axes, one tiny) are the normal, desirable
    // shape for surfaces, so the shortest axis must not count — using longest / shortest squashed every disc
    // into faint thin strokes ("painted" splats, 70 % of them nearly transparent). Only the long axis shrinks.
    if (reg.z > 0.0f) {
        uint ia = s.x >= s.y ? (s.x >= s.z ? 0u : 2u) : (s.y >= s.z ? 1u : 2u);
        uint ib = s.x <= s.y ? (s.x <= s.z ? 0u : 2u) : (s.y <= s.z ? 1u : 2u);
        if (ia != ib) {   // all three equal → not a needle (and ia == ib would make im out of range)
            uint im = 3u - ia - ib;
            float r = s[ia] / max(s[im], 1e-12f);
            if (r > reg.w) { gls[ia] += reg.z * r; }
        }
    }
    float3 gp = float3(0);
    float4 gq = float4(0);
    bool visible = tilesTouched[i] > 0;
    int deg = int(cam.shDegree);
    for (int k = 0; k < 48; k++) gSH[i*48 + k] = 0.0f;
    if (visible) {
        device const float* A = acc + i * 12;
        float2 dmean = float2(A[0], A[1]);
        float3 dconic = float3(A[2], A[3], A[4]);
        float dop = A[5];
        float3 dcol = float3(A[6], A[7], A[8]);
        float ddep = A[9];

        float3 p = float3(pos[i*3], pos[i*3+1], pos[i*3+2]);
        float4 qr = float4(quat[i*4], quat[i*4+1], quat[i*4+2], quat[i*4+3]);
        float qlen = max(length(qr), 1e-12f);
        float4 q = qr / qlen;
        float3x3 R = quat_to_rot(q);
        // Row-major helpers: Rm[r][c]
        float Rm[3][3], Mm[3][3], Sg[3][3], Wm[3][3];
        for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) {
            Rm[r][c] = R[c][r];
            Wm[r][c] = cam.worldToCam[c][r];
        }
        for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) Mm[r][c] = Rm[r][c] * s[c];
        for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) {
            float v = 0; for (int k = 0; k < 3; k++) v += Mm[r][k] * Mm[c][k];
            Sg[r][c] = v;
        }
        float3 t = (cam.worldToCam * float4(p, 1.0f)).xyz;
        float fx = cam.intr.x, fy = cam.intr.y;
        float limx = cam.viewport.z, limy = cam.viewport.w;
        float rx = t.x / t.z, ry = t.y / t.z;
        float txz = clamp(rx, -limx, limx) * t.z, tyz = clamp(ry, -limy, limy) * t.z;
        float mulx = (rx < -limx || rx > limx) ? 0.0f : 1.0f;
        float muly = (ry < -limy || ry > limy) ? 0.0f : 1.0f;
        float iz = 1.0f / t.z, iz2 = iz * iz, iz3 = iz2 * iz;
        float Jm[2][3] = { { fx * iz, 0, -fx * txz * iz2 }, { 0, fy * iz, -fy * tyz * iz2 } };
        float Tm[2][3];
        for (int r = 0; r < 2; r++) for (int c = 0; c < 3; c++) {
            float v = 0; for (int k = 0; k < 3; k++) v += Jm[r][k] * Wm[k][c];
            Tm[r][c] = v;
        }
        float TS[2][3];
        for (int r = 0; r < 2; r++) for (int c = 0; c < 3; c++) {
            float v = 0; for (int k = 0; k < 3; k++) v += Tm[r][k] * Sg[k][c];
            TS[r][c] = v;
        }
        float c00 = 0, c01 = 0, c11 = 0;
        for (int k = 0; k < 3; k++) { c00 += TS[0][k] * Tm[0][k]; c01 += TS[0][k] * Tm[1][k]; c11 += TS[1][k] * Tm[1][k]; }
        float dil = cam.misc.z;
        float a = c00 + dil, b = c01, c = c11 + dil;
        float det = a * c - b * b;
        float det0 = c00 * c11 - c01 * c01;
        float idet = 1.0f / det;
        float Ca = c * idet, Cb = -b * idet, Cc = a * idet;   // conic
        bool aa = (cam.flags & 1) != 0;
        float comp = aa ? sqrt(max(det0, 0.0f) * idet) : 1.0f;
        // opacity
        gop += dop * comp * o * (1.0f - o);
        // conic -> covariance (with dilation)
        float G00 = dconic.x, G01 = 0.5f * dconic.y, G11 = dconic.z;
        // dSigma' = -Cinv * G * Cinv
        float P00 = Ca * G00 + Cb * G01, P01 = Ca * G01 + Cb * G11;
        float P10 = Cb * G00 + Cc * G01, P11 = Cb * G01 + Cc * G11;
        float D00 = -(P00 * Ca + P01 * Cb);
        float D01 = -(P00 * Cb + P01 * Cc);
        float D11 = -(P10 * Cb + P11 * Cc);
        float dc00 = D00, dc11 = D11, dc01 = 2.0f * D01;     // scalar (tied) off-diagonal
        if (aa && comp > 1e-8f) {
            float dLdcomp = dop * o;
            float dcdr = 0.5f / comp;
            float idet2 = idet * idet;
            float ddet0_00 = c11, ddet0_11 = c00, ddet0_01 = -2.0f * c01;
            float ddet_00 = c, ddet_11 = a, ddet_01 = -2.0f * b;
            float k = dLdcomp * dcdr;
            dc00 += k * (ddet0_00 * det - det0 * ddet_00) * idet2;
            dc11 += k * (ddet0_11 * det - det0 * ddet_11) * idet2;
            dc01 += k * (ddet0_01 * det - det0 * ddet_01) * idet2;
        }
        // full-matrix gradient of the 2x2 covariance
        float G2[2][2] = { { dc00, 0.5f * dc01 }, { 0.5f * dc01, dc11 } };
        // dSigma3 = T^T G2 T
        float dS3[3][3];
        for (int r = 0; r < 3; r++) for (int cc = 0; cc < 3; cc++) {
            float v = 0;
            for (int u = 0; u < 2; u++) for (int w = 0; w < 2; w++) v += Tm[u][r] * G2[u][w] * Tm[w][cc];
            dS3[r][cc] = v;
        }
        // dT = 2 G2 T Sigma  (2x3)
        float dT[2][3];
        for (int r = 0; r < 2; r++) for (int cc = 0; cc < 3; cc++) {
            dT[r][cc] = 2.0f * (G2[r][0] * TS[0][cc] + G2[r][1] * TS[1][cc]);
        }
        // dJ = dT W^T
        float dJ[2][3];
        for (int r = 0; r < 2; r++) for (int cc = 0; cc < 3; cc++) {
            float v = 0; for (int k = 0; k < 3; k++) v += dT[r][k] * Wm[cc][k];
            dJ[r][cc] = v;
        }
        // Big-splat penalty (only for splats this view draws). The splat's visible radius is where its
        // alpha falls to 1/255: rho = sigma_max * sqrt(2 ln(255 o)). Seen from distance d, it spans a cone of
        // half-angle asin(rho/d); tan(half-angle/2)-style ratio rho / (d + sqrt(d^2 - rho^2)) stays finite when
        // the camera is inside the splat (d <= rho gives 1). Above the limit, log-scale gradients push the
        // splat smaller in proportion to how far over it is (log2), the longest axis fully, the others half.
        if (big.x > 0.0f) {
            float smax = max(s.x, max(s.y, s.z));
            float lnA = log(max(255.0f * o, 1.0f));
            if (lnA > 0.0f) {
                float rho = smax * sqrt(2.0f * lnA);
                float d = length(t);
                float rr = rho / (max(d, rho) + sqrt(max(d * d - rho * rho, 0.0f)));
                if (rr > big.y) {
                    float g = big.x * log2(rr / big.y);
                    for (int k = 0; k < 3; k++) gls[k] += g * (s[k] >= smax ? 1.0f : 0.5f);
                }
            }
        }
        float3 dt = float3(0);
        dt.x += -fx * iz2 * dJ[0][2] * mulx;
        dt.y += -fy * iz2 * dJ[1][2] * muly;
        dt.z += -fx * iz2 * dJ[0][0] + 2.0f * fx * txz * iz3 * dJ[0][2] - fy * iz2 * dJ[1][1] + 2.0f * fy * tyz * iz3 * dJ[1][2];
        // mean2D
        dt.x += fx * iz * dmean.x;
        dt.y += fy * iz * dmean.y;
        dt.z += -fx * t.x * iz2 * dmean.x - fy * t.y * iz2 * dmean.y;
        // depth
        dt.z += ddep;
        // position: t = W p + tvec  ->  dp = W^T dt
        for (int k = 0; k < 3; k++) gp[k] = Wm[0][k] * dt.x + Wm[1][k] * dt.y + Wm[2][k] * dt.z;
        // Sigma3 = M M^T -> dM = 2 dS3 M
        float dM[3][3];
        for (int r = 0; r < 3; r++) for (int cc = 0; cc < 3; cc++) {
            float v = 0; for (int k = 0; k < 3; k++) v += (dS3[r][k] + dS3[k][r]) * Mm[k][cc];
            dM[r][cc] = v;
        }
        float dR[3][3];
        float3 ds = float3(0);
        for (int r = 0; r < 3; r++) for (int cc = 0; cc < 3; cc++) {
            dR[r][cc] = dM[r][cc] * s[cc];
            ds[cc] += dM[r][cc] * Rm[r][cc];
        }
        gls += ds * s;
        float w = q.x, x = q.y, y = q.z, z = q.w;
        float gw = 2.0f * (z * (dR[1][0] - dR[0][1]) + y * (dR[0][2] - dR[2][0]) + x * (dR[2][1] - dR[1][2]));
        float gx = 2.0f * (y * (dR[1][0] + dR[0][1]) + z * (dR[2][0] + dR[0][2]) + w * (dR[2][1] - dR[1][2])) - 4.0f * x * (dR[1][1] + dR[2][2]);
        float gy = 2.0f * (x * (dR[1][0] + dR[0][1]) + w * (dR[0][2] - dR[2][0]) + z * (dR[2][1] + dR[1][2])) - 4.0f * y * (dR[0][0] + dR[2][2]);
        float gz = 2.0f * (w * (dR[1][0] - dR[0][1]) + x * (dR[2][0] + dR[0][2]) + y * (dR[2][1] + dR[1][2])) - 4.0f * z * (dR[0][0] + dR[1][1]);
        float4 gqn = float4(gw, gx, gy, gz);
        gq = (gqn - q * dot(q, gqn)) / qlen;
        // SH colour
        float bsh[16];
        float3 dir = normalize(p - cam.camPos.xyz);
        sh_basis(dir, deg, bsh);
        float3 raw = float3(0);
        int nc = (deg + 1) * (deg + 1);
        for (int k = 0; k < nc; k++) raw += bsh[k] * float3(sh[i*48 + k*3], sh[i*48 + k*3 + 1], sh[i*48 + k*3 + 2]);
        raw += 0.5f;
        float3 dc = float3(raw.x > 0 ? dcol.x : 0, raw.y > 0 ? dcol.y : 0, raw.z > 0 ? dcol.z : 0);
        for (int k = 0; k < nc; k++) {
            gSH[i*48 + k*3] = bsh[k] * dc.x; gSH[i*48 + k*3 + 1] = bsh[k] * dc.y; gSH[i*48 + k*3 + 2] = bsh[k] * dc.z;
        }
        // densification statistics (screen-space gradient in NDC units, like 3DGS)
        float2 g2 = rp.absGrad ? float2(A[10], A[11]) : dmean;
        float gn = length(g2 * float2(0.5f * float(rp.width), 0.5f * float(rp.height)));
        gradStats[i*3] += gn;
        gradStats[i*3 + 1] += 1.0f;
        float mid = 0.5f * (a + c);
        float l1 = mid + sqrt(max(0.1f, mid * mid - det));
        gradStats[i*3 + 2] = max(gradStats[i*3 + 2], 3.0f * sqrt(l1) / float(max(rp.width, rp.height)));
    }
    gPos[i*3] = gp.x; gPos[i*3+1] = gp.y; gPos[i*3+2] = gp.z;
    gScale[i*3] = gls.x; gScale[i*3+1] = gls.y; gScale[i*3+2] = gls.z;
    gQuat[i*4] = gq.x; gQuat[i*4+1] = gq.y; gQuat[i*4+2] = gq.z; gQuat[i*4+3] = gq.w;
    gOp[i] = gop;
}

// ---------------------------------------------------------------- optimiser
struct AdamParams { float lr; float lr2; uint period; uint split; float beta1; float beta2; float eps; float bc1; float bc2; uint n; };

kernel void adam_step(device float* param [[buffer(0)]], device const float* grad [[buffer(1)]],
                      device float* m [[buffer(2)]], device float* v [[buffer(3)]],
                      constant AdamParams& ap [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= ap.n) return;
    float g = grad[i];
    if (!isfinite(g)) g = 0.0f;
    float mi = ap.beta1 * m[i] + (1.0f - ap.beta1) * g;
    float vi = ap.beta2 * v[i] + (1.0f - ap.beta2) * g * g;
    m[i] = mi; v[i] = vi;
    float lr = (i % ap.period) < ap.split ? ap.lr : ap.lr2;
    param[i] -= lr * (mi / ap.bc1) / (sqrt(vi / ap.bc2) + ap.eps);
}

inline uint pcg(uint v) { uint s = v * 747796405u + 2891336453u; uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }
inline float rnd(thread uint& s) { s = pcg(s); return (float(s) + 0.5f) / 4294967296.0f; }
inline float randn(thread uint& s) { float u1 = rnd(s), u2 = rnd(s); return sqrt(-2.0f * log(u1)) * cos(6.2831853f * u2); }

// 3DGS-MCMC: perturb positions of low-opacity gaussians along their covariance (SGLD noise).
kernel void mcmc_noise(device float* pos [[buffer(0)]], device const float* logScale [[buffer(1)]],
                       device const float* quat [[buffer(2)]], device const float* opLogit [[buffer(3)]],
                       constant float4& np [[buffer(4)]],  // x: noiseLR * lrPos, y: seed, z: n
                       uint i [[thread_position_in_grid]]) {
    if (i >= uint(np.z)) return;
    float o = 1.0f / (1.0f + exp(-opLogit[i]));
    float gate = 1.0f / (1.0f + exp(-100.0f * ((1.0f - o) - 0.995f)));
    float k = gate * np.x;
    if (k < 1e-12f) return;
    uint seed = pcg(i * 9781u + as_type<uint>(np.y));
    float3 e = float3(randn(seed), randn(seed), randn(seed));
    float3 s = exp(float3(logScale[i*3], logScale[i*3+1], logScale[i*3+2]));
    float4 q = float4(quat[i*4], quat[i*4+1], quat[i*4+2], quat[i*4+3]);
    float ql = length(q); q = ql > 1e-12f ? q / ql : float4(1,0,0,0);
    float3x3 R = quat_to_rot(q);
    float3x3 M = float3x3(R[0] * s.x, R[1] * s.y, R[2] * s.z);
    float3 d = (M * transpose(M)) * e * k;
    pos[i*3] += d.x; pos[i*3+1] += d.y; pos[i*3+2] += d.z;
}

kernel void zero_float(device float* b [[buffer(0)]], constant uint& n [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i < n) b[i] = 0.0f;
}

// Converts a uint8 RGBA target + float render into a preview RGBA8 image (for the UI).
kernel void render_to_rgba8(device const float* img [[buffer(0)]], device uchar4* out [[buffer(1)]],
                            constant RasterParams& rp [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
    if (p.x >= rp.width || p.y >= rp.height) return;
    uint hw = rp.width * rp.height, pid = p.y * rp.width + p.x;
    float3 c = clamp(float3(img[pid], img[hw + pid], img[2*hw + pid]), 0.0f, 1.0f);
    out[pid] = uchar4(uchar3(c * 255.0f + 0.5f), 255);
}
"""#
}
