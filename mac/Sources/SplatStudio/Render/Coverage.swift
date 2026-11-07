import Foundation
import Metal
import simd

/// "How well did the capture see this?" for every splat — the Coverage display mode.
///
/// For each splat, every dataset camera is checked: is the splat inside its image, and (when the camera has a
/// LiDAR depth map) not hidden behind a nearer surface? That gives the number of photos that saw it and the
/// best detail any of them had (pixels per scene unit at that distance). The score combines both, relative to
/// this capture — "seen by only one or two photos" or "seen from 4× further away than typical" is poor —
/// so it works for a bedroom in metres and an SfM scene in arbitrary units alike.
extension ShaderSource {
    static let coverage = #"""
struct CovCam { float4 r0; float4 r1; float4 r2; float4 intr; uint4 dims; };   // w2c rows, fx fy cx cy, w h dw dh

kernel void coverage_compute(device const float4* gpos [[buffer(0)]],
                             device const CovCam* cams [[buffer(1)]],
                             device const half* depth [[buffer(2)]],
                             device const uint* depthOff [[buffer(3)]],
                             constant uint& ncam [[buffer(4)]],
                             constant uint& n [[buffer(5)]],
                             device float2* out [[buffer(6)]],
                             uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float3 p = gpos[i].xyz;
    float cnt = 0, best = 0;
    for (uint c = 0; c < ncam; c++) {
        CovCam k = cams[c];
        float z = dot(k.r2.xyz, p) + k.r2.w;
        if (z < 1e-3f) continue;
        float x = dot(k.r0.xyz, p) + k.r0.w, y = dot(k.r1.xyz, p) + k.r1.w;
        float u = k.intr.x * x / z + k.intr.z, v = k.intr.y * y / z + k.intr.w;
        if (u < 0 || v < 0 || u >= float(k.dims.x) || v >= float(k.dims.y)) continue;
        uint off = depthOff[c];
        if (off != 0xFFFFFFFFu && k.dims.z > 0) {
            uint du = min(uint(u * float(k.dims.z) / float(k.dims.x)), k.dims.z - 1);
            uint dv = min(uint(v * float(k.dims.w) / float(k.dims.y)), k.dims.w - 1);
            float d = float(depth[off + dv * k.dims.z + du]);
            if (d > 0 && z > d * 1.1f + 0.05f) continue;   // behind the surface this camera saw
        }
        cnt += 1;
        best = max(best, k.intr.x / z);
    }
    out[i] = float2(cnt, best);
}
"""#
}

enum Coverage {
    struct Result {
        let score: MTLBuffer          // float per splat, 0 = poor … 1 = good
        let poorFraction: Double      // share of the scene (splat size × opacity) seen, but poorly (score < 0.34)
        let unseenFraction: Double    // share no photo saw at all (usually floaters beyond the walls)
        let medianViews: Int
        let usedDepth: Bool
        /// Photos that saw each splat (0 = none: behind every camera's depth surface or outside every frame).
        let views: [UInt16]
    }

    /// What "Auto clean" removes: splats no photo saw, and splats outside the box the photos covered well.
    struct CleanPlan {
        /// Splats to remove: seen by no photo, or outside the box and seen by fewer than 3 photos (well-seen
        /// geometry at the edges of a room is always kept).
        var remove: [Bool]
        var boxMin: SIMD3<Float>
        var boxMax: SIMD3<Float>
        var unseenCount: Int
        var outsideCount: Int
    }

    /// Box = 1st–99th percentile, per axis, of reasonably opaque splats seen by ≥ 3 photos, grown by 5 % of its
    /// size. Proposed for removal: splats seen by no photo, and splats outside the box that fewer than 3 photos
    /// saw. (Cutting everything outside the box removed real walls at the room's edges.)
    static func cleanPlan(scene: SplatScene, result r: Result, minViewsOutside: Int = 3) -> CleanPlan {
        let n = scene.count
        let pos = scene.gpos.contents().bindMemory(to: SIMD4<Float>.self, capacity: n)
        var xs: [Float] = [], ys: [Float] = [], zs: [Float] = []
        for i in stride(from: 0, to: n, by: max(1, n / 400_000)) where r.views[i] >= 3 && pos[i].w > 0.3 {
            xs.append(pos[i].x); ys.append(pos[i].y); zs.append(pos[i].z)
        }
        func pct(_ a: inout [Float], _ q: Double) -> Float { a.isEmpty ? 0 : a[min(a.count - 1, Int(Double(a.count - 1) * q))] }
        xs.sort(); ys.sort(); zs.sort()
        var lo = SIMD3(pct(&xs, 0.01), pct(&ys, 0.01), pct(&zs, 0.01))
        var hi = SIMD3(pct(&xs, 0.99), pct(&ys, 0.99), pct(&zs, 0.99))
        let pad = (hi - lo) * 0.05
        lo -= pad; hi += pad
        var remove = [Bool](repeating: false, count: n)
        var u = 0, o = 0
        for i in 0..<n {
            let q = SIMD3(pos[i].x, pos[i].y, pos[i].z)
            if r.views[i] == 0 { remove[i] = true; u += 1 }
            else if Int(r.views[i]) < minViewsOutside, !xs.isEmpty, any(q .< lo) || any(q .> hi) { remove[i] = true; o += 1 }
        }
        return CleanPlan(remove: remove, boxMin: lo, boxMax: hi, unseenCount: u, outsideCount: o)
    }

    private struct CovCam {
        var r0: SIMD4<Float>, r1: SIMD4<Float>, r2: SIMD4<Float>, intr: SIMD4<Float>, dims: SIMD4<UInt32>
    }

    /// Views at or below this count score 0; at or above `goodViews` they score 1.
    static let poorViews: Float = 2, goodViews: Float = 8
    /// Detail relative to the median: ¼ of typical (seen from 4× further) scores 0, typical or better scores 1.
    static let poorDetailRatio: Float = 0.25

    static func compute(scene: SplatScene, dataset: PreparedDataset, ctx: MetalContext = .shared) throws -> Result {
        let cams = dataset.manifest.cameras
        var gpuCams: [CovCam] = []
        var offsets: [UInt32] = []
        var depthHalves = Data()
        for c in cams {
            let w2c = simd_inverse(c.c2w)
            func row(_ r: Int) -> SIMD4<Float> { SIMD4(w2c[0][r], w2c[1][r], w2c[2][r], w2c[3][r]) }
            var dims = SIMD4<UInt32>(UInt32(c.width), UInt32(c.height), 0, 0)
            var off = UInt32.max
            if let rel = c.depth, let dw = c.depthWidth, let dh = c.depthHeight,
               let d = try? Data(contentsOf: dataset.root.appendingPathComponent(rel)), d.count >= dw * dh * 2 {
                off = UInt32(depthHalves.count / 2)
                depthHalves.append(d.prefix(dw * dh * 2))
                dims.z = UInt32(dw); dims.w = UInt32(dh)
            }
            offsets.append(off)
            gpuCams.append(CovCam(r0: row(0), r1: row(1), r2: row(2), intr: SIMD4(c.fx, c.fy, c.cx, c.cy), dims: dims))
        }
        let n = scene.count
        guard n > 0, !gpuCams.isEmpty else { throw PipelineError.noInput("no cameras or splats") }
        let camBuf = ctx.device.makeBuffer(bytes: gpuCams, length: MemoryLayout<CovCam>.stride * gpuCams.count)!
        let offBuf = ctx.device.makeBuffer(bytes: offsets, length: 4 * offsets.count)!
        let depthBuf: MTLBuffer = depthHalves.isEmpty ? ctx.buffer(4)
            : depthHalves.withUnsafeBytes { ctx.device.makeBuffer(bytes: $0.baseAddress!, length: depthHalves.count)! }
        let out = ctx.buffer(n * 8, label: "coverage raw")
        var ncam = UInt32(gpuCams.count), nn = UInt32(n)
        let cb = ctx.queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setBuffer(scene.gpos, offset: 0, index: 0); enc.setBuffer(camBuf, offset: 0, index: 1)
        enc.setBuffer(depthBuf, offset: 0, index: 2); enc.setBuffer(offBuf, offset: 0, index: 3)
        enc.setBytes(&ncam, length: 4, index: 4); enc.setBytes(&nn, length: 4, index: 5)
        enc.setBuffer(out, offset: 0, index: 6)
        enc.dispatch1D(ctx.pipeline("coverage_compute"), count: n)
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()

        let raw = out.contents().bindMemory(to: SIMD2<Float>.self, capacity: n)
        var seenBest: [Float] = [], views: [Float] = []
        seenBest.reserveCapacity(n); views.reserveCapacity(n)
        let pos = scene.gpos.contents().bindMemory(to: SIMD4<Float>.self, capacity: n)
        for i in stride(from: 0, to: n, by: max(1, n / 200_000)) where pos[i].w > 0.1 {
            views.append(raw[i].x)
            if raw[i].x > 0 { seenBest.append(raw[i].y) }
        }
        seenBest.sort(); views.sort()
        let medianBest = seenBest.isEmpty ? 1 : seenBest[seenBest.count / 2]
        let score = ctx.buffer(n * 4, label: "coverage")
        let sp = score.contents().bindMemory(to: Float.self, capacity: n)
        // Summarise by visible size, not splat count: poorly seen areas have few, large splats.
        let cov = scene.gcov.contents().bindMemory(to: SIMD4<Float>.self, capacity: n * 2)
        var poor: Double = 0, unseen: Double = 0, counted: Double = 0
        let lp = log2(poorDetailRatio)
        for i in 0..<n {
            let (cnt, best) = (raw[i].x, raw[i].y)
            var s: Float = 0
            if cnt > 0 {
                let v = min(max((cnt - poorViews) / (goodViews - poorViews), 0), 1)
                let d = min(max((log2(max(best / medianBest, 1e-6)) - lp) / -lp, 0), 1)
                s = min(v, d)
            }
            sp[i] = s
            if pos[i].w > 0.1 {
                let w = Double(max(cov[i * 2].x + cov[i * 2].w + cov[i * 2 + 1].y, 0)).squareRoot() * Double(pos[i].w)   // size, not area: a few huge floaters must not dominate
                counted += w
                if cnt == 0 { unseen += w } else if s < 0.34 { poor += w }
            }
        }
        let viewCounts = (0..<n).map { UInt16(min(raw[$0].x, 65535)) }
        return Result(score: score, poorFraction: counted > 0 ? poor / counted : 0,
                      unseenFraction: counted > 0 ? unseen / counted : 0,
                      medianViews: views.isEmpty ? 0 : Int(views[views.count / 2]),
                      usedDepth: offsets.contains { $0 != UInt32.max }, views: viewCounts)
    }
}
