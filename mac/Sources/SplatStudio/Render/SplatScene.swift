import Foundation
import Metal
import simd

/// GPU-resident, render-ready gaussians. Immutable once built except for per-splat flags
/// (deleted / selected), which the editor changes in place.
final class SplatScene: @unchecked Sendable {
    let count: Int
    let shDegree: Int
    let antialiased: Bool
    let gpos: MTLBuffer      // float4: xyz, opacity
    let gcov: MTLBuffer      // 2 × float4 covariance
    let gsh: MTLBuffer       // half × 48
    let gflags: MTLBuffer    // uchar: bit0 deleted, bit1 selected
    let boundsMin: SIMD3<Float>
    let boundsMax: SIMD3<Float>
    let center: SIMD3<Float>
    let radius: Float

    /// Builds the render buffers from trainer-layout parameter buffers (no CPU copy).
    init(ctx: MetalContext = .shared, count: Int, shDegree: Int, antialiased: Bool,
         pos: MTLBuffer, logScale: MTLBuffer, quat: MTLBuffer, opacity: MTLBuffer, sh: MTLBuffer,
         flags: MTLBuffer? = nil, bounds: (SIMD3<Float>, SIMD3<Float>)? = nil) {
        self.count = count
        self.shDegree = shDegree
        self.antialiased = antialiased
        gpos = ctx.buffer(count * 16, label: "gpos")
        gcov = ctx.buffer(count * 32, label: "gcov")
        gsh = ctx.buffer(count * 96, label: "gsh")
        if let flags, flags.length >= count { gflags = flags } else {
            gflags = ctx.buffer(count, label: "gflags")
            memset(gflags.contents(), 0, max(count, 1))
        }
        let cb = ctx.queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setBuffer(pos, offset: 0, index: 0); enc.setBuffer(logScale, offset: 0, index: 1)
        enc.setBuffer(quat, offset: 0, index: 2); enc.setBuffer(opacity, offset: 0, index: 3)
        enc.setBuffer(sh, offset: 0, index: 4); enc.setBuffer(gpos, offset: 0, index: 5)
        enc.setBuffer(gcov, offset: 0, index: 6); enc.setBuffer(gsh, offset: 0, index: 7)
        enc.set(UInt32(count), 8)
        enc.dispatch1D(ctx.pipeline("prepare_splats"), count: count)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let bounds {
            boundsMin = bounds.0; boundsMax = bounds.1
        } else {
            (boundsMin, boundsMax) = Self.robustBounds(gpos, count)
        }
        center = (boundsMin + boundsMax) * 0.5
        radius = max(simd_length(boundsMax - boundsMin) * 0.5, 1e-3)
    }

    convenience init(cloud: GaussianCloud, ctx: MetalContext = .shared) {
        let n = max(cloud.count, 1)
        func buf(_ a: [Float], _ per: Int) -> MTLBuffer { a.isEmpty ? ctx.buffer(n * per * 4) : ctx.buffer(a) }
        let b = cloud.count > 0 ? cloud.robustBounds() : (SIMD3<Float>(-1, -1, -1), SIMD3<Float>(1, 1, 1))
        self.init(ctx: ctx, count: cloud.count, shDegree: cloud.shDegree, antialiased: cloud.antialiased,
                  pos: buf(cloud.positions, 3), logScale: buf(cloud.scales, 3), quat: buf(cloud.rotations, 4),
                  opacity: buf(cloud.opacities, 1), sh: buf(cloud.sh, 48), bounds: b)
    }

    private static func robustBounds(_ b: MTLBuffer, _ n: Int) -> (SIMD3<Float>, SIMD3<Float>) {
        guard n > 0 else { return (SIMD3(-1, -1, -1), SIMD3(1, 1, 1)) }
        let p = b.contents().bindMemory(to: SIMD4<Float>.self, capacity: n)
        let stride = max(1, n / 100_000)
        var xs: [[Float]] = [[], [], []]
        var i = 0
        while i < n { for a in 0..<3 { xs[a].append(p[i][a]) }; i += stride }
        var lo = SIMD3<Float>(), hi = SIMD3<Float>()
        for a in 0..<3 {
            xs[a].sort()
            lo[a] = xs[a][Int(Float(xs[a].count - 1) * 0.02)]
            hi[a] = xs[a][Int(Float(xs[a].count - 1) * 0.98)]
        }
        return (lo, hi)
    }

    // MARK: editing helpers (CPU side, unified memory)

    var flagsPointer: UnsafeMutablePointer<UInt8> { gflags.contents().bindMemory(to: UInt8.self, capacity: max(count, 1)) }
    var positionsPointer: UnsafeMutablePointer<SIMD4<Float>> { gpos.contents().bindMemory(to: SIMD4<Float>.self, capacity: max(count, 1)) }

    var deletedCount: Int { var c = 0; let f = flagsPointer; for i in 0..<count where f[i] & 1 != 0 { c += 1 }; return c }
    var selectedCount: Int { var c = 0; let f = flagsPointer; for i in 0..<count where f[i] & 3 == 2 { c += 1 }; return c }
}
