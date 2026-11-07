import Foundation
import simd

/// Progress snapshot of a training run (local Metal trainer or remote job).
struct TrainingProgressSnapshot: Sendable, Hashable {
    var step: Int
    var totalSteps: Int
    var loss: Double?
    var psnr: Double?
    var numGaussians: Int
    var elapsed: TimeInterval
    var eta: TimeInterval?

    var fraction: Double { totalSteps > 0 ? min(1, Double(step) / Double(totalSteps)) : 0 }

    init(step: Int, totalSteps: Int, loss: Double? = nil, psnr: Double? = nil, numGaussians: Int = 0, elapsed: TimeInterval = 0, eta: TimeInterval? = nil) {
        self.step = step; self.totalSteps = totalSteps; self.loss = loss; self.psnr = psnr
        self.numGaussians = numGaussians; self.elapsed = elapsed; self.eta = eta
    }

    init(_ s: JobStatus) {
        self.init(step: s.step ?? 0, totalSteps: s.totalSteps ?? 0, loss: s.loss, psnr: s.psnr, numGaussians: s.numGaussians ?? 0,
                  elapsed: s.elapsedSeconds ?? 0, eta: s.etaSeconds)
    }
}

/// Implemented by the Metal trainer. Must honour `config.region` (keep/densify only gaussians inside the
/// expanded box) and `config.cameraIds` (train only on those cameras) for block-parallel training (§7).
protocol LocalTrainingRunner: Sendable {
    func train(dataset: PreparedDataset, config: TrainingConfig,
               progress: @escaping @Sendable (TrainingProgressSnapshot) -> Void) async throws -> GaussianCloud
}

extension PreparedDataset {
    /// Same root, manifest restricted to the given camera ids (bounds recomputed). Not saved.
    func restricted(to cameraIds: Set<Int>) -> PreparedDataset {
        var d = self
        d.manifest.cameras = manifest.cameras.filter { cameraIds.contains($0.id) }
        d.recomputeSceneBounds()
        return d
    }
}

// MARK: - Blocks

/// One spatial block of a large scene.
struct SceneBlock: Codable, Hashable, Sendable, Identifiable {
    var index: Int
    /// Un-expanded box (merge keeps gaussians whose centre is inside it).
    var min: SIMD3<Float>
    var max: SIMD3<Float>
    /// Expansion as a fraction of the box size (TrainingConfig.Region.margin).
    var margin: Float
    /// Faces open towards infinity during merging (outer faces of the partition, and the vertical axis),
    /// so no gaussian outside the scene bounds is dropped.
    var openMin: [Bool]
    var openMax: [Bool]
    /// Cameras that see the (expanded) block.
    var cameraIds: [Int]
    /// Points / camera weight used to balance the split.
    var weight: Double
    var id: Int { index }

    var expandedMin: SIMD3<Float> { min - (max - min) * margin }
    var expandedMax: SIMD3<Float> { max + (max - min) * margin }
    var region: TrainingConfig.Region { .init(min: [min.x, min.y, min.z], max: [max.x, max.y, max.z], margin: margin) }

    /// Whether a gaussian centre belongs to this block when merging.
    func owns(_ p: SIMD3<Float>) -> Bool {
        for a in 0..<3 {
            if !openMin[a] && p[a] < min[a] { return false }
            if !openMax[a] && p[a] >= max[a] { return false }
        }
        return true
    }
}

/// Splits a scene into K spatial blocks (VastGaussian / CityGaussian style) and assigns cameras.
struct BlockPartitioner: Sendable {
    /// Number of blocks.
    var blockCount: Int
    /// Box expansion used for training and camera selection.
    var margin: Float = 0.15
    /// A camera is assigned to a block if at least this fraction of the block's sample points project into it.
    var minVisibleFraction: Float = 0.05
    /// Without points: minimum fraction of the image covered by the projected block box.
    var minCoverage: Float = 0.1
    /// Weight of a camera relative to a point when balancing the split.
    var cameraWeight: Double = 50

    init(blockCount: Int, margin: Float = 0.15) { self.blockCount = blockCount; self.margin = margin }

    /// Two horizontal axes (most orthogonal to `up`) and the vertical axis index.
    static func axes(up: SIMD3<Float>) -> (horizontal: [Int], vertical: Int) {
        let a = simd_abs(up)
        let v = a.x >= a.y && a.x >= a.z ? 0 : (a.y >= a.z ? 1 : 2)
        return ([0, 1, 2].filter { $0 != v }, v)
    }

    func partition(dataset: PreparedDataset, points: PointCloud?) -> [SceneBlock] {
        let cams = dataset.manifest.cameras.filter { $0.holdout != true }
        let up = SIMD3<Float>(dataset.manifest.upVector.count == 3 ? dataset.manifest.upVector[0] : 0,
                              dataset.manifest.upVector.count == 3 ? dataset.manifest.upVector[1] : -1,
                              dataset.manifest.upVector.count == 3 ? dataset.manifest.upVector[2] : 0)
        let (hAxes, vAxis) = Self.axes(up: up)
        // Weighted samples: points (subsampled) + camera centres.
        var samples: [(SIMD3<Float>, Double)] = []
        let pts = points?.points ?? []
        let stride = Swift.max(1, pts.count / 200_000)
        var i = 0
        while i < pts.count { samples.append((pts[i], 1)); i += stride }
        for c in cams { samples.append((c.center, cameraWeight)) }
        guard !samples.isEmpty else { return [] }

        // Robust scene bounds (1–99 % of points, always including camera centres).
        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude), hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for a in 0..<3 {
            var v = samples.map { $0.0[a] }
            v.sort()
            lo[a] = v[Int(Float(v.count - 1) * 0.01)]; hi[a] = v[Int(Float(v.count - 1) * 0.99)]
        }
        for c in cams { lo = simd_min(lo, c.center); hi = simd_max(hi, c.center) }
        hi = simd_max(hi, lo + 1e-3)

        // Recursive weighted median splits along the longest horizontal extent.
        struct Cell { var lo: SIMD3<Float>; var hi: SIMD3<Float>; var k: Int; var items: [Int]; var openLo: [Bool]; var openHi: [Bool] }
        var done: [Cell] = []
        var stack = [Cell(lo: lo, hi: hi, k: Swift.max(1, blockCount), items: Array(samples.indices),
                          openLo: [true, true, true], openHi: [true, true, true])]
        while let cell = stack.popLast() {
            if cell.k <= 1 || cell.items.count < 2 { done.append(cell); continue }
            let ext = cell.hi - cell.lo
            let axis = ext[hAxes[0]] >= ext[hAxes[1]] ? hAxes[0] : hAxes[1]
            let kl = cell.k / 2, kr = cell.k - kl
            let sorted = cell.items.sorted { samples[$0].0[axis] < samples[$1].0[axis] }
            let total = sorted.reduce(0.0) { $0 + samples[$1].1 }
            let target = total * Double(kl) / Double(cell.k)
            var acc = 0.0, cut = sorted.count / 2
            for (j, idx) in sorted.enumerated() { acc += samples[idx].1; if acc >= target { cut = Swift.max(1, Swift.min(sorted.count - 1, j + 1)); break } }
            let splitV = (samples[sorted[cut - 1]].0[axis] + samples[sorted[cut]].0[axis]) / 2
            let s = Swift.min(Swift.max(splitV, cell.lo[axis] + 1e-4), cell.hi[axis] - 1e-4)
            var L = cell, R = cell
            L.hi[axis] = s; L.k = kl; L.items = Array(sorted[..<cut]); L.openHi[axis] = false
            R.lo[axis] = s; R.k = kr; R.items = Array(sorted[cut...]); R.openLo[axis] = false
            stack.append(R); stack.append(L)
        }
        done.sort { ($0.lo[hAxes[0]], $0.lo[hAxes[1]]) < ($1.lo[hAxes[0]], $1.lo[hAxes[1]]) }

        return done.enumerated().map { (n, cell) in
            var openLo = cell.openLo, openHi = cell.openHi
            openLo[vAxis] = true; openHi[vAxis] = true   // never split vertically; keep everything above/below
            var b = SceneBlock(index: n, min: cell.lo, max: cell.hi, margin: margin, openMin: openLo, openMax: openHi,
                               cameraIds: [], weight: cell.items.reduce(0.0) { $0 + samples[$1].1 })
            let blockPts = pts.isEmpty ? [] : subsample(pts.filter { inside($0, b.expandedMin, b.expandedMax) }, max: 2000)
            b.cameraIds = cams.filter { sees($0, block: b, points: blockPts) }.map(\.id)
            return b
        }
    }

    private func inside(_ p: SIMD3<Float>, _ lo: SIMD3<Float>, _ hi: SIMD3<Float>) -> Bool {
        p.x >= lo.x && p.y >= lo.y && p.z >= lo.z && p.x <= hi.x && p.y <= hi.y && p.z <= hi.z
    }

    private func subsample(_ p: [SIMD3<Float>], max: Int) -> [SIMD3<Float>] {
        guard p.count > max else { return p }
        let s = Double(p.count) / Double(max)
        return (0..<max).map { p[Int(Double($0) * s)] }
    }

    /// Frustum test: centre inside the expanded box, or enough block points visible, or (no points)
    /// the projected box covers enough of the image.
    func sees(_ c: DatasetCamera, block b: SceneBlock, points: [SIMD3<Float>]) -> Bool {
        let lo = b.expandedMin, hi = b.expandedMax
        if inside(c.center, lo, hi) { return true }
        let w2c = simd_inverse(c.c2w)
        func project(_ p: SIMD3<Float>) -> SIMD3<Float> {
            let q = w2c * SIMD4(p, 1)
            return SIMD3(c.fx * q.x / q.z + c.cx, c.fy * q.y / q.z + c.cy, q.z)
        }
        let W = Float(c.width), H = Float(c.height)
        if !points.isEmpty {
            var n = 0
            for p in points {
                let q = project(p)
                if q.z > 0.01 && q.x >= 0 && q.x < W && q.y >= 0 && q.y < H { n += 1 }
            }
            return Float(n) / Float(points.count) >= minVisibleFraction
        }
        // Box corners (with an infinite-ish vertical extent clipped to the box).
        var minU = Float.greatestFiniteMagnitude, maxU = -Float.greatestFiniteMagnitude
        var minV = Float.greatestFiniteMagnitude, maxV = -Float.greatestFiniteMagnitude
        var front = 0
        for k in 0..<8 {
            let p = SIMD3(k & 1 == 0 ? lo.x : hi.x, k & 2 == 0 ? lo.y : hi.y, k & 4 == 0 ? lo.z : hi.z)
            let q = project(p)
            guard q.z > 0.01 else { continue }
            front += 1
            minU = Swift.min(minU, q.x); maxU = Swift.max(maxU, q.x); minV = Swift.min(minV, q.y); maxV = Swift.max(maxV, q.y)
        }
        guard front > 0 else { return false }
        if front < 8 { return maxU > 0 && minU < W && maxV > 0 && minV < H }   // partially behind: conservative
        let iw = Swift.max(0, Swift.min(maxU, W) - Swift.max(minU, 0)), ih = Swift.max(0, Swift.min(maxV, H) - Swift.max(minV, 0))
        return iw * ih / (W * H) >= minCoverage
    }
}

/// Merges per-block results: each block contributes only gaussians whose centre lies in its un-expanded box.
enum BlockMerger {
    static func merge(_ parts: [(block: SceneBlock, cloud: GaussianCloud)]) -> GaussianCloud {
        var out = GaussianCloud()
        out.shDegree = parts.map(\.cloud.shDegree).max() ?? 0
        out.antialiased = parts.contains { $0.cloud.antialiased }
        out.reserve(parts.reduce(0) { $0 + $1.cloud.count })
        for (b, g) in parts {
            for i in 0..<g.count where b.owns(g.position(i)) {
                out.positions.append(contentsOf: g.positions[i*3..<i*3+3])
                out.scales.append(contentsOf: g.scales[i*3..<i*3+3])
                out.rotations.append(contentsOf: g.rotations[i*4..<i*4+4])
                out.opacities.append(g.opacities[i])
                out.sh.append(contentsOf: g.sh[i*GaussianCloud.shStride..<(i+1)*GaussianCloud.shStride])
            }
        }
        return out
    }
}
