import Foundation
import Metal
import ImageIO
import CoreGraphics
import simd

/// A training view uploaded to the GPU (unified memory, so this is a single copy).
final class TrainView {
    let camera: DatasetCamera          // intrinsics scaled to the loaded resolution
    let width: Int
    let height: Int
    let target: MTLBuffer              // uchar4 RGBA; A = mask (255 keep)
    let depth: MTLBuffer?              // float, depthWidth × depthHeight
    let depthSize: SIMD2<UInt32>
    let worldToCam: simd_float4x4
    let holdout: Bool
    /// Sky mask (uchar per pixel, > 127 = sky; learned skybox) and its sky-pixel count, when the photo has one.
    var sky: MTLBuffer?
    var skyCount = 0
    /// Lower-resolution copies for coarse-to-fine training (factor → view), built on first use.
    var levels: [Int: TrainView] = [:]

    init(camera: DatasetCamera, width: Int, height: Int, target: MTLBuffer, depth: MTLBuffer?, depthSize: SIMD2<UInt32>, holdout: Bool) {
        self.camera = camera; self.width = width; self.height = height
        self.target = target; self.depth = depth; self.depthSize = depthSize
        self.worldToCam = camera.c2w.inverse
        self.holdout = holdout
    }
}

enum TrainingDataLoader {
    /// Loads all images of the dataset, downscaled so the longest side ≤ maxImageSize.
    static func load(dataset: PreparedDataset, config: TrainingConfig, ctx: MetalContext,
                     progress: (Double) -> Void) throws -> [TrainView] {
        var cams = dataset.manifest.cameras
        if let ids = config.cameraIds { let s = Set(ids); cams = cams.filter { s.contains($0.id) } }
        var views = [TrainView?](repeating: nil, count: cams.count)
        let lock = NSLock()
        var done = 0
        var firstError: Error?
        DispatchQueue.concurrentPerform(iterations: cams.count) { i in
            do {
                let v = try loadView(cams[i], dataset: dataset, maxSize: config.maxImageSize, ctx: ctx)
                lock.lock(); views[i] = v; done += 1; lock.unlock()
            } catch {
                lock.lock(); if firstError == nil { firstError = error }; lock.unlock()
            }
        }
        progress(1)
        if let firstError { throw firstError }
        _ = done
        return views.compactMap { $0 }
    }

    static func loadView(_ cam: DatasetCamera, dataset: PreparedDataset, maxSize: Int, ctx: MetalContext) throws -> TrainView {
        let url = dataset.url(cam.image)
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { throw FormatError.invalid("cannot open \(cam.image)") }
        let longest = max(cam.width, cam.height)
        let scale = longest > maxSize ? Double(maxSize) / Double(longest) : 1
        let w = max(1, Int((Double(cam.width) * scale).rounded()))
        let h = max(1, Int((Double(cam.height) * scale).rounded()))
        let opts: [CFString: Any] = [kCGImageSourceCreateThumbnailFromImageAlways: true,
                                     kCGImageSourceThumbnailMaxPixelSize: max(w, h),
                                     kCGImageSourceCreateThumbnailWithTransform: false]
        guard let img = CGImageSourceCreateThumbnailAtIndex(src, 0, opts as CFDictionary) ?? CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            throw FormatError.invalid("cannot decode \(cam.image)")
        }
        let buf = ctx.buffer(w * h * 4, label: "target")
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        guard let cg = CGContext(data: buf.contents(), width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: cs,
                                 bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { throw FormatError.invalid("context") }
        cg.interpolationQuality = .high
        cg.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
        let px = buf.contents().bindMemory(to: UInt8.self, capacity: w * h * 4)
        for i in 0..<(w * h) { px[i * 4 + 3] = 255 }
        if let m = cam.mask, let msrc = CGImageSourceCreateWithURL(dataset.url(m) as CFURL, nil),
           let mimg = CGImageSourceCreateImageAtIndex(msrc, 0, nil) {
            var gray = [UInt8](repeating: 255, count: w * h)
            let gs = CGColorSpaceCreateDeviceGray()
            gray.withUnsafeMutableBytes { raw in
                if let gc = CGContext(data: raw.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w, space: gs, bitmapInfo: CGImageAlphaInfo.none.rawValue) {
                    gc.draw(mimg, in: CGRect(x: 0, y: 0, width: w, height: h))
                }
            }
            for i in 0..<(w * h) { px[i * 4 + 3] = gray[i] }
        }
        var skyBuf: MTLBuffer?, skyCount = 0
        if let sk = cam.sky, let ssrc = CGImageSourceCreateWithURL(dataset.url(sk) as CFURL, nil),
           let simg = CGImageSourceCreateImageAtIndex(ssrc, 0, nil) {
            let b = ctx.buffer(w * h, label: "sky")
            memset(b.contents(), 0, w * h)
            if let gc = CGContext(data: b.contents(), width: w, height: h, bitsPerComponent: 8, bytesPerRow: w,
                                  space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue) {
                gc.interpolationQuality = .high
                gc.draw(simg, in: CGRect(x: 0, y: 0, width: w, height: h))
            }
            let sp = b.contents().bindMemory(to: UInt8.self, capacity: w * h)
            for i in 0..<(w * h) where sp[i] > 127 { skyCount += 1 }
            skyBuf = b
        }
        var c = cam
        let sx = Float(w) / Float(cam.width), sy = Float(h) / Float(cam.height)
        c.fx *= sx; c.fy *= sy; c.cx *= sx; c.cy *= sy; c.width = w; c.height = h
        var depthBuf: MTLBuffer?
        var dsize = SIMD2<UInt32>(1, 1)
        if let d = cam.depth, let dw = cam.depthWidth, let dh = cam.depthHeight {
            let values = try PreparedDataset.readDepth(dataset.url(d), width: dw, height: dh)
            depthBuf = ctx.buffer(values, label: "depth")
            dsize = SIMD2(UInt32(dw), UInt32(dh))
        }
        let view = TrainView(camera: c, width: w, height: h, target: buf, depth: depthBuf, depthSize: dsize, holdout: cam.holdout ?? false)
        view.sky = skyBuf; view.skyCount = skyCount
        return view
    }
}

/// Builds the initial gaussians from a point cloud (or randomly).
enum GaussianInitializer {
    static func fromPoints(_ pc: PointCloud, config: TrainingConfig) -> GaussianCloud {
        var pts = pc.points, cols = pc.colors
        if let r = config.region {
            let lo = SIMD3(r.min[0], r.min[1], r.min[2]), hi = SIMD3(r.max[0], r.max[1], r.max[2])
            let m = (hi - lo) * r.margin
            var p2: [SIMD3<Float>] = [], c2: [SIMD3<UInt8>] = []
            for (i, p) in pts.enumerated() where all(p .>= lo - m) && all(p .<= hi + m) { p2.append(p); c2.append(i < cols.count ? cols[i] : SIMD3(128, 128, 128)) }
            pts = p2; cols = c2
            if pts.isEmpty {
                // A block with no triangulated points (foliage, sky-facing, textureless): start from a uniform
                // fill of its box instead of nothing (the iPhone trained such a block to 0 splats).
                var rng = SplitMix64(seed: UInt64(config.seed) &+ 11)
                let n = min(50_000, max(config.maxGaussians / 10, 1_000))
                for _ in 0..<n {
                    pts.append(lo - m + SIMD3(rng.uniform(), rng.uniform(), rng.uniform()) * (hi - lo + 2 * m))
                    cols.append(SIMD3(128, 128, 128))
                }
            }
        }
        // Sparse SfM clouds (Apple Object Capture yields ~1–5 k points) are topped up so densification
        // starts from a useful population: jittered copies of real points at their local spacing, plus a
        // light uniform fill of the points' bounding region for background the solver didn't triangulate.
        let floor = min(100_000, max(config.maxGaussians / 10, 1))
        if pts.count > 0 && pts.count < floor {
            var rng = SplitMix64(seed: UInt64(config.seed) &+ 7)
            let d2 = KNN.meanSquaredDistance3(pts)
            var lo = pts[0], hi = pts[0]
            for p in pts { lo = simd_min(lo, p); hi = simd_max(hi, p) }
            let extra = floor - pts.count
            let fillCount = extra / 5
            let base = pts.count
            for k in 0..<(extra - fillCount) {
                let i = k % base
                let sigma = sqrt(max(d2[i], 1e-10))
                pts.append(pts[i] + SIMD3(rng.normal(), rng.normal(), rng.normal()) * sigma)
                cols.append(i < cols.count ? cols[i] : SIMD3(128, 128, 128))
            }
            let ext = hi - lo
            for _ in 0..<fillCount {
                pts.append(lo - ext * 0.25 + SIMD3(rng.uniform(), rng.uniform(), rng.uniform()) * ext * 1.5)
                cols.append(SIMD3(128, 128, 128))
            }
        }
        // Cap the initial count to the budget (keep a uniform random subset).
        if pts.count > config.maxGaussians {
            var rng = SplitMix64(seed: UInt64(config.seed))
            var idx = Array(0..<pts.count)
            for i in 0..<config.maxGaussians { let j = i + Int(rng.next() % UInt64(idx.count - i)); idx.swapAt(i, j) }
            let keep = idx.prefix(config.maxGaussians)
            pts = keep.map { pts[$0] }; cols = keep.map { $0 < cols.count ? cols[$0] : SIMD3(128, 128, 128) }
        }
        let n = pts.count
        var g = GaussianCloud(count: n, shDegree: config.shDegree)
        g.antialiased = config.antialiased
        let d2 = KNN.meanSquaredDistance3(pts)
        let logit = GaussianCloud.invSigmoid(config.initialOpacity)
        for i in 0..<n {
            let p = pts[i]
            g.positions[i*3] = p.x; g.positions[i*3+1] = p.y; g.positions[i*3+2] = p.z
            let s = log(max(sqrt(d2[i]), 1e-7))
            g.scales[i*3] = s; g.scales[i*3+1] = s; g.scales[i*3+2] = s
            g.opacities[i] = logit
            let c = i < cols.count ? SIMD3<Float>(Float(cols[i].x), Float(cols[i].y), Float(cols[i].z)) / 255 : SIMD3(repeating: 0.5)
            g.sh[i*48] = (c.x - 0.5) / GaussianCloud.C0
            g.sh[i*48+1] = (c.y - 0.5) / GaussianCloud.C0
            g.sh[i*48+2] = (c.z - 0.5) / GaussianCloud.C0
        }
        return g
    }

    static func random(count: Int, center: SIMD3<Float>, extent: Float, config: TrainingConfig) -> GaussianCloud {
        var rng = SplitMix64(seed: UInt64(config.seed))
        var pc = PointCloud()
        for _ in 0..<count {
            let p = SIMD3<Float>(rng.uniform() * 2 - 1, rng.uniform() * 2 - 1, rng.uniform() * 2 - 1) * extent * 3 + center
            pc.points.append(p)
            pc.colors.append(SIMD3(UInt8(rng.next() & 255), UInt8(rng.next() & 255), UInt8(rng.next() & 255)))
        }
        return fromPoints(pc, config: config)
    }
}

/// Mean squared distance to the three nearest neighbours, via a uniform hash grid.
enum KNN {
    static func meanSquaredDistance3(_ pts: [SIMD3<Float>]) -> [Float] {
        let n = pts.count
        guard n > 3 else { return [Float](repeating: 0.01, count: n) }
        var lo = pts[0], hi = pts[0]
        for p in pts { lo = simd_min(lo, p); hi = simd_max(hi, p) }
        let ext = simd_max(hi - lo, SIMD3(repeating: 1e-6))
        let vol = ext.x * ext.y * ext.z
        var cell = cbrt(vol / Float(n) * 4)
        if !cell.isFinite || cell <= 0 { cell = max(ext.max() / 100, 1e-6) }
        struct Key: Hashable { var x: Int32; var y: Int32; var z: Int32 }
        func key(_ p: SIMD3<Float>) -> Key {
            let q = (p - lo) / cell
            return Key(x: Int32(q.x), y: Int32(q.y), z: Int32(q.z))
        }
        var grid: [Key: [Int32]] = [:]
        grid.reserveCapacity(n)
        for (i, p) in pts.enumerated() { grid[key(p), default: []].append(Int32(i)) }
        var out = [Float](repeating: 0, count: n)
        let lockedGrid = grid
        DispatchQueue.concurrentPerform(iterations: 64) { chunk in
            let start = n * chunk / 64, end = n * (chunk + 1) / 64
            for i in start..<end {
                let p = pts[i]
                let k = key(p)
                var best = SIMD3<Float>(repeating: .infinity)
                var ring: Int32 = 1
                while true {
                    for dx in -ring...ring { for dy in -ring...ring { for dz in -ring...ring {
                        if ring > 1 && abs(dx) < ring && abs(dy) < ring && abs(dz) < ring { continue }
                        guard let list = lockedGrid[Key(x: k.x + dx, y: k.y + dy, z: k.z + dz)] else { continue }
                        for j in list where Int(j) != i {
                            let d = simd_distance_squared(p, pts[Int(j)])
                            if d < best.z {
                                if d < best.x { best = SIMD3(d, best.x, best.y) }
                                else if d < best.y { best = SIMD3(best.x, d, best.y) }
                                else { best.z = d }
                            }
                        }
                    }}}
                    // Points beyond `ring` cells are at least (ring) * cell away.
                    let bound = Float(ring) * cell
                    if best.z <= bound * bound || ring >= 4 { break }
                    ring += 1
                }
                let m = best.z.isFinite ? (best.x + best.y + best.z) / 3 : (best.x.isFinite ? best.x : cell * cell)
                out[i] = max(m, 1e-12)
            }
        }
        return out
    }
}

struct SplitMix64 {
    var state: UInt64
    init(seed: UInt64) { state = seed &+ 0x9E37_79B9_7F4A_7C15 }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    mutating func uniform() -> Float { Float(next() >> 40) / Float(1 << 24) }
    mutating func normal() -> Float {
        let u1 = max(uniform(), 1e-7), u2 = uniform()
        return sqrt(-2 * log(u1)) * cos(2 * .pi * u2)
    }
}
