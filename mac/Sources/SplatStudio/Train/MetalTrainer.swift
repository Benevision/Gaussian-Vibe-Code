import Foundation
import Metal
import simd
import QuartzCore

/// Mirrors `RasterParams` in TrainShaders.
struct RasterParams {
    var width: UInt32 = 0, height: UInt32 = 0, tilesX: UInt32 = 0, tilesY: UInt32 = 0
    var bg = SIMD4<Float>()
    var absGrad: UInt32 = 0, hasDepth: UInt32 = 0, numGaussians: UInt32 = 0, step: UInt32 = 0
    var ssimWeight: Float = 0.2, depthWeight: Float = 0, invNumPixels: Float = 0, pad: Float = 0
}

struct AdamParams {
    var lr: Float, lr2: Float, period: UInt32, split: UInt32
    var beta1: Float = 0.9, beta2: Float = 0.999, eps: Float = 1e-15
    var bc1: Float, bc2: Float, n: UInt32
}

struct TrainerStatus: Sendable, Equatable {
    var step = 0
    var totalSteps = 0
    var loss: Float = 0
    var psnr: Float = 0
    var evalPSNR: Float? = nil
    var numGaussians = 0
    var elapsed: Double = 0
    var eta: Double = 0
    var itersPerSecond: Double = 0
    /// Fraction of pixels currently ignored as moving / transient (robust training).
    var transientFraction: Float = 0
    var shDegree = 0
    var phase = "Starting"
}

enum TrainerError: LocalizedError {
    case noViews, cancelled, outOfMemory(String)
    var errorDescription: String? {
        switch self {
        case .noViews: return "The dataset has no usable training images."
        case .cancelled: return "Training was cancelled."
        case .outOfMemory(let s): return "Not enough GPU memory: \(s)"
        }
    }
}

/// 3D Gaussian Splatting optimiser running entirely on Metal.
///
/// Pipeline per step: preprocess (project, SH colour, tile bounds) → depth radix sort →
/// tile-key duplication in depth order → stable tile sort → per-tile forward raster →
/// fused L1 + D-SSIM (+ depth L1) loss and gradient → backward raster (SIMD-reduced atomics)
/// → analytic preprocess backward → Adam → 3DGS-MCMC noise. Densification (MCMC relocation
/// or classic ADC) runs on the CPU over the same unified-memory buffers.
final class MetalTrainer: @unchecked Sendable {
    let ctx: MetalContext
    private(set) var config: TrainingConfig
    /// Appearance correction was requested but switched off because every photo has the same exposure.
    private(set) var appearanceAutoOff = false
    let dataset: PreparedDataset
    private(set) var views: [TrainView] = []
    private var trainViews: [TrainView] = []
    private var evalViews: [TrainView] = []

    // parameters: 0 pos(3) 1 logScale(3) 2 quat(4) 3 opacityLogit(1) 4 sh(48)
    static let widths = [3, 3, 4, 1, 48]
    private(set) var n = 0
    private var params: [MTLBuffer] = []
    private var grads: [MTLBuffer] = []
    private var m1: [MTLBuffer] = []
    private var m2: [MTLBuffer] = []
    private var alive: MTLBuffer!
    private var gradStats: MTLBuffer!  // accum, count, maxRadius
    private var capacity = 0

    // per-gaussian scratch
    private var p2d: MTLBuffer!, conicO: MTLBuffer!, rgbd: MTLBuffer!, rects: MTLBuffer!
    private var tilesTouched: MTLBuffer!, depthKeys: MTLBuffer!, sortVals: MTLBuffer!
    private var counts: MTLBuffer!, offsets: MTLBuffer!, acc: MTLBuffer!
    private let total: MTLBuffer
    // per-tile-instance
    private var tileKeys: MTLBuffer?, tileVals: MTLBuffer?
    private var tileCapacity = 0
    // per-pixel
    private var ranges: MTLBuffer!, img: MTLBuffer!, finalT: MTLBuffer!, nContrib: MTLBuffer!, gradImg: MTLBuffer!
    private var planes: MTLBuffer!, tmp: MTLBuffer!, mu: MTLBuffer!, dmaps: MTLBuffer!
    private let stats: MTLBuffer
    // transient robustness
    private var resid: MTLBuffer!, weights: MTLBuffer!
    // Per-photo appearance correction (config.appearanceCorrection): 12 floats per view (3×3 M + b),
    // the raw render it was applied to, the GPU-reduced gradient, and CPU-side Adam moments.
    private var rawImg: MTLBuffer!, apParams: MTLBuffer!, apGrad: MTLBuffer!
    /// Correction used for views without their own (held-out views): the average learned one (see meanAppearance).
    private var apHoldout: MTLBuffer?
    private var apM1: [Float] = [], apM2: [Float] = [], apSteps: [Int] = [], apInit: [Float] = []
    private var viewIndex: [ObjectIdentifier: Int] = [:]
    /// Moving-object masking is only trusted when every surface is seen from several nearby views.
    private(set) var transientEnabled = false
    /// Median over photos of the gap to the most similar other photo (degrees of view direction plus
    /// camera offset as an angle at scene scale). ~5° = dense capture, >15° = sparse.
    private(set) var medianViewGap: Double = 0

    static func medianNearestViewGap(_ cams: [DatasetCamera], sceneExtent: Float) -> Double {
        guard cams.count > 2 else { return 0 }
        let pos = cams.map { $0.center }, fwd = cams.map { $0.forward }
        let ext = max(sceneExtent, 1e-3)
        var gaps: [Double] = []
        // Subsample big sets (O(n²)): every k-th photo against all.
        let stride = max(1, cams.count / 400)
        for i in Swift.stride(from: 0, to: cams.count, by: stride) {
            var best = Double.infinity
            for j in cams.indices where j != i {
                let ang = acos(Double(min(1, max(-1, simd_dot(fwd[i], fwd[j]))))) * 180 / .pi
                let off = Double(simd_distance(pos[i], pos[j]) / ext) * 180 / .pi
                best = min(best, ang + off)
            }
            gaps.append(best)
        }
        gaps.sort()
        return gaps[gaps.count / 2]
    }
    private let residHist: MTLBuffer
    private let outlierCount: MTLBuffer
    private var transientThreshold: Float = -1
    private(set) var transientFraction: Float = 0
    private let emptyDepth: MTLBuffer
    private var maxPixels = 0
    private let sorter: GPURadixSort
    private let tileSorter: GPURadixSort
    private let scan: GPUScan
    private let ssimWindow: [Float]

    private var rng: SplitMix64
    private var order: [Int] = []
    private var orderPos = 0
    private(set) var step = 0
    private var sceneExtent: Float = 1
    private var lastStats = (loss: Float(0), psnr: Float(0))
    private var adamStep = 0

    // control
    private let controlLock = NSLock()
    private var _cancelled = false
    private var _paused = false
    var isCancelled: Bool { controlLock.lock(); defer { controlLock.unlock() }; return _cancelled }
    var isPaused: Bool { controlLock.lock(); defer { controlLock.unlock() }; return _paused }
    func cancel() { controlLock.lock(); _cancelled = true; controlLock.unlock() }
    func setPaused(_ p: Bool) { controlLock.lock(); _paused = p; controlLock.unlock() }

    /// Serialises access to the parameter buffers between the training loop and preview readers.
    let paramLock = NSLock()

    init(dataset: PreparedDataset, config: TrainingConfig, ctx: MetalContext = .shared) {
        self.ctx = ctx
        self.dataset = dataset
        self.config = config
        sorter = GPURadixSort(ctx: ctx)
        tileSorter = GPURadixSort(ctx: ctx)
        scan = GPUScan(ctx: ctx)
        total = ctx.buffer(16)
        stats = ctx.buffer(64)
        residHist = ctx.buffer(256 * 4, label: "residHist")
        outlierCount = ctx.buffer(16, label: "outliers")
        emptyDepth = ctx.buffer(16)
        rng = SplitMix64(seed: UInt64(config.seed))
        var w = (0..<11).map { i -> Float in let d = Float(i - 5); return exp(-d * d / (2 * 1.5 * 1.5)) }
        let s = w.reduce(0, +); w = w.map { $0 / s }
        ssimWindow = w
        sceneExtent = max(dataset.manifest.sceneExtent, 1e-3)
    }

    // MARK: setup

    func prepare(progress: (String) -> Void) throws {
        progress("Loading images")
        views = try TrainingDataLoader.load(dataset: dataset, config: config, ctx: ctx) { _ in }
        trainViews = views.filter { !$0.holdout }
        evalViews = views.filter { $0.holdout }
        if trainViews.isEmpty { trainViews = views }
        guard !trainViews.isEmpty else { throw TrainerError.noViews }
        medianViewGap = Self.medianNearestViewGap(trainViews.map { $0.camera }, sceneExtent: sceneExtent)
        transientEnabled = config.transientMasking && medianViewGap <= Double(config.transientMaxViewGap)
        if config.transientMasking && !transientEnabled {
            progress(String(format: "Sparse coverage (photos ~%.0f° apart): not ignoring moving objects, real scenery would be mistaken for them", medianViewGap))
        }
        maxPixels = views.map { $0.width * $0.height }.max() ?? 0
        allocPixelBuffers()
        progress("Initialising gaussians")
        let cloud: GaussianCloud
        if let pURL = dataset.pointCloudURL, FileManager.default.fileExists(atPath: pURL.path),
           let pc = try? PLYCodec.readPointCloud(url: pURL), pc.count > 10 {
            cloud = GaussianInitializer.fromPoints(pc, config: config)
        } else {
            let c = dataset.manifest.sceneCenter
            cloud = GaussianInitializer.random(count: config.randomInitCount, center: SIMD3(c[0], c[1], c[2]), extent: sceneExtent, config: config)
        }
        load(cloud)
    }

    /// Replaces all parameters (also used to resume from a checkpoint).
    func load(_ cloud: GaussianCloud) {
        n = cloud.count
        allocGaussianBuffers(max(n, 1))
        let arrays = [cloud.positions, cloud.scales, cloud.rotations, cloud.opacities, cloud.sh]
        for k in 0..<5 {
            arrays[k].withUnsafeBytes { src in memcpy(params[k].contents(), src.baseAddress!, src.count) }
            memset(m1[k].contents(), 0, m1[k].length); memset(m2[k].contents(), 0, m2[k].length)
        }
        memset(alive.contents(), 1, max(n, 1))
        memset(gradStats.contents(), 0, gradStats.length)
    }

    private func allocGaussianBuffers(_ count: Int) {
        let cap = Int(Double(count) * 1.1) + 1024
        func mk(_ bytes: Int, _ l: String) -> MTLBuffer { ctx.buffer(bytes, label: l) }
        params = Self.widths.enumerated().map { mk(cap * $0.element * 4, "param\($0.offset)") }
        grads = Self.widths.enumerated().map { mk(cap * $0.element * 4, "grad\($0.offset)") }
        m1 = Self.widths.map { mk(cap * $0 * 4, "adamM") }
        m2 = Self.widths.map { mk(cap * $0 * 4, "adamV") }
        alive = mk(cap, "alive")
        gradStats = mk(cap * 12, "gradStats")
        p2d = mk(cap * 8, "p2d"); conicO = mk(cap * 16, "conicO"); rgbd = mk(cap * 16, "rgbd")
        rects = mk(cap * 16, "rects"); tilesTouched = mk(cap * 4, "tiles"); depthKeys = mk(cap * 4, "dkeys")
        sortVals = mk(cap * 4, "svals"); counts = mk(cap * 4, "counts"); offsets = mk(cap * 4, "offsets")
        acc = mk(cap * 48, "acc")
        capacity = cap
    }

    private func allocPixelBuffers() {
        let hw = maxPixels
        let tiles = views.map { (($0.width + 15) / 16) * (($0.height + 15) / 16) }.max() ?? 1
        ranges = ctx.buffer(tiles * 8, label: "ranges")
        img = ctx.buffer(hw * 16, label: "img"); finalT = ctx.buffer(hw * 4, label: "finalT")
        nContrib = ctx.buffer(hw * 4, label: "nContrib"); gradImg = ctx.buffer(hw * 16, label: "gradImg")
        planes = ctx.buffer(hw * 15 * 4, label: "planes"); tmp = ctx.buffer(hw * 15 * 4, label: "tmp")
        resid = ctx.buffer(hw * 4, label: "resid"); weights = ctx.buffer(hw * 4, label: "weights")
        mu = ctx.buffer(hw * 15 * 4, label: "mu"); dmaps = ctx.buffer(hw * 9 * 4, label: "dmaps")
        rawImg = ctx.buffer(hw * 12, label: "rawImg"); apGrad = ctx.buffer(64, label: "apGrad")
        if config.appearanceCorrection, Self.exposureLocked(views.map(\.camera)) {
            // Locked exposure (Splat Capture): nothing to correct, and the extra freedom costs held-out quality
            // (bedroom, 8k: 21.20 dB with correction, 22.21 dB without) and drifts colours too bright.
            config.appearanceCorrection = false
            appearanceAutoOff = true
        }
        setupAppearance()
    }

    /// Identity correction per view, scaled by recorded exposure when available:
    /// photo ≈ gain · render with gain = (E_i / E_median)^(1/2.2) (sRGB-ish), clamped to [1/3, 3].
    /// Every photo has a recorded exposure and they're all within 10 % of each other.
    static func exposureLocked(_ cams: [DatasetCamera]) -> Bool {
        let e = cams.map { $0.exposure ?? 0 }
        guard !e.isEmpty, let lo = e.min(), let hi = e.max(), lo > 0 else { return false }
        return hi / lo <= 1.1
    }

    private func setupAppearance() {
        let nv = views.count
        apParams = ctx.buffer(max(nv, 1) * 48, label: "appearance")
        apM1 = Array(repeating: 0, count: nv * 12); apM2 = apM1; apSteps = Array(repeating: 0, count: nv)
        viewIndex = Dictionary(uniqueKeysWithValues: views.enumerated().map { (ObjectIdentifier($0.element), $0.offset) })
        let exps = views.compactMap { $0.camera.exposure }.filter { $0 > 0 }.sorted()
        let ref = exps.isEmpty ? nil : exps[exps.count / 2]
        let p = apParams.contents().bindMemory(to: Float.self, capacity: nv * 12)
        for (i, v) in views.enumerated() {
            var gain: Float = 1
            if let ref, let e = v.camera.exposure, e > 0 { gain = min(3, max(1.0 / 3, Float(pow(e / ref, 1 / 2.2)))) }
            let base = i * 12
            for k in 0..<12 { p[base + k] = 0 }
            p[base + 0] = gain; p[base + 4] = gain; p[base + 8] = gain
        }
        apInit = (0..<nv * 12).map { p[$0] }
        apHoldout = ctx.buffer(48, label: "appearance holdout")
    }

    /// The average per-photo colour correction with each photo's exposure seed divided out (row-major 3×3 +
    /// offset), or nil when appearance correction is off. Splat colours and the corrections share an overall
    /// level that training can't pin down (brighter splats + darker corrections fit equally well); over 30k steps
    /// that drifted the bedroom splat 10–30 levels too bright. Exports bake this average in, and held-out views
    /// are rendered with it, so the splat looks like a typical photo of the capture.
    func meanAppearance() -> [Float]? {
        guard config.appearanceCorrection, let apParams, !views.isEmpty, apInit.count == views.count * 12 else { return nil }
        let p = apParams.contents().bindMemory(to: Float.self, capacity: views.count * 12)
        var m = [Float](repeating: 0, count: 12)
        for i in views.indices {
            let g = max(apInit[i * 12], 1e-3)   // exposure seed gain (diagonal)
            for k in 0..<12 { m[k] += p[i * 12 + k] / g }
        }
        return m.map { $0 / Float(views.count) }
    }

    /// Adam on one view's 12 correction parameters, after the GPU has reduced their gradient.
    private func updateAppearance(view i: Int) {
        let g = apGrad.contents().bindMemory(to: Float.self, capacity: 12)
        let p = apParams.contents().bindMemory(to: Float.self, capacity: views.count * 12)
        if (0..<12).contains(where: { !g[$0].isFinite }) {
            if Self.debugAppearance {
                let pr = (0..<12).map { String(format: "%.3f", p[i * 12 + $0]) }.joined(separator: " ")
                print("non-finite appearance grad view \(i) step \(step) params [\(pr)] grad \((0..<12).map { g[$0] })")
            }
            return
        }
        apSteps[i] += 1
        let t = Float(apSteps[i]), lr = config.appearanceLR, reg = config.appearanceReg
        let bc1 = 1 - pow(Float(0.9), t), bc2 = 1 - pow(Float(0.999), t)
        for k in 0..<12 {
            let j = i * 12 + k
            let gk = g[k] + reg * (p[j] - apInit[j])   // pulled towards the exposure-seeded start
            apM1[j] = 0.9 * apM1[j] + 0.1 * gk
            apM2[j] = 0.999 * apM2[j] + 0.001 * gk * gk
            p[j] -= lr * (apM1[j] / bc1) / (sqrt(apM2[j] / bc2) + 1e-8)
        }
        if Self.debugAppearance, step % 250 == 0 {
            let row = (0..<12).map { String(format: "%.3f", p[i * 12 + $0]) }.joined(separator: " ")
            let gr = (0..<12).map { String(format: "%.1e", g[$0]) }.joined(separator: " ")
            print("appearance view \(i) step \(step): [\(row)]  grad [\(gr)]")
        }
    }
    static let debugAppearance = ProcessInfo.processInfo.environment["SPLAT_DEBUG_APPEARANCE"] != nil

    /// The learned correction of each photo (row-major 3×3 + offset), e.g. for inspection.
    var appearanceCorrections: [[Float]] {
        guard let apParams else { return [] }
        let p = apParams.contents().bindMemory(to: Float.self, capacity: views.count * 12)
        return (0..<views.count).map { i in (0..<12).map { p[i * 12 + $0] } }
    }

    // MARK: per-view helpers

    private func uniforms(for v: TrainView, shDegree: Int) -> CameraUniforms {
        let c = v.camera
        var u = CameraUniforms.make(worldToCam: v.worldToCam, intr: SIMD4(c.fx, c.fy, c.cx, c.cy), width: v.width, height: v.height, count: n, shDegree: shDegree)
        u.flags = config.antialiased ? 1 : 0
        u.misc = SIMD4(1.0 / 255.0, 0, 0.3, 0)
        return u
    }

    private func rasterParams(for v: TrainView, bg: SIMD3<Float>) -> RasterParams {
        var rp = RasterParams()
        rp.width = UInt32(v.width); rp.height = UInt32(v.height)
        rp.tilesX = UInt32((v.width + 15) / 16); rp.tilesY = UInt32((v.height + 15) / 16)
        rp.bg = SIMD4(bg, 0)
        rp.absGrad = config.absGrad && config.strategy == .adaptive ? 1 : 0
        rp.hasDepth = (v.depth != nil && config.depthWeight > 0) ? 1 : 0
        rp.numGaussians = UInt32(n)
        rp.step = UInt32(step)
        rp.ssimWeight = config.ssimWeight
        rp.depthWeight = config.depthWeight
        rp.invNumPixels = 1 / Float(v.width * v.height)
        rp.pad = config.depthEdgeAware ? 1 : 0   // edge-aware depth flag (loss_grad)
        return rp
    }

    private var activeSHDegree: Int { min(config.shDegree, step / max(config.shDegreeInterval, 1)) }

    /// Forward pass up to rasterisation. Returns false if nothing is visible.
    private func encodeForward(view v: TrainView, u: inout CameraUniforms, rp: inout RasterParams) throws -> Bool {
        // Phase 1: preprocess + depth sort + count scan (needs CPU readback of the total).
        let cb = ctx.queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setBuffer(params[0], offset: 0, index: 0); enc.setBuffer(params[1], offset: 0, index: 1)
        enc.setBuffer(params[2], offset: 0, index: 2); enc.setBuffer(params[3], offset: 0, index: 3)
        enc.setBuffer(params[4], offset: 0, index: 4)
        enc.setBytes(&u, length: MemoryLayout<CameraUniforms>.stride, index: 5)
        enc.setBytes(&rp, length: MemoryLayout<RasterParams>.stride, index: 6)
        enc.setBuffer(p2d, offset: 0, index: 7); enc.setBuffer(conicO, offset: 0, index: 8)
        enc.setBuffer(rgbd, offset: 0, index: 9); enc.setBuffer(rects, offset: 0, index: 10)
        enc.setBuffer(tilesTouched, offset: 0, index: 11); enc.setBuffer(depthKeys, offset: 0, index: 12)
        enc.setBuffer(sortVals, offset: 0, index: 13); enc.setBuffer(alive, offset: 0, index: 14)
        enc.dispatch1D(ctx.pipeline("train_preprocess"), count: n)
        enc.memoryBarrier(scope: .buffers)
        sorter.encode(enc, keys: depthKeys, vals: sortVals, count: n, bits: 32)
        enc.setBuffer(sortVals, offset: 0, index: 0); enc.setBuffer(tilesTouched, offset: 0, index: 1)
        enc.setBuffer(counts, offset: 0, index: 2); enc.set(UInt32(n), 3)
        enc.dispatch1D(ctx.pipeline("gather_counts"), count: n)
        enc.memoryBarrier(scope: .buffers)
        scan.encode(enc, input: counts, output: offsets, count: n)
        enc.memoryBarrier(scope: .buffers)
        enc.setBuffer(offsets, offset: 0, index: 0); enc.setBuffer(counts, offset: 0, index: 1)
        enc.setBuffer(total, offset: 0, index: 2); enc.set(UInt32(n), 3)
        enc.dispatch1D(ctx.pipeline("total_count"), count: 1)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let e = cb.error { throw TrainerError.outOfMemory(e.localizedDescription) }
        let m = Int(total.contents().load(as: UInt32.self))
        guard m > 0 else { return false }
        if m > tileCapacity {
            let cap = Int(Double(m) * 1.3) + 4096
            tileKeys = ctx.buffer(cap * 4, label: "tileKeys"); tileVals = ctx.buffer(cap * 4, label: "tileVals")
            tileCapacity = cap
        }
        tileCount = m
        return true
    }
    private var tileCount = 0

    private func encodeRaster(_ enc: MTLComputeCommandEncoder, v: TrainView, rp: inout RasterParams, peak: MTLBuffer? = nil) {
        let m = tileCount
        let numTiles = Int(rp.tilesX * rp.tilesY)
        enc.setBuffer(sortVals, offset: 0, index: 0); enc.setBuffer(offsets, offset: 0, index: 1)
        enc.setBuffer(rects, offset: 0, index: 2); enc.setBuffer(tileKeys!, offset: 0, index: 3)
        enc.setBuffer(tileVals!, offset: 0, index: 4)
        enc.setBytes(&rp, length: MemoryLayout<RasterParams>.stride, index: 5)
        enc.set(UInt32(n), 6)
        enc.dispatch1D(ctx.pipeline("duplicate_keys"), count: n)
        enc.memoryBarrier(scope: .buffers)
        tileSorter.encode(enc, keys: tileKeys!, vals: tileVals!, count: m, bits: numTiles > 65536 ? 32 : 16)
        enc.setBuffer(ranges, offset: 0, index: 0); enc.set(UInt32(numTiles * 2), 1); enc.set(UInt32(0), 2)
        enc.dispatch1D(ctx.pipeline("fill_uint"), count: numTiles * 2)
        enc.memoryBarrier(scope: .buffers)
        enc.setBuffer(tileKeys!, offset: 0, index: 0); enc.setBuffer(ranges, offset: 0, index: 1); enc.set(UInt32(m), 2)
        enc.dispatch1D(ctx.pipeline("tile_ranges"), count: m)
        enc.memoryBarrier(scope: .buffers)
        enc.setComputePipelineState(ctx.pipeline(peak == nil ? "rasterize_forward" : "rasterize_peak"))
        enc.setBuffer(ranges, offset: 0, index: 0); enc.setBuffer(tileVals!, offset: 0, index: 1)
        enc.setBuffer(p2d, offset: 0, index: 2); enc.setBuffer(conicO, offset: 0, index: 3)
        if let peak { enc.setBuffer(peak, offset: 0, index: 9) }
        enc.setBuffer(rgbd, offset: 0, index: 4)
        enc.setBytes(&rp, length: MemoryLayout<RasterParams>.stride, index: 5)
        enc.setBuffer(img, offset: 0, index: 6); enc.setBuffer(finalT, offset: 0, index: 7)
        enc.setBuffer(nContrib, offset: 0, index: 8)
        enc.dispatchThreadgroups(MTLSize(width: Int(rp.tilesX), height: Int(rp.tilesY), depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.memoryBarrier(scope: .buffers)
    }

    private func dispatch2D(_ enc: MTLComputeCommandEncoder, _ name: String, _ w: Int, _ h: Int, _ d: Int = 1) {
        let pso = ctx.pipeline(name)
        enc.setComputePipelineState(pso)
        enc.dispatchThreads(MTLSize(width: w, height: h, depth: d), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
    }

    private func encodeBlur(_ enc: MTLComputeCommandEncoder, src: MTLBuffer, dst: MTLBuffer, planesCount: Int, rp: inout RasterParams, w: Int, h: Int) {
        var win = ssimWindow
        enc.setBuffer(src, offset: 0, index: 0); enc.setBuffer(tmp, offset: 0, index: 1)
        enc.setBytes(&rp, length: MemoryLayout<RasterParams>.stride, index: 2)
        enc.setBytes(&win, length: 44, index: 3)
        dispatch2D(enc, "blur_h", w, h, planesCount)
        enc.memoryBarrier(scope: .buffers)
        enc.setBuffer(tmp, offset: 0, index: 0); enc.setBuffer(dst, offset: 0, index: 1)
        dispatch2D(enc, "blur_v", w, h, planesCount)
        enc.memoryBarrier(scope: .buffers)
    }

    private func encodeLoss(_ enc: MTLComputeCommandEncoder, v: TrainView, rp: inout RasterParams) {
        let w = v.width, h = v.height
        enc.setBuffer(stats, offset: 0, index: 0); enc.set(UInt32(16), 1)
        enc.dispatch1D(ctx.pipeline("zero_float"), count: 16)
        enc.setBuffer(img, offset: 0, index: 0); enc.setBuffer(v.target, offset: 0, index: 1)
        enc.setBuffer(planes, offset: 0, index: 2)
        enc.setBytes(&rp, length: MemoryLayout<RasterParams>.stride, index: 3)
        dispatch2D(enc, "ssim_prep", w, h)
        enc.memoryBarrier(scope: .buffers)
        encodeBlur(enc, src: planes, dst: mu, planesCount: 15, rp: &rp, w: w, h: h)
        enc.setBuffer(mu, offset: 0, index: 0); enc.setBuffer(dmaps, offset: 0, index: 1)
        enc.setBuffer(stats, offset: 0, index: 2)
        enc.setBytes(&rp, length: MemoryLayout<RasterParams>.stride, index: 3)
        enc.setBuffer(planes, offset: 0, index: 4)
        dispatch2D(enc, "ssim_terms", w, h)
        enc.memoryBarrier(scope: .buffers)
    }

    private func encodeLossGrad(_ enc: MTLComputeCommandEncoder, v: TrainView, rp: inout RasterParams) {
        let w = v.width, h = v.height
        encodeBlur(enc, src: dmaps, dst: mu, planesCount: 9, rp: &rp, w: w, h: h)
        enc.setBuffer(planes, offset: 0, index: 0); enc.setBuffer(mu, offset: 0, index: 1)
        enc.setBuffer(img, offset: 0, index: 2); enc.setBuffer(gradImg, offset: 0, index: 3)
        enc.setBuffer(v.depth ?? emptyDepth, offset: 0, index: 4)
        var ds = v.depthSize
        enc.setBytes(&ds, length: 8, index: 5)
        enc.setBytes(&rp, length: MemoryLayout<RasterParams>.stride, index: 6)
        enc.setBuffer(v.target, offset: 0, index: 7)
        enc.setBuffer(stats, offset: 0, index: 8)
        dispatch2D(enc, "loss_grad", w, h)
        enc.memoryBarrier(scope: .buffers)
    }

    // MARK: training step

    private func nextView() -> TrainView {
        if orderPos >= order.count {
            order = Array(0..<trainViews.count)
            for i in stride(from: order.count - 1, to: 0, by: -1) { order.swapAt(i, Int(rng.next() % UInt64(i + 1))) }
            orderPos = 0
        }
        defer { orderPos += 1 }
        return trainViews[order[orderPos]]
    }

    private func positionLR() -> Float {
        let t = min(Float(step) / Float(max(config.iterations, 1)), 1)
        return exp(log(config.lrPosition) * (1 - t) + log(config.lrPositionFinal) * t) * sceneExtent
    }

    /// Runs one optimisation step. Returns (loss, psnr) of this step's view.
    @discardableResult
    func trainStep() throws -> (Float, Float) {
        paramLock.lock(); defer { paramLock.unlock() }
        let v = nextView()
        let bg: SIMD3<Float>
        switch config.background {
        case .black: bg = .zero
        case .white: bg = .one
        case .random: bg = SIMD3(rng.uniform(), rng.uniform(), rng.uniform())
        }
        let r = try forwardBackward(view: v, bg: bg, optimize: true)
        step += 1
        let td = CACurrentMediaTime()
        try densifyIfNeeded()
        if Self.profiling {
            profile("7 densify (CPU)", CACurrentMediaTime() - td)
            if step % 500 == 0 { dumpProfile() }
        }
        lastStats = r
        return r
    }

    /// Test hook: loss and analytic gradients for one view, parameters unchanged.
    func debugGradients(viewIndex: Int) throws -> (loss: Float, grads: [[Float]]) {
        paramLock.lock(); defer { paramLock.unlock() }
        let r = try forwardBackward(view: views[viewIndex], bg: .zero, optimize: false)
        let g = (0..<5).map { k in Array(UnsafeBufferPointer(start: ptr(grads[k]), count: n * Self.widths[k])) }
        return (r.0, g)
    }
    func debugLoss(viewIndex: Int) throws -> Float {
        paramLock.lock(); defer { paramLock.unlock() }
        return try forwardBackward(view: views[viewIndex], bg: .zero, optimize: false).0
    }
    func debugParamPointer(_ k: Int) -> UnsafeMutablePointer<Float> { ptr(params[k]) }
    func debugImage(_ c: Int) -> [Float] { let hw = views[0].width * views[0].height; return Array(UnsafeBufferPointer(start: ptr(img).advanced(by: c * hw), count: hw)) }
    func debugRanges() -> [UInt32] { Array(UnsafeBufferPointer(start: ranges.contents().bindMemory(to: UInt32.self, capacity: 24), count: 24)) }
    func debugTileInfo() -> (m: Int, keys: [UInt32], vals: [UInt32]) {
        let k = tileKeys!.contents().bindMemory(to: UInt32.self, capacity: tileCount), v = tileVals!.contents().bindMemory(to: UInt32.self, capacity: tileCount)
        return (tileCount, Array(UnsafeBufferPointer(start: k, count: tileCount)), Array(UnsafeBufferPointer(start: v, count: tileCount)))
    }

    // MARK: profiling (SPLAT_PROFILE=1)
    static let profiling = ProcessInfo.processInfo.environment["SPLAT_PROFILE"] != nil
    private var profileTotals: [String: (Double, Int)] = [:]
    private func profile(_ name: String, _ t: Double) {
        let key = name.components(separatedBy: " (").first ?? name
        let e = profileTotals[key] ?? (0, 0); profileTotals[key] = (e.0 + t, e.1 + 1)
    }
    private func dumpProfile() {
        print("  -- profile at step \(step), N=\(n), M=\(tileCount) (avg ms per step)")
        for k in profileTotals.keys.sorted() { let e = profileTotals[k]!; print(String(format: "     %-34@ %7.2f", k as NSString, e.0 / Double(max(step - step0, 1)) * 1000)) }
        profileTotals.removeAll(); step0 = step
    }
    private var step0 = 0

    private func forwardBackward(view v: TrainView, bg: SIMD3<Float>, optimize: Bool) throws -> (Float, Float) {
        var u = uniforms(for: v, shDegree: activeSHDegree)
        var rp = rasterParams(for: v, bg: bg)
        let tf0 = CACurrentMediaTime()
        guard try encodeForward(view: v, u: &u, rp: &rp) else { return (0, 0) }
        if Self.profiling { profile("1 preprocess+depth sort+scan", CACurrentMediaTime() - tf0) }
        var cb = ctx.queue.makeCommandBuffer()!
        var enc = cb.makeComputeCommandEncoder()!
        func split(_ name: String) {
            guard Self.profiling else { return }
            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            profile(name, cb.gpuEndTime - cb.gpuStartTime)
            cb = ctx.queue.makeCommandBuffer()!; enc = cb.makeComputeCommandEncoder()!
        }
        encodeRaster(enc, v: v, rp: &rp)
        let apIndex = config.appearanceCorrection ? viewIndex[ObjectIdentifier(v)] : nil
        if let ai = apIndex {
            enc.setBuffer(img, offset: 0, index: 0); enc.setBuffer(rawImg, offset: 0, index: 1)
            enc.setBuffer(apParams, offset: ai * 48, index: 2)
            enc.setBytes(&rp, length: MemoryLayout<RasterParams>.stride, index: 3)
            dispatch2D(enc, "appearance_apply", v.width, v.height)
            enc.memoryBarrier(scope: .buffers)
        }
        split("2 duplicate+tile sort+forward raster (M=\(tileCount))")
        encodeLoss(enc, v: v, rp: &rp)
        let robust = optimize && transientEnabled
        if robust {
            memset(residHist.contents(), 0, 256 * 4)
            memset(outlierCount.contents(), 0, 4)
            enc.setBuffer(planes, offset: 0, index: 0); enc.setBuffer(v.target, offset: 0, index: 1)
            enc.setBuffer(resid, offset: 0, index: 2); enc.setBuffer(residHist, offset: 0, index: 3)
            enc.setBytes(&rp, length: MemoryLayout<RasterParams>.stride, index: 4)
            enc.setBuffer(finalT, offset: 0, index: 5)
            dispatch2D(enc, "residual_hist", v.width, v.height)
            enc.memoryBarrier(scope: .buffers)
        }
        encodeLossGrad(enc, v: v, rp: &rp)
        if robust && step >= transientStartStep && transientThreshold > 0 {
            var thr = transientThreshold
            enc.setComputePipelineState(ctx.pipeline("transient_weights"))
            enc.setBuffer(resid, offset: 0, index: 0); enc.setBuffer(weights, offset: 0, index: 1)
            enc.setBytes(&rp, length: MemoryLayout<RasterParams>.stride, index: 2)
            enc.setBytes(&thr, length: 4, index: 3)
            enc.setBuffer(outlierCount, offset: 0, index: 4)
            enc.dispatchThreadgroups(MTLSize(width: Int(rp.tilesX), height: Int(rp.tilesY), depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            enc.memoryBarrier(scope: .buffers)
            enc.setBuffer(gradImg, offset: 0, index: 0); enc.setBuffer(weights, offset: 0, index: 1)
            enc.setBytes(&rp, length: MemoryLayout<RasterParams>.stride, index: 2)
            dispatch2D(enc, "apply_weights", v.width, v.height)
            enc.memoryBarrier(scope: .buffers)
        }
        if let ai = apIndex, optimize {
            memset(apGrad.contents(), 0, 48)
            enc.setBuffer(gradImg, offset: 0, index: 0); enc.setBuffer(rawImg, offset: 0, index: 1)
            enc.setBuffer(apParams, offset: ai * 48, index: 2)
            enc.setBytes(&rp, length: MemoryLayout<RasterParams>.stride, index: 3)
            enc.setBuffer(apGrad, offset: 0, index: 4)
            dispatch2D(enc, "appearance_backward", v.width, v.height)
            enc.memoryBarrier(scope: .buffers)
        }
        split("3 SSIM loss + grad")
        // backward raster
        enc.setBuffer(acc, offset: 0, index: 0); enc.set(UInt32(n * 12), 1)
        enc.dispatch1D(ctx.pipeline("zero_float"), count: n * 12)
        enc.memoryBarrier(scope: .buffers)
        enc.setComputePipelineState(ctx.pipeline("rasterize_backward"))
        enc.setBuffer(ranges, offset: 0, index: 0); enc.setBuffer(tileVals!, offset: 0, index: 1)
        enc.setBuffer(p2d, offset: 0, index: 2); enc.setBuffer(conicO, offset: 0, index: 3)
        enc.setBuffer(rgbd, offset: 0, index: 4)
        enc.setBytes(&rp, length: MemoryLayout<RasterParams>.stride, index: 5)
        enc.setBuffer(finalT, offset: 0, index: 6); enc.setBuffer(nContrib, offset: 0, index: 7)
        enc.setBuffer(gradImg, offset: 0, index: 8); enc.setBuffer(acc, offset: 0, index: 9)
        enc.dispatchThreadgroups(MTLSize(width: Int(rp.tilesX), height: Int(rp.tilesY), depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.memoryBarrier(scope: .buffers)
        split("4 backward raster")
        // preprocess backward
        var reg = SIMD4<Float>(0, 0, 0, 0)
        if config.strategy == .mcmc {
            reg.x = config.opacityReg / Float(max(n, 1))
            reg.y = config.scaleReg / Float(max(3 * n, 1))
        }
        reg.z = config.anisotropyReg / Float(max(n, 1)); reg.w = max(config.anisotropyMaxRatio, 1.5)
        for k in 0..<5 { enc.setBuffer(params[k], offset: 0, index: k) }
        enc.setBytes(&u, length: MemoryLayout<CameraUniforms>.stride, index: 5)
        enc.setBytes(&rp, length: MemoryLayout<RasterParams>.stride, index: 6)
        enc.setBuffer(acc, offset: 0, index: 7); enc.setBuffer(tilesTouched, offset: 0, index: 8)
        for k in 0..<5 { enc.setBuffer(grads[k], offset: 0, index: 9 + k) }
        enc.setBuffer(gradStats, offset: 0, index: 14)
        enc.setBytes(&reg, length: 16, index: 15)
        var big = SIMD2<Float>(config.bigSplatReg / Float(max(n, 1)), max(config.bigSplatMaxAngle, 0.01))
        enc.setBytes(&big, length: 8, index: 16)
        enc.dispatch1D(ctx.pipeline("preprocess_backward"), count: n)
        enc.memoryBarrier(scope: .buffers)
        split("5 preprocess backward")
        if !optimize {
            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            let s = stats.contents().bindMemory(to: Float.self, capacity: 16)
            let numel = Float(v.width * v.height * 3)
            let loss = (1 - config.ssimWeight) * s[0] / numel + config.ssimWeight * (1 - s[1] / numel) + config.depthWeight * s[3] / Float(v.width * v.height)
            return (loss, 0)
        }
        // Adam
        adamStep += 1
        let bc1 = 1 - pow(Float(0.9), Float(adamStep)), bc2 = 1 - pow(Float(0.999), Float(adamStep))
        let lrPos = positionLR()
        let lrs: [(Float, Float, UInt32, UInt32)] = [
            (lrPos, lrPos, 1, 1),
            (config.lrScale, config.lrScale, 1, 1),
            (config.lrRotation, config.lrRotation, 1, 1),
            (config.lrOpacity, config.lrOpacity, 1, 1),
            (config.lrColor, config.lrSHRest, 48, 3),
        ]
        let adam = ctx.pipeline("adam_step")
        for k in 0..<5 {
            var ap = AdamParams(lr: lrs[k].0, lr2: lrs[k].1, period: lrs[k].2, split: lrs[k].3, bc1: bc1, bc2: bc2, n: UInt32(n * Self.widths[k]))
            enc.setBuffer(params[k], offset: 0, index: 0); enc.setBuffer(grads[k], offset: 0, index: 1)
            enc.setBuffer(m1[k], offset: 0, index: 2); enc.setBuffer(m2[k], offset: 0, index: 3)
            enc.setBytes(&ap, length: MemoryLayout<AdamParams>.stride, index: 4)
            enc.dispatch1D(adam, count: n * Self.widths[k])
        }
        if config.strategy == .mcmc && config.noiseLR > 0 {
            enc.memoryBarrier(scope: .buffers)
            var np = SIMD4<Float>(config.noiseLR * lrPos, Float(bitPattern: UInt32(truncatingIfNeeded: rng.next())), Float(n), 0)
            for k in 0..<4 { enc.setBuffer(params[k], offset: 0, index: k) }
            enc.setBytes(&np, length: 16, index: 4)
            enc.dispatch1D(ctx.pipeline("mcmc_noise"), count: n)
        }
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if Self.profiling { profile("6 adam+noise", cb.gpuEndTime - cb.gpuStartTime) }
        if let e = cb.error { throw TrainerError.outOfMemory(e.localizedDescription) }
        let s = stats.contents().bindMemory(to: Float.self, capacity: 16)
        let numel = Float(v.width * v.height * 3)
        let l1 = s[0] / numel, ssim = s[1] / numel, mse = s[2] / numel
        let loss = (1 - config.ssimWeight) * l1 + config.ssimWeight * (1 - ssim)
        let psnr = mse > 0 ? -10 * log10(mse) : 99
        if Self.debugAppearance, !(loss.isFinite) { print("non-finite loss at step \(step): l1 \(l1) ssim \(ssim) mse \(mse) view \(apIndex ?? -1) size \(v.width)x\(v.height)") }
        if transientEnabled { updateTransientThreshold(pixels: v.width * v.height) }
        if let ai = apIndex, !v.holdout { updateAppearance(view: ai) }
        return (loss, psnr)
    }

    /// Percentile of this step's residual histogram, smoothed over steps (EMA) so one bad view can't
    /// swing it; used as the inlier threshold for the next step.
    /// Masking starts once textures have had time to converge (≥ 30 % of training).
    private var transientStartStep: Int { max(config.transientStart, Int(Double(config.iterations) * 0.3)) }

    private func updateTransientThreshold(pixels: Int) {
        let h = residHist.contents().bindMemory(to: UInt32.self, capacity: 256)
        var total: UInt64 = 0
        for i in 0..<256 { total += UInt64(h[i]) }
        guard total > 0 else { return }
        // Median residual of this view; transients are far above it, fine texture only modestly.
        let target = total / 2
        var acc: UInt64 = 0, bin = 255
        for i in 0..<256 { acc += UInt64(h[i]); if acc >= target { bin = i; break } }
        let median = (Float(bin) + 0.5) / 512
        // Floor so that a near-perfect fit never starts discarding ordinary noise.
        var tNew = max(median * config.transientSensitivity, 0.10)
        if step >= transientStartStep {
            let out = Float(outlierCount.contents().load(as: UInt32.self)) / Float(max(pixels, 1))
            transientFraction = transientFraction * 0.9 + out * 0.1
            // Safety valve: transients rarely cover more than a quarter of a frame; if far more is being
            // discarded the threshold is too tight for this scene, so relax it.
            if out > 0.25 { tNew = max(tNew, transientThreshold * 1.25) }
        }
        transientThreshold = transientThreshold < 0 ? tNew : transientThreshold * 0.95 + tNew * 0.05
        if ProcessInfo.processInfo.environment["SPLAT_DEBUG_TRANSIENT"] != nil && step % 250 == 0 {
            print(String(format: "    [transient] step %d median %.4f thr %.4f outliers(last) %.2f%%", step, median, transientThreshold,
                         Float(outlierCount.contents().load(as: UInt32.self)) * 100 / Float(max(pixels, 1))))
        }
    }

    // MARK: rendering for previews / evaluation

    /// Renders a training view with the current parameters; returns RGBA8 pixels.
    func render(view v: TrainView, background: SIMD3<Float> = .zero) throws -> (pixels: [UInt8], mse: Float)? {
        paramLock.lock(); defer { paramLock.unlock() }
        var u = uniforms(for: v, shDegree: activeSHDegree)
        var rp = rasterParams(for: v, bg: background)
        guard try encodeForward(view: v, u: &u, rp: &rp) else { return nil }
        let cb = ctx.queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        encodeRaster(enc, v: v, rp: &rp)
        // Training views: their own colour correction. Held-out views: the average one (what exports bake in),
        // so evaluation during training scores what the exported splat will look like. For a splat loaded from
        // a file (--eval) the average is the identity.
        if config.appearanceCorrection, let apParams {
            var buf: MTLBuffer? = nil, offset = 0
            if let ai = viewIndex[ObjectIdentifier(v)] { buf = apParams; offset = ai * 48 }
            else if let m = meanAppearance(), let hb = apHoldout {
                hb.contents().bindMemory(to: Float.self, capacity: 12).update(from: m, count: 12); buf = hb
            }
            if let buf {
                enc.memoryBarrier(scope: .buffers)
                enc.setBuffer(img, offset: 0, index: 0); enc.setBuffer(rawImg, offset: 0, index: 1)
                enc.setBuffer(buf, offset: offset, index: 2)
                enc.setBytes(&rp, length: MemoryLayout<RasterParams>.stride, index: 3)
                dispatch2D(enc, "appearance_apply", v.width, v.height)
                enc.memoryBarrier(scope: .buffers)
            }
        }
        encodeLoss(enc, v: v, rp: &rp)
        let out = ctx.buffer(v.width * v.height * 4)
        enc.setBuffer(img, offset: 0, index: 0); enc.setBuffer(out, offset: 0, index: 1)
        enc.setBytes(&rp, length: MemoryLayout<RasterParams>.stride, index: 2)
        dispatch2D(enc, "render_to_rgba8", v.width, v.height)
        enc.endEncoding()
        cb.commit(); cb.waitUntilCompleted()
        let s = stats.contents().bindMemory(to: Float.self, capacity: 16)
        let mse = s[2] / Float(v.width * v.height * 3)
        let px = Array(UnsafeBufferPointer(start: out.contents().bindMemory(to: UInt8.self, capacity: v.width * v.height * 4), count: v.width * v.height * 4))
        return (px, mse)
    }

    /// For every splat of `g`, the most it adds to any pixel of any of this dataset's photos (training and
    /// held-out). Below ~1/255 a splat makes no visible difference from any capture viewpoint.
    func peakContribution(cloud g: GaussianCloud, progress: ((Double) -> Void)? = nil) throws -> [Float] {
        load(g)
        step = config.shDegreeInterval * (config.shDegree + 1)
        let peak = ctx.buffer(max(n, 1) * 4, label: "peak")
        memset(peak.contents(), 0, max(n, 1) * 4)
        let all = trainViews + evalViews
        for (k, v) in all.enumerated() {
            paramLock.lock()
            var u = uniforms(for: v, shDegree: activeSHDegree)
            var rp = rasterParams(for: v, bg: .zero)
            if try encodeForward(view: v, u: &u, rp: &rp) {
                let cb = ctx.queue.makeCommandBuffer()!
                let enc = cb.makeComputeCommandEncoder()!
                encodeRaster(enc, v: v, rp: &rp, peak: peak)
                enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            }
            paramLock.unlock()
            progress?(Double(k + 1) / Double(max(all.count, 1)))
        }
        let p = peak.contents().bindMemory(to: Float.self, capacity: n)
        return Array(UnsafeBufferPointer(start: p, count: n))
    }

    /// Contribution census over every photo: each splat's largest α·T in any pixel (as `peakContribution`), and
    /// the number of photos in which it contributes at least `viewThreshold` somewhere. Real surfaces are drawn
    /// clearly by many photos; a blob hanging in front of a few cameras only by those few.
    func contributionCensus(cloud g: GaussianCloud, viewThreshold: Float,
                            progress: ((Double) -> Void)? = nil) throws -> (peak: [Float], views: [UInt32], photos: Int) {
        load(g)
        step = config.shDegreeInterval * (config.shDegree + 1)
        let m = max(n, 1)
        let peak = ctx.buffer(m * 4, label: "peak"), viewPeak = ctx.buffer(m * 4, label: "view peak"), views = ctx.buffer(m * 4, label: "views")
        memset(peak.contents(), 0, m * 4); memset(views.contents(), 0, m * 4)
        let all = trainViews + evalViews
        var thr = viewThreshold, nn = UInt32(n)
        for (k, v) in all.enumerated() {
            paramLock.lock()
            var u = uniforms(for: v, shDegree: activeSHDegree)
            var rp = rasterParams(for: v, bg: .zero)
            if try encodeForward(view: v, u: &u, rp: &rp) {
                let cb = ctx.queue.makeCommandBuffer()!
                let enc = cb.makeComputeCommandEncoder()!
                enc.setBuffer(viewPeak, offset: 0, index: 0); enc.set(UInt32(n), 1)
                enc.dispatch1D(ctx.pipeline("zero_float"), count: n)
                enc.memoryBarrier(scope: .buffers)
                encodeRaster(enc, v: v, rp: &rp, peak: viewPeak)
                enc.memoryBarrier(scope: .buffers)
                enc.setComputePipelineState(ctx.pipeline("peak_accumulate"))
                enc.setBuffer(viewPeak, offset: 0, index: 0); enc.setBuffer(peak, offset: 0, index: 1)
                enc.setBuffer(views, offset: 0, index: 2)
                enc.setBytes(&thr, length: 4, index: 3); enc.setBytes(&nn, length: 4, index: 4)
                enc.dispatch1D(ctx.pipeline("peak_accumulate"), count: n)
                enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            }
            paramLock.unlock()
            progress?(Double(k + 1) / Double(max(all.count, 1)))
        }
        return (Array(UnsafeBufferPointer(start: peak.contents().bindMemory(to: Float.self, capacity: n), count: n)),
                Array(UnsafeBufferPointer(start: views.contents().bindMemory(to: UInt32.self, capacity: n), count: n)), all.count)
    }

    /// Renders `g` from every dataset view (training and held out), e.g. to make a synthetic benchmark scene.
    func renderEach(cloud g: GaussianCloud, _ body: (_ camera: DatasetCamera, _ width: Int, _ height: Int, _ rgba: [UInt8]) throws -> Void) throws {
        load(g)
        step = config.shDegreeInterval * (config.shDegree + 1)
        for v in trainViews + evalViews {
            if let r = try render(view: v) { try body(v.camera, v.width, v.height, r.pixels) }
        }
    }

    /// Mean PSNR over the holdout views (or a sample of training views if there are none).
    /// Scores any splat (e.g. from another trainer) on this dataset's held-out photos with the same rasteriser
    /// and exact intrinsics, at full SH degree. Returns mean PSNR and per-view PSNR.
    func evaluate(cloud g: GaussianCloud) throws -> (mean: Float, perView: [Float], names: [String], failed: [String]) {
        load(g)
        step = config.shDegreeInterval * (config.shDegree + 1)
        let set = evalViews.isEmpty ? Array(trainViews.prefix(8)) : evalViews
        var per: [Float] = [], names: [String] = [], failed: [String] = []
        for v in set {
            if let r = try render(view: v) { per.append(r.mse > 0 ? -10 * log10(r.mse) : 99); names.append(v.camera.image) }
            else { failed.append(v.camera.image) }   // nothing rendered (e.g. every splat behind / outside the view)
        }
        return (per.isEmpty ? 0 : per.reduce(0, +) / Float(per.count), per, names, failed)
    }

    func evaluate() throws -> Float {
        let set = evalViews.isEmpty ? Array(trainViews.prefix(8)) : evalViews
        var sum: Float = 0, c = 0
        for v in set {
            if let r = try render(view: v) { sum += r.mse > 0 ? -10 * log10(r.mse) : 99; c += 1 }
        }
        return c > 0 ? sum / Float(c) : 0
    }

    // MARK: export

    func snapshot() -> GaussianCloud {
        paramLock.lock(); defer { paramLock.unlock() }
        var g = GaussianCloud()
        g.shDegree = config.shDegree
        g.antialiased = config.antialiased
        func arr(_ k: Int) -> [Float] {
            Array(UnsafeBufferPointer(start: params[k].contents().bindMemory(to: Float.self, capacity: n * Self.widths[k]), count: n * Self.widths[k]))
        }
        g.positions = arr(0); g.scales = arr(1); g.rotations = arr(2); g.opacities = arr(3); g.sh = arr(4)
        if let m = meanAppearance() { Self.bakeColourCorrection(m, into: &g.sh, count: n) }
        return g
    }

    /// colour' = M·colour + b applied to SH colours (colour = Σ b_k·sh_k + 0.5, sh stored [coef][rgb], 16 coefs):
    /// every coefficient → M·sh_k, and the DC term also gets (M·0.5 + b − 0.5) / C0. Exact for each splat's
    /// colour; for the composited image exact wherever the pixel is opaque.
    static func bakeColourCorrection(_ m: [Float], into sh: inout [Float], count: Int) {
        let c0: Float = 0.28209479177387814
        var off = [Float](repeating: 0, count: 3)
        for r in 0..<3 {
            let rowSum: Float = m[r * 3] + m[r * 3 + 1] + m[r * 3 + 2]
            off[r] = rowSum * 0.5 + m[9 + r] - 0.5
        }
        sh.withUnsafeMutableBufferPointer { s in
            for i in 0..<count {
                for k in 0..<16 {
                    let b = i * 48 + k * 3
                    let x = s[b], y = s[b + 1], z = s[b + 2]
                    for r in 0..<3 { s[b + r] = m[r * 3] * x + m[r * 3 + 1] * y + m[r * 3 + 2] * z }
                }
                for r in 0..<3 { s[i * 48 + r] += off[r] / c0 }
            }
        }
    }

    /// Builds a render scene directly from the training buffers (no CPU copy).
    func makeScene() -> SplatScene {
        paramLock.lock(); defer { paramLock.unlock() }
        return SplatScene(ctx: ctx, count: n, shDegree: config.shDegree, antialiased: config.antialiased,
                          pos: params[0], logScale: params[1], quat: params[2], opacity: params[3], sh: params[4])
    }

    // MARK: densification (CPU, unified memory)

    private func ptr(_ b: MTLBuffer) -> UnsafeMutablePointer<Float> { b.contents().bindMemory(to: Float.self, capacity: b.length / 4) }

    private func densifyIfNeeded() throws {
        let stop = config.densifyStop
        guard step >= config.densifyStart, step <= stop, step % max(config.densifyInterval, 1) == 0 else {
            if config.strategy == .adaptive, config.opacityResetInterval > 0, step % config.opacityResetInterval == 0, step > 0, step <= stop {
                resetOpacity()
            }
            return
        }
        switch config.strategy {
        case .mcmc: mcmcDensify()
        case .adaptive:
            adaptiveDensify()
            if config.opacityResetInterval > 0 && step % config.opacityResetInterval == 0 { resetOpacity() }
        }
        // Block training no longer kills gaussians that leave the block: the block's cameras also see
        // neighbouring blocks and background, and deleting those splats every pass made training
        // fight itself (PSNR fell while MCMC kept regrowing them). The region only seeds the
        // initial points and picks cameras; BlockMerger crops each block to the area it owns.
    }

    private static let binoms: [[Float]] = {
        let nMax = 51
        var b = [[Float]](repeating: [Float](repeating: 0, count: nMax), count: nMax)
        for i in 0..<nMax { b[i][0] = 1; if i > 0 { for k in 1...i { b[i][k] = b[i-1][k-1] + (k < i ? b[i-1][k] : 0) } } }
        return b
    }()

    /// 3DGS-MCMC relocation: splitting a gaussian into `ratio` copies that render like the original.
    private static func relocation(opacity o: Float, ratio: Int) -> (newOpacity: Float, scaleCoeff: Float) {
        let r = min(max(ratio, 1), 51)
        let no = min(max(1 - pow(1 - o, 1 / Float(r)), 0.005), 1 - 1e-6)
        var denom: Float = 0
        for i in 1...r {
            for k in 0..<i {
                let term = (k % 2 == 0 ? 1 : -1) / sqrt(Float(k + 1)) * pow(no, Float(k + 1))
                denom += binoms[i - 1][k] * term
            }
        }
        return (no, o / denom)
    }

    private func sampleWeighted(_ weights: [Float], count: Int) -> [Int] {
        var cdf = [Float](repeating: 0, count: weights.count)
        var s: Float = 0
        for i in 0..<weights.count { s += max(weights[i], 0); cdf[i] = s }
        guard s > 0 else { return [] }
        var out: [Int] = []; out.reserveCapacity(count)
        for _ in 0..<count {
            let x = rng.uniform() * s
            var lo = 0, hi = cdf.count - 1
            while lo < hi { let mid = (lo + hi) / 2; if cdf[mid] < x { lo = mid + 1 } else { hi = mid } }
            out.append(lo)
        }
        return out
    }

    private func copyGaussian(from s: Int, to d: Int) {
        for k in 0..<5 {
            let w = Self.widths[k]
            let p = ptr(params[k])
            for j in 0..<w { p[d*w + j] = p[s*w + j] }
        }
    }

    private func zeroMoments(_ i: Int) {
        for k in 0..<5 {
            let w = Self.widths[k]
            let a = ptr(m1[k]), b = ptr(m2[k])
            for j in 0..<w { a[i*w + j] = 0; b[i*w + j] = 0 }
        }
    }

    private func applyRelocation(sources: [Int]) {
        var counts: [Int: Int] = [:]
        for s in sources { counts[s, default: 0] += 1 }
        let op = ptr(params[3]), sc = ptr(params[1])
        for (s, c) in counts {
            let o = 1 / (1 + exp(-op[s]))
            let (no, coeff) = Self.relocation(opacity: o, ratio: c + 1)
            op[s] = GaussianCloud.invSigmoid(no)
            let lc = log(max(coeff, 1e-8))
            for j in 0..<3 { sc[s*3 + j] += lc }
            zeroMoments(s)
        }
    }

    /// ImprovedGS / IGS+ style split: a source and its new copies (m members) are spread evenly along the
    /// source's longest axis (spacing 2σ/m, centred on the source) and only that axis is shortened by √m,
    /// instead of all members sitting on the same spot. Opacity keeps the MCMC relocation formula.
    private func spreadAlongLongAxis(sources: [Int], copies: [Int]) {
        var groups: [Int: [Int]] = [:]
        for (k, s) in sources.enumerated() { groups[s, default: [s]].append(copies[k]) }
        let p = ptr(params[0]), sc = ptr(params[1]), q = ptr(params[2])
        for (s, members) in groups {
            let m = members.count
            guard m > 1 else { continue }
            let ls = SIMD3(sc[s*3], sc[s*3+1], sc[s*3+2])
            let ax = ls.x >= ls.y ? (ls.x >= ls.z ? 0 : 2) : (ls.y >= ls.z ? 1 : 2)
            let sigma = exp(ls[ax])
            // rotation matrix column `ax` of the (w, x, y, z) quaternion = the long axis in world space
            var qw = q[s*4], qx = q[s*4+1], qy = q[s*4+2], qz = q[s*4+3]
            let ql = max(sqrt(qw*qw + qx*qx + qy*qy + qz*qz), 1e-12); qw /= ql; qx /= ql; qy /= ql; qz /= ql
            let u: SIMD3<Float>
            switch ax {
            case 0: u = SIMD3(1 - 2*(qy*qy + qz*qz), 2*(qx*qy + qw*qz), 2*(qx*qz - qw*qy))
            case 1: u = SIMD3(2*(qx*qy - qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz + qw*qx))
            default: u = SIMD3(2*(qx*qz + qw*qy), 2*(qy*qz - qw*qx), 1 - 2*(qx*qx + qy*qy))
            }
            let centre = SIMD3(p[s*3], p[s*3+1], p[s*3+2])
            let shrink = log(1 / sqrt(Float(m)))
            for (j, g) in members.enumerated() {
                let t = (Float(j) - Float(m - 1) / 2) * 2 * sigma / Float(m)
                let pos = centre + u * t
                p[g*3] = pos.x; p[g*3+1] = pos.y; p[g*3+2] = pos.z
                sc[g*3 + ax] = ls[ax] + shrink
            }
        }
    }

    private func mcmcDensify() {
        let op = ptr(params[3])
        var dead: [Int] = [], weights = [Float](repeating: 0, count: n)
        for i in 0..<n {
            let o = 1 / (1 + exp(-op[i]))
            if o <= config.pruneOpacity { dead.append(i) } else { weights[i] = o }
        }
        // relocate dead → sampled alive
        if !dead.isEmpty {
            let src = sampleWeighted(weights, count: dead.count)
            if src.count == dead.count {
                applyRelocation(sources: src)
                for (k, d) in dead.enumerated() { copyGaussian(from: src[k], to: d); zeroMoments(d) }
                if config.longAxisSplit { spreadAlongLongAxis(sources: src, copies: dead) }
            }
        }
        // grow by 5 % up to the cap
        // 5 % per pass (3DGS-MCMC), or faster when that pace could not reach the budget before
        // densification ends (sparse initialisations).
        let passesLeft = max(1, (config.densifyStop - step) / max(config.densifyInterval, 1))
        let linear = n + (config.maxGaussians - n) / passesLeft
        let target = min(config.maxGaussians, max(Int(Double(n) * 1.05), min(linear, n * 2)))
        let add = target - n
        guard add > 0 else { return }
        var w2 = [Float](repeating: 0, count: n)
        for i in 0..<n { w2[i] = 1 / (1 + exp(-op[i])) }
        let src = sampleWeighted(w2, count: add)
        guard !src.isEmpty else { return }
        applyRelocation(sources: src)
        let newN = n + src.count
        grow(to: newN)
        for (k, s) in src.enumerated() { copyGaussian(from: s, to: n + k); zeroMoments(n + k) }
        memset(alive.contents().advanced(by: n), 1, src.count)
        let base = n
        n = newN
        if config.longAxisSplit { spreadAlongLongAxis(sources: src, copies: (0..<src.count).map { base + $0 }) }
    }

    /// Ensures buffers can hold `count` gaussians, preserving contents.
    private func grow(to count: Int) {
        guard count > capacity else { return }
        let oldParams = params, oldM1 = m1, oldM2 = m2, oldAlive = alive!, oldStats = gradStats!
        let oldN = n
        allocGaussianBuffers(Int(Double(count) * 1.2))
        for k in 0..<5 {
            let bytes = oldN * Self.widths[k] * 4
            memcpy(params[k].contents(), oldParams[k].contents(), bytes)
            memcpy(m1[k].contents(), oldM1[k].contents(), bytes)
            memcpy(m2[k].contents(), oldM2[k].contents(), bytes)
        }
        memcpy(alive.contents(), oldAlive.contents(), oldN)
        memcpy(gradStats.contents(), oldStats.contents(), oldN * 12)
    }

    /// Rebuilds all gaussian arrays from a list of source indices (used by ADC prune/split).
    private func gather(_ src: [Int], fresh: [Bool]) {
        let newN = src.count
        var newParams: [[Float]] = [], newM1: [[Float]] = [], newM2: [[Float]] = []
        for k in 0..<5 {
            let w = Self.widths[k]
            let p = ptr(params[k]), a = ptr(m1[k]), b = ptr(m2[k])
            var np = [Float](repeating: 0, count: newN * w), na = np, nb = np
            for (i, s) in src.enumerated() {
                for j in 0..<w { np[i*w + j] = p[s*w + j] }
                if !fresh[i] { for j in 0..<w { na[i*w + j] = a[s*w + j]; nb[i*w + j] = b[s*w + j] } }
            }
            newParams.append(np); newM1.append(na); newM2.append(nb)
        }
        if newN > capacity { allocGaussianBuffers(newN) }
        for k in 0..<5 {
            newParams[k].withUnsafeBytes { memcpy(params[k].contents(), $0.baseAddress!, $0.count) }
            newM1[k].withUnsafeBytes { memcpy(m1[k].contents(), $0.baseAddress!, $0.count) }
            newM2[k].withUnsafeBytes { memcpy(m2[k].contents(), $0.baseAddress!, $0.count) }
        }
        memset(alive.contents(), 1, max(newN, 1))
        memset(gradStats.contents(), 0, gradStats.length)
        n = newN
    }

    private func adaptiveDensify() {
        let gs = ptr(gradStats), sc = ptr(params[1]), op = ptr(params[3]), q = ptr(params[2]), p = ptr(params[0])
        let thr = config.absGrad ? config.densifyGradThreshold * 4 : config.densifyGradThreshold
        let percentDense: Float = 0.01 * sceneExtent
        var src: [Int] = [], fresh: [Bool] = []
        var splits: [(Int, Int)] = []  // (new index, source)
        src.reserveCapacity(n + n / 10)
        let budget = config.maxGaussians
        for i in 0..<n {
            let o = 1 / (1 + exp(-op[i]))
            let maxS = exp(max(sc[i*3], max(sc[i*3+1], sc[i*3+2])))
            let tooBig = step > config.opacityResetInterval && (gs[i*3+2] > 0.1 || maxS > 0.1 * sceneExtent)
            if o < config.pruneOpacity || tooBig { continue }
            let g = gs[i*3+1] > 0 ? gs[i*3] / gs[i*3+1] : 0
            let canGrow = src.count + (n - i) < budget
            if g >= thr && canGrow {
                if maxS <= percentDense {
                    src.append(i); fresh.append(false)
                    src.append(i); fresh.append(true)              // clone
                } else {
                    splits.append((src.count, i)); src.append(i); fresh.append(true)
                    splits.append((src.count, i)); src.append(i); fresh.append(true)
                }
            } else {
                src.append(i); fresh.append(false)
            }
        }
        // Capture source values before gather overwrites buffers.
        var splitData: [(Int, SIMD3<Float>, SIMD3<Float>, SIMD4<Float>)] = []
        for (ni, s) in splits {
            splitData.append((ni, SIMD3(p[s*3], p[s*3+1], p[s*3+2]), SIMD3(sc[s*3], sc[s*3+1], sc[s*3+2]), SIMD4(q[s*4], q[s*4+1], q[s*4+2], q[s*4+3])))
        }
        gather(src, fresh: fresh)
        let np = ptr(params[0]), ns = ptr(params[1])
        for (ni, pos, ls, qq) in splitData {
            let s = SIMD3(exp(ls.x), exp(ls.y), exp(ls.z))
            let qn = simd_normalize(simd_quatf(ix: qq.y, iy: qq.z, iz: qq.w, r: qq.x))
            let e = SIMD3(rng.normal(), rng.normal(), rng.normal()) * s
            let d = qn.act(e)
            np[ni*3] = pos.x + d.x; np[ni*3+1] = pos.y + d.y; np[ni*3+2] = pos.z + d.z
            let l16 = log(Float(1.6))
            ns[ni*3] = ls.x - l16; ns[ni*3+1] = ls.y - l16; ns[ni*3+2] = ls.z - l16
        }
    }

    private func resetOpacity() {
        let op = ptr(params[3])
        let cap = GaussianCloud.invSigmoid(0.01)
        for i in 0..<n { op[i] = min(op[i], cap) }
        let a = ptr(m1[3]), b = ptr(m2[3])
        for i in 0..<n { a[i] = 0; b[i] = 0 }
    }

    // MARK: full run

    /// Runs the whole optimisation. `onStatus` is called ~10×/s, `onPreview` every `previewEvery` steps.
    func run(previewEvery: Int = 200,
             onStatus: @escaping (TrainerStatus) -> Void,
             onPreview: ((SplatScene) -> Void)? = nil) throws -> GaussianCloud {
        let start = CACurrentMediaTime()
        var lastReport = start
        var status = TrainerStatus(totalSteps: config.iterations)
        var emaLoss: Float = 0, emaPSNR: Float = 0
        while step < config.iterations {
            if isCancelled { throw TrainerError.cancelled }
            while isPaused && !isCancelled { Thread.sleep(forTimeInterval: 0.05) }
            let (l, p) = try trainStep()
            emaLoss = step <= 1 ? l : emaLoss * 0.95 + l * 0.05
            emaPSNR = step <= 1 ? p : emaPSNR * 0.95 + p * 0.05
            if config.evalInterval > 0 && step % config.evalInterval == 0 {
                status.evalPSNR = try evaluate()
            }
            if let onPreview, previewEvery > 0, step % previewEvery == 0 { onPreview(makeScene()) }
            let now = CACurrentMediaTime()
            if now - lastReport > 0.1 || step == config.iterations {
                lastReport = now
                status.step = step
                status.loss = emaLoss
                status.psnr = emaPSNR
                status.numGaussians = n
                status.elapsed = now - start
                status.itersPerSecond = Double(step) / max(status.elapsed, 1e-6)
                status.eta = Double(config.iterations - step) / max(status.itersPerSecond, 1e-6)
                status.shDegree = activeSHDegree
                status.transientFraction = transientFraction
                status.phase = step < config.densifyStop ? "Optimising & densifying" : "Refining"
                onStatus(status)
            }
        }
        if config.evalInterval > 0 { status.evalPSNR = try evaluate(); onStatus(status) }
        var result = snapshot()
        if config.strategy == .mcmc {
            // Drop gaussians that ended up dead.
            result = result.filtered((0..<result.count).map { result.opacity($0) > config.pruneOpacity })
        }
        return result
    }
}
