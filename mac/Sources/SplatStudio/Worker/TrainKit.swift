import Foundation
import Metal
import os
import simd

// Public facade over the Metal trainer for the worker server (Worker/WorkerEngine.swift). On iOS this is the
// SplatTrainKit module's API (the trainer sources stay internal there, in Swift 5 mode); in SplatWork it is
// compiled into the app and used by "act as a worker" (MacWorker).

/// Progress of a running training job (mirrors the Mac's `TrainerStatus`).
public struct TrainKitProgress: Sendable, Equatable {
    public var step: Int
    public var totalSteps: Int
    public var loss: Double
    public var psnr: Double
    public var evalPsnr: Double?
    public var numGaussians: Int
    public var elapsed: Double
    public var eta: Double
    public var phase: String
}

public enum TrainKitError: LocalizedError, Sendable {
    case invalidDataset(String)
    case invalidConfig(String)
    case cancelled
    public var errorDescription: String? {
        switch self {
        case .invalidDataset(let s): return "invalid dataset: \(s)"
        case .invalidConfig(let s): return "invalid config: \(s)"
        case .cancelled: return "cancelled"
        }
    }
}

/// What the memory check decided for a job.
public struct TrainKitBudget: Sendable, Equatable {
    public var requestedGaussians: Int
    public var maxGaussians: Int
    public var requestedImageSize: Int
    public var maxImageSize: Int
    public var cameras: Int
    public var availableBytes: UInt64
    public var estimatedBytes: UInt64
    public var summary: String
}

public enum TrainKit {
    /// Memory a training job may still allocate: `os_proc_available_memory()` on iOS (the app's jetsam
    /// limit), the GPU's recommended working set on a Mac (unified memory, minus what is already in use).
    public static var availableMemory: UInt64 {
        #if os(iOS)
        let a = UInt64(os_proc_available_memory())
        // 0 in the Simulator (no jetsam limit there): assume an 8 GB phone's typical app budget
        return a > 0 ? a : min(ProcessInfo.processInfo.physicalMemory / 2, 5 << 30)
        #else
        guard let d = MTLCreateSystemDefaultDevice() else { return ProcessInfo.processInfo.physicalMemory / 2 }
        let ws = d.recommendedMaxWorkingSetSize, used = UInt64(d.currentAllocatedSize)
        return ws > used ? ws - used : 0
        #endif
    }

    /// GPU name, e.g. "Apple A18 Pro".
    public static var gpuName: String {
        guard let d = MTLCreateSystemDefaultDevice() else { return "Apple GPU" }
        var n = d.name
        if n.hasSuffix(" GPU") { n.removeLast(4) }
        return n
    }

    /// Checks that `root/dataset.json` parses and every training image exists. Returns the camera count.
    public static func validateDataset(root: URL) throws -> Int {
        let ds: PreparedDataset
        do { ds = try PreparedDataset.load(root) } catch { throw TrainKitError.invalidDataset("dataset.json: \(error.localizedDescription)") }
        guard !ds.manifest.cameras.isEmpty else { throw TrainKitError.invalidDataset("no cameras") }
        let fm = FileManager.default
        if let missing = ds.manifest.cameras.first(where: { !fm.fileExists(atPath: ds.url($0.image).path) }) {
            throw TrainKitError.invalidDataset("missing image \(missing.image)")
        }
        return ds.manifest.cameras.count
    }

    /// `GET /v1/jobs/{id}/cameras` body: the dataset's cameras (same coordinates as the result's splats).
    public static func camerasJSON(root: URL) throws -> Data {
        let ds = try PreparedDataset.load(root)
        struct Body: Encodable { var cameras: [DatasetCamera]; var sceneExtent: Float; var upVector: [Float] }
        return try JSONEncoder().encode(Body(cameras: ds.manifest.cameras, sceneExtent: ds.manifest.sceneExtent, upVector: ds.manifest.upVector))
    }

    /// Iterations a config asks for (for `totalSteps` before the job starts).
    public static func iterations(configJSON: Data?) -> Int {
        (try? decodeConfig(configJSON))?.iterations ?? TrainingConfig().iterations
    }

    static func decodeConfig(_ data: Data?) throws -> TrainingConfig {
        guard let data, !data.isEmpty, data != Data("null".utf8) else { return TrainingConfig() }
        do { return try JSONDecoder().decode(TrainingConfig.self, from: data) } catch {
            throw TrainKitError.invalidConfig(error.localizedDescription)
        }
    }

    // MARK: memory budget

    /// Bytes per gaussian while training: 59 parameters × (value, gradient, Adam m, Adam v) × 4 B, the
    /// per-splat scratch (projection, sort keys, accumulators), tile instances, the 10 % growth headroom and
    /// the CPU-side copy made at densification / export, rounded up.
    static let bytesPerGaussian: Double = 2000
    /// Per-pixel training buffers (render, gradients, SSIM planes …) for the largest view.
    static let bytesPerTrainPixel: Double = 280
    /// Fixed overhead: shader library, command buffers, app, network buffers.
    static let fixedBytes: Double = 250 * 1024 * 1024

    /// Clamps `maxGaussians` (and, when even a modest budget wouldn't fit, `maxImageSize`) to what fits in
    /// `availableBytes` (`os_proc_available_memory()` at job start).
    static func budget(dataset: PreparedDataset, config: inout TrainingConfig, availableBytes: UInt64) -> TrainKitBudget {
        let cams = dataset.manifest.cameras.filter { c in config.cameraIds.map { Set($0).contains(c.id) } ?? true }
        let requestedG = config.maxGaussians, requestedS = config.maxImageSize
        let usable = Double(availableBytes) * 0.80
        func imageBytes(_ maxSize: Int) -> (all: Double, largest: Double) {
            var all = 0.0, largest = 0.0
            for c in cams {
                let longest = max(c.width, c.height)
                let s = longest > maxSize ? Double(maxSize) / Double(longest) : 1
                let px = Double(c.width) * s * Double(c.height) * s
                // RGBA target (+ coarse levels / masks ≈ 40 %), depth maps are small
                all += px * 4 * 1.4 + Double((c.depthWidth ?? 0) * (c.depthHeight ?? 0) * 4)
                largest = max(largest, px)
            }
            // per-photo bilateral grid (GPU + 2 CPU Adam moments)
            if config.bilateralGrid { all += Double(cams.count) * Double(12 * 8 * 16 * 16 * 4 * 3) }
            return (all, largest)
        }
        var size = config.maxImageSize
        var img = imageBytes(size)
        let minGaussians = 150_000
        func gaussianRoom() -> Double { usable - fixedBytes - img.all - img.largest * bytesPerTrainPixel }
        while gaussianRoom() < Double(minGaussians) * bytesPerGaussian && size > 800 {
            size = max(800, Int(Double(size) * 0.85))
            img = imageBytes(size)
        }
        config.maxImageSize = size
        let fit = max(50_000, Int(gaussianRoom() / bytesPerGaussian))
        config.maxGaussians = min(requestedG, fit)
        // keep random init / densification counts consistent with the new cap
        config.randomInitCount = min(config.randomInitCount, max(10_000, config.maxGaussians / 2))
        let est = fixedBytes + img.all + img.largest * bytesPerTrainPixel + Double(config.maxGaussians) * bytesPerGaussian
        let mb = { (b: Double) in String(format: "%.0f MB", b / 1_048_576) }
        var s = "memory: \(mb(Double(availableBytes))) available to the app, \(cams.count) cameras; "
        s += "maxGaussians \(requestedG)" + (config.maxGaussians != requestedG ? " → \(config.maxGaussians)" : " (fits)")
        if size != requestedS { s += ", maxImageSize \(requestedS) → \(size)" }
        s += ", estimated peak \(mb(est))"
        return TrainKitBudget(requestedGaussians: requestedG, maxGaussians: config.maxGaussians, requestedImageSize: requestedS,
                              maxImageSize: size, cameras: cams.count, availableBytes: availableBytes,
                              estimatedBytes: UInt64(max(0, est)), summary: s)
    }
}

/// One training job on this device. `run` blocks (call it from a background queue); `cancel` is thread-safe.
public final class TrainKitJob: @unchecked Sendable {
    public let datasetRoot: URL
    public private(set) var budget: TrainKitBudget
    private var config: TrainingConfig
    private let dataset: PreparedDataset
    private let lock = NSLock()
    private var trainer: MetalTrainer?
    private var cancelled = false

    /// - Parameters:
    ///   - configJSON: the job's `TrainingConfig` as sent by SplatWork (tolerant decoding, missing keys = defaults).
    ///   - availableBytes: memory the app may still allocate (`os_proc_available_memory()`).
    public init(datasetRoot: URL, configJSON: Data?, availableBytes: UInt64) throws {
        self.datasetRoot = datasetRoot
        var cfg = try TrainKit.decodeConfig(configJSON)
        var ds: PreparedDataset
        do { ds = try PreparedDataset.load(datasetRoot) } catch { throw TrainKitError.invalidDataset(error.localizedDescription) }
        // Same as the Mac's local block training (ComputeScheduler → dataset.restricted(to:)): the manifest is
        // limited to the block's cameras and the scene bounds recomputed; the trainer itself also filters by
        // config.cameraIds and keeps / densifies only gaussians inside config.region.
        if let ids = cfg.cameraIds {
            let keep = Set(ids)
            ds.manifest.cameras = ds.manifest.cameras.filter { keep.contains($0.id) }
            guard !ds.manifest.cameras.isEmpty else { throw TrainKitError.invalidConfig("cameraIds match no camera of the dataset") }
            ds.recomputeSceneBounds()
        }
        if let r = cfg.region, r.min.count != 3 || r.max.count != 3 { throw TrainKitError.invalidConfig("region needs 3-element min and max") }
        budget = TrainKit.budget(dataset: ds, config: &cfg, availableBytes: availableBytes)
        config = cfg
        dataset = ds
    }

    public var totalSteps: Int { config.iterations }

    public func cancel() {
        lock.lock(); cancelled = true; let t = trainer; lock.unlock()
        t?.cancel()
    }

    private var isCancelled: Bool { lock.lock(); defer { lock.unlock() }; return cancelled }

    /// Loads the images, trains, and writes the result (INRIA 3DGS PLY) to `output`. Returns the splat count.
    @discardableResult
    public func run(output: URL, onPhase: (String) -> Void, onProgress: @escaping (TrainKitProgress) -> Void) throws -> Int {
        let t = MetalTrainer(dataset: dataset, config: config)
        lock.lock(); trainer = t; let c = cancelled; lock.unlock()
        if c { throw TrainKitError.cancelled }
        do {
            try t.prepare { onPhase($0) }
            if isCancelled { throw TrainKitError.cancelled }
            onPhase("Training")
            let cloud = try t.run(previewEvery: 0, onStatus: { s in
                onProgress(TrainKitProgress(step: s.step, totalSteps: s.totalSteps, loss: Double(s.loss), psnr: Double(s.psnr),
                                            evalPsnr: s.evalPSNR.map(Double.init), numGaussians: s.numGaussians,
                                            elapsed: s.elapsed, eta: s.eta, phase: s.phase))
            })
            onPhase("Writing result")
            let up = dataset.manifest.upVector.count == 3
                ? SIMD3<Float>(dataset.manifest.upVector[0], dataset.manifest.upVector[1], dataset.manifest.upVector[2])
                : Orientation.standardUp
            try FileManager.default.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
            try PLYCodec.writeGaussians(cloud, to: output, shDegree: config.shDegree, up: up)
            lock.lock(); trainer = nil; lock.unlock()
            return cloud.count
        } catch TrainerError.cancelled {
            lock.lock(); trainer = nil; lock.unlock()
            throw TrainKitError.cancelled
        } catch {
            lock.lock(); trainer = nil; lock.unlock()
            if isCancelled { throw TrainKitError.cancelled }
            throw error
        }
    }
}
