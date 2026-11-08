import Foundation
import Metal
import Observation
import os
import simd

// Public facade for "capture → trained splat" on the iPhone, built from the Mac pipeline compiled into this
// module (CaptureImporter, FrameMasker, WorkerClient / WorkerBrowser, ComputeScheduler, the Metal trainer).
//
// Works with nothing else on the network: preparation (frames + ARKit poses → dataset), Apple Vision masks and
// training all run on the phone. Steps that need Python (learned masks, sky, pose refinement) only run when a
// SplatWork Node that can do them is reachable, and are skipped (with a note) otherwise. Training can be
// handed to Macs / PCs / other iPhones running a worker, whole or as blocks spread over several devices.

// MARK: - Options

public enum SplatQuality: String, CaseIterable, Identifiable, Sendable, Codable {
    case preview, balanced, high
    public var id: String { rawValue }
    public var title: String {
        switch self {
        case .preview: return "Quick"
        case .balanced: return "Balanced"
        case .high: return "High"
        }
    }
    public var subtitle: String {
        switch self {
        case .preview: return "5k steps · ≈10–20 min on iPhone, 2–4 min on a Mac"
        case .balanced: return "15k steps · ≈45 min+ on iPhone, ~10 min on a Mac"
        case .high: return "30k steps · best on a Mac or PC"
        }
    }
    var preset: TrainingConfig.Preset {
        switch self {
        case .preview: return .preview
        case .balanced: return .balanced
        case .high: return .high
        }
    }
}

/// On-device (Apple Vision) masks.
public enum SplatMaskMode: String, CaseIterable, Identifiable, Sendable, Codable {
    case none, removePeople, removePeopleAndPets, keepObject
    public var id: String { rawValue }
    public var title: String {
        switch self {
        case .none: return "No masks"
        case .removePeople: return "Remove people"
        case .removePeopleAndPets: return "Remove people & pets"
        case .keepObject: return "Keep main object only"
        }
    }
    var model: MaskingModel {
        switch self {
        case .none: return .none
        case .removePeople: return .appleVisionPerson
        case .removePeopleAndPets: return .appleVisionDynamic
        case .keepObject: return .appleVisionForeground
        }
    }
}

public struct SplatMakeOptions: Sendable, Codable, Equatable {
    public var quality: SplatQuality = .preview
    public var masks: SplatMaskMode = .none
    /// Learned moving-object masks (SAM 2 based `dynamic` model) — on a Linux worker only.
    public var learnedMovingMasks: Bool = false
    /// Sky masks + learned skybox — on a Linux worker only.
    public var sky: Bool = false
    /// Polish the ARKit poses with a quick SfM refinement — on a Linux worker only.
    public var refinePoses: Bool = true
    /// Training devices: `SplatDeviceInfo.localId` and/or worker ids. Empty = this iPhone.
    public var deviceIds: Set<String> = [SplatDeviceInfo.localId]
    /// Worker for the Python steps (learned masks, sky, pose refinement). nil = a ticked device that can, else
    /// any online one.
    public var processorId: String? = nil
    /// Tests: fewer training steps than the quality preset / every Nth frame only.
    public var iterationsOverride: Int? = nil
    public var frameStride: Int = 1
    public init() {}
}

// MARK: - Devices

public struct SplatDeviceInfo: Identifiable, Hashable, Sendable {
    public static let localId = "local"
    public enum Kind: String, Sendable { case thisPhone, mac, pc, iphone }
    public var id: String
    public var name: String
    public var kind: Kind
    /// GPU / backend / status line.
    public var detail: String
    public var isOnline: Bool
    public var canTrain: Bool
    /// Runs splatprep (learned masks, sky, pose refinement): Linux worker.
    public var canProcess: Bool
    public var isLocal: Bool { id == Self.localId }
}

/// Discovered training devices: this iPhone plus SplatWork Nodes / Macs / iPhones on the local network.
@MainActor
@Observable
public final class SplatDevices {
    public static let shared = SplatDevices()
    private let browser = WorkerBrowser.shared
    /// Bonjour names to leave out (this iPhone's own Compute worker).
    public var excludedNames: Set<String> {
        get { browser.ownNames }
        set { if browser.ownNames != newValue { browser.ownNames = newValue } }
    }

    public init() {}

    public func start() { browser.start(healthInterval: 10) }
    public func stop() { browser.stop() }
    public func refresh() async { await browser.refreshHealth() }
    public var isBrowsing: Bool { browser.isBrowsing }
    public var browserError: String? { browser.browserError }

    /// Adds a worker by address (when Bonjour can't see it, e.g. another subnet / VPN).
    @discardableResult
    public func addManual(host: String, port: Int = 7878) -> String {
        let ep = browser.addManual(host: host, port: port)
        Task { await browser.checkHealth(ep.id) }
        return ep.id
    }

    public var thisPhone: SplatDeviceInfo {
        SplatDeviceInfo(id: SplatDeviceInfo.localId, name: "This iPhone", kind: .thisPhone,
                        detail: "\(TrainKit.gpuName) · \(ByteCountFormatter.string(fromByteCount: Int64(TrainKit.availableMemory), countStyle: .memory)) free",
                        isOnline: true, canTrain: true, canProcess: false)
    }

    /// Remote devices (online first, then by name).
    public var remote: [SplatDeviceInfo] {
        browser.workers.filter { !excludedNames.contains($0.name) }.map { w in
            let h = browser.health[w.id]
            let info = h?.info
            var online = false
            var status = ""
            switch h?.status ?? .unknown {
            case .online: online = true
            case .checking, .unknown: status = "checking…"
            case .unauthorized: status = "needs an access token (set it in SplatWork)"
            case .offline(let e): status = "offline: \(e)"
            }
            let kind: SplatDeviceInfo.Kind = {
                switch (info?.kind ?? w.txt["kind"])?.lowercased() {
                case "iphone": return .iphone
                case "mac": return .mac
                default: return .pc
                }
            }()
            let gpu = info?.gpus?.first?.name ?? w.gpuName ?? info?.backend ?? ""
            var detail = online ? [gpu, info?.busy == true ? "busy" : nil].compactMap { $0 }.filter { !$0.isEmpty }.joined(separator: " · ") : status
            if online, let q = info?.queueLength, q > 0 { detail += " · \(q) queued" }
            return SplatDeviceInfo(id: w.id, name: info?.name ?? w.name, kind: kind, detail: detail, isOnline: online,
                                   canTrain: info?.canTrain ?? true, canProcess: info?.canProcess ?? false)
        }.sorted { ($0.isOnline ? 0 : 1, $0.name) < ($1.isOnline ? 0 : 1, $1.name) }
    }

    public var all: [SplatDeviceInfo] { [thisPhone] + remote }
    public var anyProcessor: Bool { remote.contains { $0.isOnline && $0.canProcess } }

    /// Online workers as clients + infos (for the pipeline, off the main actor).
    func snapshot() -> [(device: ComputeDevice, client: WorkerClient, info: WorkerInfo)] {
        browser.workers.filter { !excludedNames.contains($0.name) }.compactMap { w in
            guard let h = browser.health[w.id], case .online = h.status, let info = h.info, let c = browser.client(for: w.id) else { return nil }
            return (ComputeScheduler.remoteDevice(w, info: info), c, info)
        }
    }
}

// MARK: - Progress / result

public struct SplatMakeProgress: Sendable, Equatable {
    public var stage: String
    /// 0…1 over the whole run.
    public var fraction: Double
    public var message: String
    /// Training numbers (when training).
    public var step: Int = 0
    public var totalSteps: Int = 0
    public var psnr: Double? = nil
    public var splats: Int = 0
    public var eta: Double? = nil
    /// One line per device / block while training on several.
    public var lines: [String] = []

    public init(stage: String, fraction: Double, message: String, step: Int = 0, totalSteps: Int = 0, psnr: Double? = nil,
                splats: Int = 0, eta: Double? = nil, lines: [String] = []) {
        self.stage = stage; self.fraction = fraction; self.message = message; self.step = step; self.totalSteps = totalSteps
        self.psnr = psnr; self.splats = splats; self.eta = eta; self.lines = lines
    }
}

public struct SplatMakeResult: Sendable {
    public var plyURL: URL
    public var datasetURL: URL
    public var splatCount: Int
    public var cameras: Int
    public var seconds: Double
    /// What happened along the way (skipped steps, where it trained, memory clamps).
    public var notes: [String]
}

// MARK: - Pipeline

public enum SplatMaker {
    static let log = Logger(subsystem: "SplatCapture", category: "make-splat")

    /// Turns a `.splatcapture` bundle into a trained splat (`workDir/splat.ply`). Cancel the calling task to stop.
    /// - Parameter devices: pass `SplatDevices.shared` (read on the main actor for the online workers).
    public static func make(capture: URL, workDir: URL, options: SplatMakeOptions, devices: SplatDevices,
                            progress: @escaping @Sendable (SplatMakeProgress) -> Void) async throws -> SplatMakeResult {
        let t0 = Date()
        let notes = NoteList()
        let fm = FileManager.default
        try fm.createDirectory(at: workDir, withIntermediateDirectories: true)
        let dsDir = workDir.appendingPathComponent("dataset")
        try? fm.removeItem(at: dsDir)
        func report(_ stage: String, _ f: Double, _ m: String) { progress(SplatMakeProgress(stage: stage, fraction: f, message: m)) }
        func scaled(_ range: ClosedRange<Double>) -> PipelineProgressHandler {
            { p in
                let f = p.fraction < 0 ? -1 : range.lowerBound + (range.upperBound - range.lowerBound) * min(1, max(0, p.fraction))
                progress(SplatMakeProgress(stage: p.stage, fraction: f, message: p.message))
            }
        }

        let remotes = await MainActor.run { devices.snapshot() }
        let processors = remotes.filter { $0.info.canProcess }
        let processor = (options.processorId.flatMap { id in processors.first { $0.device.id == id } }
            ?? (options.processorId == nil ? processors.first { options.deviceIds.contains($0.device.id) } ?? processors.first : nil))?.client

        // 1. Dataset from the capture's frames + ARKit poses (images resized for the quality, LiDAR depth + points).
        var cfg = options.quality.preset.config
        if let n = options.iterationsOverride {
            cfg.iterations = n
            cfg.densifyStop = min(cfg.densifyStop, max(n * 8 / 10, 1))
        }
        var importOpts = CaptureImportOptions()
        importOpts.frameStride = max(1, options.frameStride)
        importOpts.images.maxImageSize = cfg.maxImageSize
        importOpts.images.holdoutInterval = nil
        report("Preparing", 0, "reading the capture")
        var ds = try await CaptureImporter.importBundle(capture, options: importOpts, into: dsDir, progress: scaled(0...0.15))
        try Task.checkCancellation()
        notes.add("\(ds.manifest.cameras.count) photos with ARKit poses\(ds.manifest.points != nil ? " and a LiDAR point cloud" : "")")

        // 2. Masks: Apple Vision on the phone; learned ones on a Linux worker when there is one.
        var masking = MaskingOptions()
        masking.model = options.masks.model
        if options.learnedMovingMasks {
            if let processor {
                var learned = masking
                learned.model = .splatprep(maskModelId: "dynamic")
                let before = ds.manifest.cameras.filter { $0.mask != nil }.count
                ds = try await maskDataset(ds, learned, worker: processor, notes: notes, progress: scaled(0.15...0.25))
                // the worker couldn't (model not installed, …): Apple Vision when that was picked too
                if ds.manifest.cameras.filter({ $0.mask != nil }).count == before, options.masks != .none {
                    notes.add("using Apple Vision masks instead")
                    ds = try await maskDataset(ds, masking, worker: nil, notes: notes, progress: scaled(0.15...0.25))
                }
            } else {
                notes.add("learned moving-object masks skipped (no SplatWork Node PC online)\(options.masks != .none ? " — using Apple Vision instead" : "")")
                if options.masks != .none { ds = try await maskDataset(ds, masking, worker: nil, notes: notes, progress: scaled(0.15...0.25)) }
            }
        } else if options.masks != .none {
            ds = try await maskDataset(ds, masking, worker: nil, notes: notes, progress: scaled(0.15...0.25))
        }
        try Task.checkCancellation()
        if options.sky {
            if let processor {
                var sky = MaskingOptions(); sky.sky = true
                ds = try await maskDataset(ds, sky, worker: processor, notes: notes, progress: scaled(0.25...0.3))
                cfg.skybox = "auto"   // on when any photo got a sky mask
            } else {
                notes.add("sky masks skipped (no SplatWork Node PC online)")
            }
        }
        try Task.checkCancellation()

        // 3. Pose refinement (SfM with the ARKit poses as priors) on a Linux worker.
        if options.refinePoses {
            if let processor {
                ds = await refinePoses(ds, worker: processor, notes: notes, progress: scaled(0.3...0.4))
            } else {
                notes.add("pose refinement skipped (no SplatWork Node PC online) — using the ARKit poses as recorded")
            }
        }
        try Task.checkCancellation()

        // 4. Training.
        let output = workDir.appendingPathComponent("splat.ply")
        try? fm.removeItem(at: output)
        let wanted = options.deviceIds.isEmpty ? [SplatDeviceInfo.localId] : options.deviceIds
        let chosen = remotes.filter { wanted.contains($0.device.id) && $0.device.capabilities.contains("train") }
        let useLocal = wanted.contains(SplatDeviceInfo.localId) || chosen.isEmpty
        if !wanted.contains(SplatDeviceInfo.localId) && chosen.isEmpty {
            notes.add("none of the chosen devices is online — training on this iPhone")
        }
        let trainRange = 0.4...1.0
        let cloud = try await train(dataset: ds, config: cfg, useLocal: useLocal, remotes: chosen, name: capture.deletingPathExtension().lastPathComponent,
                                    notes: notes) { p in
            var p = p
            p.fraction = trainRange.lowerBound + (trainRange.upperBound - trainRange.lowerBound) * p.fraction
            progress(p)
        }
        report("Saving", 1, "writing \(cloud.count) splats")
        try PLYCodec.writeGaussians(cloud, to: output, shDegree: cfg.shDegree, up: ds.upVectorSIMD)
        return SplatMakeResult(plyURL: output, datasetURL: dsDir, splatCount: cloud.count, cameras: ds.manifest.cameras.count,
                               seconds: Date().timeIntervalSince(t0), notes: notes.all)
    }

    // MARK: masks

    /// `DatasetBuilder.maskDataset` without the rest of the Mac's DatasetBuilder: masks the dataset's cameras.
    static func maskDataset(_ ds: PreparedDataset, _ m: MaskingOptions, worker: WorkerClient?, notes: NoteList,
                            progress: PipelineProgressHandler?) async throws -> PreparedDataset {
        var frames = frameSet(from: ds)
        do {
            switch m.model {
            case .none: break
            case .splatprep(let id):
                guard let worker else { return ds }
                try await FrameMasker.applySplatPrep(modelId: id, options: m, to: &frames, runner: SplatPrepRunner(), worker: worker, progress: progress)
                try frames.save()
            default:
                try await FrameMasker.applyVision(m, to: &frames, progress: progress)
            }
            if m.sky, let worker {
                try await FrameMasker.applySky(to: &frames, runner: SplatPrepRunner(), worker: worker, progress: progress)
            }
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            if Task.isCancelled { throw CancellationError() }
            // FrameMasker falls back to a local splatprep run when the worker fails; that can't run on iOS.
            notes.add("masking skipped: \(error.localizedDescription)")
            return ds
        }
        var out = ds
        let masks = Dictionary(frames.frames.map { ($0.image, ($0.mask, $0.sky)) }, uniquingKeysWith: { a, _ in a })
        var n = 0
        for i in out.manifest.cameras.indices {
            if let mm = masks[out.manifest.cameras[i].image] {
                if let mk = mm.0 { out.manifest.cameras[i].mask = mk; n += 1 }
                if let sk = mm.1 { out.manifest.cameras[i].sky = sk }
            }
        }
        try out.save()
        try? FileManager.default.removeItem(at: ds.root.appendingPathComponent(FrameSet.fileName))
        switch m.model {
        case .none: break
        case .splatprep: notes.add("learned masks from the worker on \(n) photos")
        default: notes.add("Apple Vision masks on \(n) photos")
        }
        if m.sky { notes.add("sky found in \(out.manifest.cameras.filter { $0.sky != nil }.count) photos on the worker (trains a skybox)") }
        return out
    }

    static func frameSet(from ds: PreparedDataset) -> FrameSet {
        FrameSet(root: ds.root, frames: ds.manifest.cameras.map { c in
            var f = IngestedFrame(image: c.image, width: c.width, height: c.height)
            f.focalPx = c.fx; f.cx = c.cx; f.cy = c.cy; f.mask = c.mask; f.sky = c.sky
            f.depth = c.depth; f.depthWidth = c.depthWidth; f.depthHeight = c.depthHeight
            return f
        })
    }

    // MARK: pose refinement

    /// Same as the Mac's capture refinement (DatasetBuilder.build, `.captureBundle`): splatprep SfM on the worker
    /// with the ARKit poses as priors; keeps the capture poses when it fails or places too few photos.
    static func refinePoses(_ ds: PreparedDataset, worker: WorkerClient, notes: NoteList, progress: PipelineProgressHandler?) async -> PreparedDataset {
        do {
            try PriorsFile.write(dataset: ds, to: ds.root.appendingPathComponent("priors.json"))
            let frames = frameSet(from: ds)
            var sp = SplatPrepPoseSolver(features: "sift_lightglue", matching: "auto", mapper: "auto", cameraModel: "pinhole")
            sp.priorPoses = "priors.json"
            let sol = try await RemoteSplatPrepPoseSolver(client: worker, local: sp).solve(frames, progress: progress)
            guard Double(sol.poses.count) >= 0.5 * Double(frames.frames.count) else {
                notes.add("pose refinement placed only \(sol.poses.count) of \(frames.frames.count) photos — keeping the ARKit poses")
                return ds
            }
            let stem = { (p: String) in ((p as NSString).lastPathComponent as NSString).deletingPathExtension }
            let captured = Dictionary(ds.manifest.cameras.map { (stem($0.image), $0) }, uniquingKeysWith: { a, _ in a })
            var out = try PoseAssembly.makeDataset(frames: frames, solution: sol, holdoutInterval: nil)
            if out.manifest.gravityUp == nil { out.manifest.gravityUp = ds.manifest.gravityUp }
            for i in out.manifest.cameras.indices {
                guard let c = captured[stem(out.manifest.cameras[i].image)] else { continue }
                if out.manifest.cameras[i].exposure == nil { out.manifest.cameras[i].exposure = c.exposure }
                if out.manifest.cameras[i].time == nil { out.manifest.cameras[i].time = c.time }
                if out.manifest.cameras[i].group == nil { out.manifest.cameras[i].group = c.group }
                if out.manifest.cameras[i].depth == nil, c.width == out.manifest.cameras[i].width {
                    out.manifest.cameras[i].depth = c.depth
                    out.manifest.cameras[i].depthWidth = c.depthWidth; out.manifest.cameras[i].depthHeight = c.depthHeight
                }
            }
            if out.manifest.points == nil { out.manifest.points = ds.manifest.points }
            try out.save()
            notes.add("poses refined on the worker (\(sol.poses.count) of \(frames.frames.count) photos)")
            return out
        } catch {
            notes.add("pose refinement skipped: \(error.localizedDescription)")
            return ds
        }
    }

    // MARK: training

    struct TrainProgress: Sendable { var p: SplatMakeProgress }

    static func train(dataset: PreparedDataset, config: TrainingConfig, useLocal: Bool,
                      remotes: [(device: ComputeDevice, client: WorkerClient, info: WorkerInfo)], name: String, notes: NoteList,
                      progress: @escaping @Sendable (SplatMakeProgress) -> Void) async throws -> GaussianCloud {
        let phone = PhoneTrainingRunner(notes: notes)
        var local = ComputeScheduler.localDevice()
        local.name = "this iPhone"
        local.throughput = ComputeScheduler.estimateThroughput(gpus: [.init(name: TrainKit.gpuName, memoryMB: nil, freeMemoryMB: nil)])
        let start = Date()
        func snap(_ s: TrainingProgressSnapshot, where dev: String) -> SplatMakeProgress {
            SplatMakeProgress(stage: "Training on \(dev)", fraction: s.fraction, message: "step \(s.step) / \(s.totalSteps)",
                              step: s.step, totalSteps: s.totalSteps, psnr: s.psnr, splats: s.numGaussians, eta: s.eta)
        }
        let devices = (useLocal ? [local] : []) + remotes.map(\.device)
        // One device, or too few photos for blocks to pay off: train the whole scene on the fastest device.
        if devices.count == 1 || dataset.manifest.cameras.count < 400 {
            let best = devices.max { $0.throughput < $1.throughput }!
            if devices.count > 1 { notes.add("\(dataset.manifest.cameras.count) photos is too few to split — the whole scene trains on \(best.name), the fastest chosen device") }
            if best.isLocal {
                progress(SplatMakeProgress(stage: "Training on this iPhone", fraction: 0, message: "loading photos"))
                return try await phone.train(dataset: dataset, config: config) { progress(snap($0, where: "this iPhone")) }
            }
            let r = remotes.first { $0.device.id == best.id }!
            if let problem = MemoryForecast.workerProblem(info: r.info, dataset: dataset, config: config) { notes.add("\(best.name): \(problem)") }
            progress(SplatMakeProgress(stage: "Sending to \(best.name)", fraction: 0, message: "uploading photos"))
            notes.add("trained on \(best.name)")
            return try await r.client.train(dataset: dataset, config: config, name: name, progress: { s in
                progress(snap(TrainingProgressSnapshot(s), where: best.name))
            }, uploadProgress: { f in
                progress(SplatMakeProgress(stage: "Sending to \(best.name)", fraction: 0, message: "uploading photos \(Int(f * 100)) %"))
            })
        }
        // Blocks: two per device, more until each fits somewhere.
        let infos = Dictionary(uniqueKeysWithValues: remotes.map { ($0.device.id, $0.info) })
        let fits: (String, SceneBlock, [SceneBlock]) -> Bool = { dev, b, all in
            let sub = dataset.restricted(to: Set(b.cameraIds))
            let c = ComputeScheduler.blockConfig(config, block: b, blocks: all)
            if dev == "local" { return PhoneTrainingRunner.fits(dataset: sub, config: c) }
            guard let info = infos[dev] else { return true }
            return MemoryForecast.workerProblem(info: info, dataset: sub, config: c) == nil
        }
        progress(SplatMakeProgress(stage: "Splitting the scene", fraction: 0, message: "partitioning into blocks"))
        let pc = dataset.pointCloudURL.flatMap { try? PLYCodec.readPointCloud(url: $0) }
        var k = max(2, devices.count * 2)
        var blocks = BlockPartitioner(blockCount: k).partition(dataset: dataset, points: pc)
        while k < 32, !blocks.allSatisfy({ b in devices.contains { fits($0.id, b, blocks) } }) {
            k += max(1, k / 2)
            blocks = BlockPartitioner(blockCount: k).partition(dataset: dataset, points: pc)
        }
        let names = Dictionary(uniqueKeysWithValues: devices.map { ($0.id, $0.name) })
        notes.add("trained as \(blocks.count) blocks on \(devices.map(\.name).joined(separator: ", "))")
        let final = blocks
        let latest = BlockLatest()
        let total = config.iterations * blocks.count
        let scheduler = ComputeScheduler(devices: devices, placement: .both)
        return try await scheduler.trainBlockParallel(dataset: dataset, config: config, blocks: blocks, name: name,
                                                      local: useLocal ? phone : nil, clients: Dictionary(uniqueKeysWithValues: remotes.map { ($0.device.id, $0.client) }),
                                                      fits: { fits($0, $1, final) }) { bp in
            let all = latest.set(bp)
            let steps = all.values.reduce(0) { $0 + $1.snapshot.step }
            let psnrs = all.values.compactMap(\.snapshot.psnr)
            let elapsed = Date().timeIntervalSince(start)
            var p = SplatMakeProgress(stage: "Training \(final.count) blocks", fraction: bp.overall, message: "\(Int(bp.overall * 100)) %",
                                      step: min(total, steps), totalSteps: total,
                                      psnr: psnrs.isEmpty ? nil : psnrs.reduce(0, +) / Double(psnrs.count),
                                      splats: all.values.reduce(0) { $0 + $1.snapshot.numGaussians },
                                      eta: bp.overall > 0.01 ? elapsed * (1 - bp.overall) / bp.overall : nil)
            p.lines = all.keys.sorted().map { i in
                let b = all[i]!
                return "Block \(i + 1) on \(names[b.deviceId] ?? b.deviceId): \(Int(b.snapshot.fraction * 100)) %"
            }
            progress(p)
        }
    }
}

/// Notes collected across the pipeline (thread-safe).
final class NoteList: @unchecked Sendable {
    private let lock = NSLock()
    private var items: [String] = []
    func add(_ s: String) { lock.lock(); items.append(s); lock.unlock(); SplatMaker.log.info("\(s, privacy: .public)") }
    var all: [String] { lock.lock(); defer { lock.unlock() }; return items }
}

final class BlockLatest: @unchecked Sendable {
    private let lock = NSLock()
    private var d: [Int: ComputeScheduler.BlockProgress] = [:]
    func set(_ b: ComputeScheduler.BlockProgress) -> [Int: ComputeScheduler.BlockProgress] {
        lock.lock(); defer { lock.unlock() }
        d[b.blockIndex] = b
        return d
    }
}

/// The Metal trainer on this iPhone, inside the app's memory budget (`TrainKit.budget`). Cancels with the task.
struct PhoneTrainingRunner: LocalTrainingRunner {
    var notes: NoteList? = nil

    /// Whether a (block) job fits the phone without cutting its splat budget below 80 %.
    static func fits(dataset: PreparedDataset, config: TrainingConfig) -> Bool {
        var c = config
        let b = TrainKit.budget(dataset: dataset, config: &c, availableBytes: TrainKit.availableMemory)
        return Double(b.maxGaussians) >= 0.8 * Double(b.requestedGaussians) && b.maxImageSize == b.requestedImageSize
    }

    func train(dataset: PreparedDataset, config: TrainingConfig,
               progress: @escaping @Sendable (TrainingProgressSnapshot) -> Void) async throws -> GaussianCloud {
        var cfg = config
        let budget = TrainKit.budget(dataset: dataset, config: &cfg, availableBytes: TrainKit.availableMemory)
        SplatMaker.log.notice("\(budget.summary, privacy: .public)")
        if budget.maxGaussians != budget.requestedGaussians || budget.maxImageSize != budget.requestedImageSize {
            notes?.add("iPhone memory: " + budget.summary)
        }
        let trainer = MetalTrainer(dataset: dataset, config: cfg)
        return try await withTaskCancellationHandler {
            try await Task.detached(priority: .userInitiated) {
                try trainer.prepare { _ in }
                try Task.checkCancellation()
                do {
                    return try trainer.run(previewEvery: 0, onStatus: { s in
                        progress(TrainingProgressSnapshot(step: s.step, totalSteps: s.totalSteps, loss: Double(s.loss), psnr: Double(s.evalPSNR ?? s.psnr),
                                                          numGaussians: s.numGaussians, elapsed: s.elapsed, eta: s.eta))
                    })
                } catch TrainerError.cancelled {
                    throw CancellationError()
                }
            }.value
        } onCancel: {
            trainer.cancel()
        }
    }
}
