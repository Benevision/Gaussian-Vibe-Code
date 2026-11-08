import Foundation
import Metal

/// Where a project's work runs (§7).
enum ComputePlacement: String, Codable, CaseIterable, Sendable {
    /// Everything on this Mac.
    case thisMac
    /// Everything on the selected remote worker (dataset prep stays local where it must, e.g. Object Capture).
    case remote
    /// Each stage on the best device; large scenes are trained block-parallel on all devices.
    case both
}

/// Pipeline stages that can be placed on a device.
enum ComputeStage: String, Codable, CaseIterable, Sendable { case frameExtraction, masking, sfm, training }

/// A compute device known to the scheduler.
struct ComputeDevice: Identifiable, Hashable, Sendable {
    enum Kind: Hashable, Sendable { case local, remote(WorkerEndpoint) }
    var id: String
    var name: String
    var kind: Kind
    /// Capabilities: "train", "process", "objectCapture", "frames", "appleVision", plus splatprep model ids.
    var capabilities: Set<String>
    /// Relative training throughput (1.0 ≈ M4 Pro Metal trainer). Refined from measured runs.
    var throughput: Double
    /// Optional per-model processing scores (higher = better), e.g. from a splatprep benchmark (1 / wall time).
    var modelScores: [String: Double] = [:]
    /// Current queue length (remote).
    var queueLength: Int = 0

    var isLocal: Bool { if case .local = kind { return true }; return false }
    var endpoint: WorkerEndpoint? { if case .remote(let e) = kind { return e }; return nil }
}

/// One placement decision.
struct StagePlacement: Hashable, Sendable {
    var stage: ComputeStage
    var deviceId: String
    var reason: String
}

/// Places stages on devices and runs block-parallel training (PROTOCOL.md §7).
struct ComputeScheduler: Sendable {
    var devices: [ComputeDevice]
    var placement: ComputePlacement = .both

    /// The local Mac device (capabilities from what is available on this machine).
    static func localDevice(splatprepModels: [String] = []) -> ComputeDevice {
        var caps: Set<String> = ["train", "frames", "appleVision", "objectCapture"]
        if !splatprepModels.isEmpty { caps.insert("process"); caps.formUnion(splatprepModels) }
        // Throughput: scale by GPU working-set size relative to an M4 Pro (24–48 GB unified memory).
        var t = 1.0
        if let dev = MTLCreateSystemDefaultDevice() {
            let gb = Double(dev.recommendedMaxWorkingSetSize) / 1e9
            t = min(3, max(0.3, gb / 20))
        }
        if let measured = BenchmarkStore.shared.throughput(for: "local") { t = measured }   // Compute → Benchmark
        return ComputeDevice(id: "local", name: LocalDeviceName.current, kind: .local, capabilities: caps, throughput: t)
    }

    /// A remote device from a health-checked worker.
    static func remoteDevice(_ ep: WorkerEndpoint, info: WorkerInfo) -> ComputeDevice {
        var caps: Set<String> = []
        if info.canTrain { caps.insert("train") }
        if info.canProcess { caps.insert("process"); caps.insert("frames") }
        caps.formUnion(info.models ?? [])
        // Measured (Compute → Benchmark) beats the GPU-name table. Both use the same scale (1.0 ≈ 20 benchmark steps/s
        // ≈ an M4 Pro), so a measured device and a table guess can still be compared.
        let t = BenchmarkStore.shared.throughput(for: ep.id) ?? estimateThroughput(gpus: info.gpus ?? [])
        return ComputeDevice(id: ep.id, name: info.name, kind: .remote(ep), capabilities: caps,
                             throughput: t, queueLength: info.queueLength ?? 0)
    }

    /// Rough CUDA throughput score relative to an M4 Pro (from GPU name / memory), summed over GPUs.
    static func estimateThroughput(gpus: [WorkerInfo.GPU]) -> Double {
        let table: [(String, Double)] = [("H200", 9), ("H100", 8), ("B200", 10), ("5090", 6), ("4090", 5), ("A100", 5), ("L40", 4.5),
                                         ("6000 Ada", 5), ("4080", 3.8), ("5080", 4.2), ("3090", 3), ("A6000", 3), ("4070", 2.5),
                                         ("3080", 2.5), ("A10", 2), ("3070", 1.8), ("T4", 0.8), ("2080", 1.5),
                                         // Apple GPUs (iPhone / other Macs running the Metal trainer); before "A10"-style
                                         // CUDA names can't collide: checked in order, longest names first
                                         ("M4 Max", 2), ("M4 Pro", 1), ("M3 Max", 1.7), ("M3 Pro", 0.8), ("M2 Max", 1.4),
                                         ("A18 Pro", 0.22), ("A18", 0.18), ("A17 Pro", 0.18)]
        // Apple chip names first (an "Apple A18 Pro" must not match the CUDA "A10"/"A100" entries)
        let apple = table.filter { ["M4", "M3", "M2", "A18", "A17"].contains(where: $0.0.hasPrefix) }
        let s = gpus.map { g -> Double in
            if g.name.hasPrefix("Apple"), let hit = apple.first(where: { g.name.localizedCaseInsensitiveContains($0.0) }) { return hit.1 }
            if let hit = table.first(where: { g.name.localizedCaseInsensitiveContains($0.0) }) { return hit.1 }
            return max(0.5, Double(g.memoryMB ?? 8000) / 8000)
        }.reduce(0, +)
        return s > 0 ? s : 1
    }

    /// Records a measured training rate (steps/s) so later scheduling uses real numbers (EMA).
    mutating func record(deviceId: String, stepsPerSecond: Double, referenceStepsPerSecond: Double = 40) {
        guard let i = devices.firstIndex(where: { $0.id == deviceId }), stepsPerSecond > 0 else { return }
        devices[i].throughput = 0.5 * devices[i].throughput + 0.5 * (stepsPerSecond / referenceStepsPerSecond)
    }

    /// Picks a device for a stage. `model` is the splatprep model id (sfm/masking) or "objectCapture"/"appleVision".
    func place(_ stage: ComputeStage, model: String? = nil) -> StagePlacement? {
        let local = devices.first(where: \.isLocal)
        let remotes = devices.filter { !$0.isLocal }
        func capable(_ d: ComputeDevice) -> Bool {
            switch stage {
            case .frameExtraction: return d.capabilities.contains("frames")
            case .masking, .sfm:
                guard let m = model else { return d.capabilities.contains("process") }
                if m == "objectCapture" || m == "appleVision" { return d.capabilities.contains(m) }
                return d.capabilities.contains("process") && d.capabilities.contains(m)
            case .training: return d.capabilities.contains("train")
            }
        }
        let pool: [ComputeDevice]
        switch placement {
        case .thisMac: pool = local.map { [$0] } ?? []
        case .remote: pool = remotes.filter(capable).isEmpty ? (local.map { [$0] } ?? []) : remotes
        case .both: pool = devices
        }
        let candidates = pool.filter(capable)
        // Frame extraction reads local files: keep it local unless forced remote. (Videos can still be split on the
        // worker: DatasetBuilder.framesOnWorker uploads the file itself when PrepOptions.videoOnWorker is on.)
        if stage == .frameExtraction, let local, capable(local) { return StagePlacement(stage: stage, deviceId: local.id, reason: "source files are local") }
        guard !candidates.isEmpty else { return nil }
        let best = candidates.max { score($0, stage, model) < score($1, stage, model) }!
        let why = best.isLocal ? "best available (\(String(format: "%.1f", score(best, stage, model))))" : "remote \(best.name), score \(String(format: "%.1f", score(best, stage, model)))"
        return StagePlacement(stage: stage, deviceId: best.id, reason: why)
    }

    private func score(_ d: ComputeDevice, _ stage: ComputeStage, _ model: String?) -> Double {
        let queuePenalty = 1.0 / (1.0 + Double(d.queueLength))
        if let m = model, let s = d.modelScores[m] { return s * queuePenalty }
        // Upload cost: prefer local for small work when scores tie.
        let localBonus = d.isLocal ? 1.15 : 1.0
        return d.throughput * queuePenalty * localBonus
    }

    /// Placement for a whole prep + training run.
    func plan(sfmModel: String?, maskModel: String?) -> [StagePlacement] {
        [place(.frameExtraction), maskModel.flatMap { place(.masking, model: $0) }, sfmModel.flatMap { place(.sfm, model: $0) }, place(.training)]
            .compactMap { $0 }
    }

    /// Assigns blocks to devices proportional to throughput (greedy longest-processing-time first).
    /// `fits(deviceId, block)`: whether that block fits the device's memory (blocks go only to devices they fit;
    /// if none fits, to the device with the most throughput).
    func distribute(_ blocks: [SceneBlock], fits: ((String, SceneBlock) -> Bool)? = nil) -> [String: [SceneBlock]] {
        let trainers = devices.filter { $0.capabilities.contains("train") && (placement != .thisMac || $0.isLocal) }
        guard !trainers.isEmpty else { return [:] }
        var load: [String: Double] = [:]
        var out: [String: [SceneBlock]] = [:]
        for b in blocks.sorted(by: { cost($0) > cost($1) }) {
            let able = trainers.filter { fits?($0.id, b) ?? true }
            let pool = able.isEmpty ? [trainers.max { $0.throughput < $1.throughput }!] : able
            let d = pool.min { a, c in
                (load[a.id, default: 0] + cost(b)) / max(a.throughput, 0.01) < (load[c.id, default: 0] + cost(b)) / max(c.throughput, 0.01)
            }!
            load[d.id, default: 0] += cost(b)
            out[d.id, default: []].append(b)
        }
        return out
    }

    private func cost(_ b: SceneBlock) -> Double { Double(max(1, b.cameraIds.count)) + b.weight / 10_000 }

    // MARK: Block-parallel training

    /// Progress of a block-parallel run.
    struct BlockProgress: Sendable {
        var blockIndex: Int
        var deviceId: String
        var snapshot: TrainingProgressSnapshot
        /// Mean fraction over all blocks.
        var overall: Double
    }

    /// Trains a scene as K blocks distributed over the devices and merges the results.
    /// - Parameters:
    ///   - clients: `WorkerClient` per remote device id (uploads the dataset once per worker).
    ///   - local: the Metal trainer (used for blocks placed on "local").
    /// Per-block training settings: the block's region and cameras, and its share of the splat budget (by block
    /// weight, +25 % for the overlap margins, at least 1/(2K) of the total) so the merged scene stays near the cap.
    static func blockConfig(_ config: TrainingConfig, block b: SceneBlock, blocks: [SceneBlock]) -> TrainingConfig {
        var cfg = config
        cfg.region = b.region
        cfg.cameraIds = b.cameraIds
        let total = blocks.reduce(0.0) { $0 + max($1.weight, 1) }
        let share = max(max(b.weight, 1) / max(total, 1), 0.5 / Double(blocks.count))
        cfg.maxGaussians = max(100_000, Int(Double(config.maxGaussians) * share * 1.25))
        return cfg
    }

    func trainBlockParallel(dataset: PreparedDataset, config: TrainingConfig, blocks: [SceneBlock], name: String,
                            local: LocalTrainingRunner?, clients: [String: WorkerClient],
                            fits: ((String, SceneBlock) -> Bool)? = nil,
                            progress: (@Sendable (BlockProgress) -> Void)? = nil) async throws -> GaussianCloud {
        guard !blocks.isEmpty else { throw PipelineError.noInput("no blocks") }
        var assignment = distribute(blocks, fits: fits)
        // Devices we can't actually drive fall back to the local trainer (or the first client).
        for (dev, bs) in assignment where (dev == "local" && local == nil) || (dev != "local" && clients[dev] == nil) {
            assignment.removeValue(forKey: dev)
            let fallback = local != nil ? "local" : clients.keys.sorted().first
            guard let f = fallback else { throw PipelineError.unsupported("no device can train") }
            assignment[f, default: []] += bs
        }
        let tracker = BlockProgressTracker(count: blocks.count)
        return try await withThrowingTaskGroup(of: [(SceneBlock, GaussianCloud)].self) { group in
            for (dev, bs) in assignment {
                group.addTask {
                    var results: [(SceneBlock, GaussianCloud)] = []
                    var datasetId: String? = nil
                    for b in bs {
                        try Task.checkCancellation()
                        let cfg = Self.blockConfig(config, block: b, blocks: blocks)
                        let report: @Sendable (TrainingProgressSnapshot) -> Void = { s in
                            let overall = tracker.update(b.index, s.fraction)
                            progress?(BlockProgress(blockIndex: b.index, deviceId: dev, snapshot: s, overall: overall))
                        }
                        let cloud: GaussianCloud
                        if dev == "local", let local {
                            cloud = try await local.train(dataset: dataset.restricted(to: Set(b.cameraIds)), config: cfg, progress: report)
                        } else if let client = clients[dev] {
                            // the worker may already hold the photos (split / masked / solved there): send only the rest
                            if datasetId == nil {
                                datasetId = try await client.uploadReusing(RemoteStore.usable(at: dataset.root, by: client), directory: dataset.root)
                            }
                            cloud = try await client.train(dataset: dataset, config: cfg, name: "\(name) block \(b.index + 1)/\(blocks.count)",
                                                           datasetId: datasetId, progress: { report(TrainingProgressSnapshot($0)) })
                        } else { throw PipelineError.unsupported("device \(dev) unavailable") }
                        _ = tracker.update(b.index, 1)
                        results.append((b, cloud))
                    }
                    return results
                }
            }
            var all: [(block: SceneBlock, cloud: GaussianCloud)] = []
            for try await r in group { all += r.map { (block: $0.0, cloud: $0.1) } }
            all.sort { $0.block.index < $1.block.index }
            return BlockMerger.merge(all)
        }
    }
}

final class BlockProgressTracker: @unchecked Sendable {
    private let lock = NSLock()
    private var fractions: [Double]
    init(count: Int) { fractions = Array(repeating: 0, count: count) }
    func update(_ i: Int, _ f: Double) -> Double {
        lock.lock(); defer { lock.unlock() }
        if i < fractions.count { fractions[i] = max(fractions[i], f) }
        return fractions.reduce(0, +) / Double(max(1, fractions.count))
    }
}
