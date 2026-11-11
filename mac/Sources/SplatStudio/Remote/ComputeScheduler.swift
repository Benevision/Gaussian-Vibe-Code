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
        var stage: BlockStage = .training
    }

    /// Where one block is in its life: waiting for its device, uploading, queued / loading on the worker,
    /// training, downloading the result, done.
    enum BlockStage: Sendable, Equatable {
        case waiting, uploading(Double), queued, loading, training, downloading, done
        case offline(TimeInterval)
        case failed(String)
        /// Failed, and the user chose to finish without it.
        case leftOut

        /// Word for the combined status line ("2 training, 1 uploading").
        var summaryKey: String {
            switch self {
            case .waiting: return "waiting"
            case .uploading: return "uploading"
            case .queued: return "queued"
            case .loading: return "loading"
            case .training: return "training"
            case .downloading: return "downloading"
            case .done: return "done"
            case .offline: return "offline"
            case .failed: return "failed"
            case .leftOut: return "left out"
            }
        }

        var label: String {
            switch self {
            case .waiting: return "Waiting"
            case .uploading(let f): return String(format: "Uploading %.0f %%", f * 100)
            case .queued: return "Queued"
            case .loading: return "Loading photos"
            case .training: return "Training"
            case .downloading: return "Downloading result"
            case .done: return "Done"
            case .offline(let t): return String(format: "Offline %.0f s", t)
            case .failed(let m): return "Failed: \(m)"
            case .leftOut: return "Left out"
            }
        }
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

    /// - Parameter retryGate: when given, a failed block waits for the user to retry it (on any device) or leave it
    ///   out, instead of failing the whole run. Without it the first failure fails the run.
    func trainBlockParallel(dataset: PreparedDataset, config: TrainingConfig, blocks: [SceneBlock], name: String,
                            local: LocalTrainingRunner?, clients: [String: WorkerClient],
                            fits: ((String, SceneBlock) -> Bool)? = nil,
                            retryGate: BlockRetryGate? = nil,
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
        // A retried block can move to another device: one block at a time per device, and one upload per worker.
        let deviceLocks = DeviceLocks()
        let datasetIds = DatasetIdCache()
        // Every block shows up straight away, on its device, before any work starts.
        for (dev, bs) in assignment {
            for b in bs {
                progress?(BlockProgress(blockIndex: b.index, deviceId: dev, snapshot: TrainingProgressSnapshot(step: 0, totalSteps: config.iterations),
                                        overall: 0, stage: .waiting))
            }
        }

        /// Trains one block on one device (waiting for that device to be free first).
        @Sendable func run(_ b: SceneBlock, on dev: String, last: LastSnapshot) async throws -> GaussianCloud {
            let cfg = Self.blockConfig(config, block: b, blocks: blocks)
            let report: @Sendable (TrainingProgressSnapshot, BlockStage) -> Void = { s, stage in
                last.set(s)
                let overall = tracker.update(b.index, s.fraction)
                progress?(BlockProgress(blockIndex: b.index, deviceId: dev, snapshot: s, overall: overall, stage: stage))
            }
            let stageOnly: @Sendable (BlockStage) -> Void = { stage in report(last.get(), stage) }
            await deviceLocks.acquire(dev)
            defer { deviceLocks.release(dev) }
            try Task.checkCancellation()
            if dev == "local", let local {
                stageOnly(.loading)
                return try await local.train(dataset: dataset.restricted(to: Set(b.cameraIds)), config: cfg,
                                             progress: { report($0, .training) })
            } else if let client = clients[dev] {
                // the worker may already hold the photos (split / masked / solved there): send only the rest
                var datasetId = datasetIds.get(dev)
                if datasetId == nil {
                    stageOnly(.uploading(0))
                    datasetId = try await client.uploadReusing(RemoteStore.usable(at: dataset.root, by: client), directory: dataset.root,
                                                               progress: { stageOnly(.uploading($0)) })
                    datasetIds.set(dev, datasetId)
                }
                stageOnly(.queued)
                return try await client.train(dataset: dataset, config: cfg, name: "\(name) block \(b.index + 1)/\(blocks.count)",
                                              datasetId: datasetId, progress: { js in
                    let stage: BlockStage = js.state == .queued ? .queued : js.state == .loading ? .loading
                        : js.state == .completed ? .downloading : .training
                    report(TrainingProgressSnapshot(js), stage)
                }, onOffline: { stageOnly(.offline($0)) })
            }
            throw PipelineError.unsupported("device \(dev) unavailable")
        }

        return try await withThrowingTaskGroup(of: [(SceneBlock, GaussianCloud)].self) { group in
            for (dev, bs) in assignment {
                group.addTask {
                    var results: [(SceneBlock, GaussianCloud)] = []
                    for b in bs {
                        try Task.checkCancellation()
                        var device = dev
                        attempts: while true {
                            let last = LastSnapshot(TrainingProgressSnapshot(step: 0, totalSteps: config.iterations))
                            do {
                                let cloud = try await run(b, on: device, last: last)
                                _ = tracker.update(b.index, 1)
                                var done = last.get(); done.step = max(done.step, done.totalSteps)
                                progress?(BlockProgress(blockIndex: b.index, deviceId: device, snapshot: done, overall: tracker.update(b.index, 1), stage: .done))
                                results.append((b, cloud))
                                break attempts
                            } catch {
                                if error is CancellationError || Task.isCancelled { throw error }
                                progress?(BlockProgress(blockIndex: b.index, deviceId: device, snapshot: last.get(),
                                                        overall: tracker.update(b.index, 0), stage: .failed(error.localizedDescription)))
                                guard let retryGate else { throw error }
                                switch await retryGate.wait(block: b.index) {
                                case .retry(let d)?:
                                    device = d
                                    tracker.reset(b.index)
                                    progress?(BlockProgress(blockIndex: b.index, deviceId: d, snapshot: TrainingProgressSnapshot(step: 0, totalSteps: config.iterations),
                                                            overall: tracker.update(b.index, 0), stage: .waiting))
                                case .leaveOut?:
                                    progress?(BlockProgress(blockIndex: b.index, deviceId: device, snapshot: last.get(),
                                                            overall: tracker.update(b.index, 1), stage: .leftOut))
                                    break attempts
                                case nil:
                                    throw CancellationError()
                                }
                            }
                        }
                    }
                    return results
                }
            }
            var all: [(block: SceneBlock, cloud: GaussianCloud)] = []
            for try await r in group { all += r.map { (block: $0.0, cloud: $0.1) } }
            guard !all.isEmpty else { throw PipelineError.unsupported("every block was left out") }
            all.sort { $0.block.index < $1.block.index }
            return BlockMerger.merge(all)
        }
    }
}

/// What the user chose for a failed block.
enum BlockRetryDecision: Sendable, Equatable {
    /// Train it again on this device id ("local" = this Mac).
    case retry(String)
    /// Finish without it (the merged scene has a gap there).
    case leaveOut
}

/// Lets a failed block wait for the user's decision. The UI calls `decide`; Stop (task cancellation) ends the wait.
final class BlockRetryGate: @unchecked Sendable {
    private let lock = NSLock()
    private var pending: [Int: CheckedContinuation<BlockRetryDecision?, Never>] = [:]

    /// Waits for `decide(block:)`; nil when the run is cancelled.
    func wait(block: Int) async -> BlockRetryDecision? {
        await withTaskCancellationHandler {
            await withCheckedContinuation { (c: CheckedContinuation<BlockRetryDecision?, Never>) in
                lock.lock()
                if Task.isCancelled { lock.unlock(); c.resume(returning: nil); return }
                pending[block] = c
                lock.unlock()
            }
        } onCancel: { self.resolve(block, nil) }
    }

    /// Returns false when that block isn't waiting (already decided, or not failed).
    @discardableResult
    func decide(block: Int, _ decision: BlockRetryDecision) -> Bool { resolve(block, decision) }

    @discardableResult
    private func resolve(_ block: Int, _ value: BlockRetryDecision?) -> Bool {
        lock.lock()
        let c = pending.removeValue(forKey: block)
        lock.unlock()
        c?.resume(returning: value)
        return c != nil
    }
}

/// One block at a time per device.
final class DeviceLocks: @unchecked Sendable {
    private let lock = NSLock()
    private var busy: Set<String> = []
    private var waiters: [String: [CheckedContinuation<Void, Never>]] = [:]

    func acquire(_ d: String) async {
        await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
            lock.lock()
            if busy.contains(d) { waiters[d, default: []].append(c); lock.unlock() }
            else { busy.insert(d); lock.unlock(); c.resume() }
        }
    }

    func release(_ d: String) {
        lock.lock()
        if var w = waiters[d], !w.isEmpty {
            let c = w.removeFirst(); waiters[d] = w
            lock.unlock(); c.resume()       // hand the device straight to the next waiter
        } else {
            busy.remove(d); lock.unlock()
        }
    }
}

/// The dataset id each worker got for this run's upload.
final class DatasetIdCache: @unchecked Sendable {
    private let lock = NSLock()
    private var ids: [String: String] = [:]
    func get(_ d: String) -> String? { lock.lock(); defer { lock.unlock() }; return ids[d] }
    func set(_ d: String, _ id: String?) { lock.lock(); ids[d] = id; lock.unlock() }
}

/// Latest snapshot of one block, so stage-only updates (upload, queued, done) keep its numbers.
final class LastSnapshot: @unchecked Sendable {
    private let lock = NSLock()
    private var value: TrainingProgressSnapshot
    init(_ v: TrainingProgressSnapshot) { value = v }
    func set(_ v: TrainingProgressSnapshot) { lock.lock(); value = v; lock.unlock() }
    func get() -> TrainingProgressSnapshot { lock.lock(); defer { lock.unlock() }; return value }
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
    func reset(_ i: Int) { lock.lock(); if i < fractions.count { fractions[i] = 0 }; lock.unlock() }
}
