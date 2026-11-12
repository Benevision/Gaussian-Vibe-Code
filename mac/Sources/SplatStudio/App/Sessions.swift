import Foundation
import SwiftUI
import Observation
import simd

/// Adapts the Metal trainer to the scheduler's runner protocol (used for local blocks).
struct MetalTrainingRunner: LocalTrainingRunner {
    /// The run's Pause / Resume (blocks started while paused start held).
    var pause: PauseController? = nil
    func train(dataset: PreparedDataset, config: TrainingConfig,
               progress: @escaping @Sendable (TrainingProgressSnapshot) -> Void) async throws -> GaussianCloud {
        try await RemoteImageFetch.shared.ensure(dataset.root)
        // Stop (task cancellation) must reach the trainer, which runs on its own thread.
        let t = MetalTrainer(dataset: dataset, config: config)
        pause?.add(t)
        defer { pause?.remove(t) }
        do {
            return try await withTaskCancellationHandler {
                try await Task.detached(priority: .userInitiated) {
                    try t.prepare { _ in }
                    return try t.run(previewEvery: 0, onStatus: { s in
                        progress(TrainingProgressSnapshot(step: s.step, totalSteps: s.totalSteps, loss: Double(s.loss), psnr: Double(s.psnr),
                                                          numGaussians: s.numGaussians, elapsed: s.elapsed, eta: s.eta))
                    })
                }.value
            } onCancel: { t.cancel() }
        } catch {
            if Task.isCancelled { throw CancellationError() }
            throw error
        }
    }
}

/// Live training state for the dashboard.
@MainActor
@Observable
final class TrainingSession {
    weak var app: AppModel?

    enum State: Equatable { case idle, preparing(String), training, paused, finishing, finished, failed(String), cancelled }
    var state: State = .idle {
        didSet {
            guard state != oldValue else { return }
            switch state {
            case .cancelled: recordTraining { $0.state = .stopped; $0.message = "Stopped" }
            case .failed(let m): recordTraining { $0.state = .failed; $0.message = m }
            default: break
            }
        }
    }
    /// The project being trained (`project.json` keeps this run's settings and progress for Resume).
    private(set) var projectDir: URL? = nil
    private var lastRecorded = Date.distantPast
    /// Block count of the run being resumed (a resumed block run splits the scene the same way).
    private var resumeBlockCount: Int? = nil

    private func recordTraining(_ change: @escaping (inout ProjectRecord.Training) -> Void) {
        guard let p = projectDir else { return }
        ProjectLibrary.shared.update(p) { r in
            guard var t = r.training else { return }
            change(&t); t.updated = Date(); r.training = t
        }
    }

    /// `outputs/.resume/<result name>.ply`: this Mac's periodic checkpoint of a single-piece run.
    nonisolated static func checkpointURL(for output: URL) -> URL {
        output.deletingLastPathComponent().appendingPathComponent(".resume", isDirectory: true).appendingPathComponent(output.lastPathComponent)
    }
    /// `outputs/.blocks/<settings>/`: finished blocks of a block-parallel run, until it is merged.
    nonisolated static func blockStore(project: URL, dataset: PreparedDataset, config: TrainingConfig) -> URL {
        let manifest = (try? Data(contentsOf: dataset.root.appendingPathComponent(PreparedDataset.fileName))) ?? Data()
        let key = sha256Hex(manifest + Data(canonicalJSON(config).utf8)).prefix(16)
        return project.appendingPathComponent("outputs/.blocks/\(key)", isDirectory: true)
    }
    var preset: TrainingConfig.Preset = .balanced { didSet { if !customised { config = preset.config.withDepth(config.depthWeight) } } }
    var config = TrainingConfig.Preset.balanced.config
    var customised = false
    var placement: ComputePlacement = .thisMac
    var workerId: String? = nil
    /// Devices ticked for block-parallel training ("local" = this Mac, else worker ids); nil = every online device.
    var deviceIds: Set<String>? = nil
    var status = TrainerStatus()
    var remoteMessage: String? = nil
    var livePreview = true
    var previewInterval = 250
    var outputURL: URL? = nil

    struct Sample: Identifiable { let id: Int; let step: Int; let loss: Double; let psnr: Double; let gaussians: Int }
    var history: [Sample] = []
    var evalHistory: [Sample] = []

    /// One row per block of a block-parallel run (empty otherwise): which device has it, what it's doing, its numbers.
    struct BlockRow: Identifiable, Equatable {
        let id: Int                 // block index
        var deviceId: String
        var device: String
        var stage: ComputeScheduler.BlockStage
        var snapshot: TrainingProgressSnapshot
        var stepsPerSecond: Double = 0
        /// Seconds left for this block (worker's estimate, else from its rate).
        var eta: TimeInterval? {
            if stage == .done { return 0 }
            if let e = snapshot.eta, e > 0 { return e }
            guard stepsPerSecond > 0.1 else { return nil }
            return Double(max(0, snapshot.totalSteps - snapshot.step)) / stepsPerSecond
        }
    }
    var blockRows: [BlockRow] = []
    /// Devices a failed block can be retried on (this run's trainers), with display names.
    struct RetryDevice: Identifiable, Hashable { let id: String; let name: String }
    var retryDevices: [RetryDevice] = []
    private var retryGate: BlockRetryGate?
    /// Pause / Resume for runs on workers or in blocks (a single run on this Mac uses `trainer` directly).
    private var pauseControl: PauseController?

    /// Trains a failed block again on `deviceId` (it waits there if that device is busy with another block).
    func retryBlock(_ block: Int, on deviceId: String) { retryGate?.decide(block: block, .retry(deviceId)) }
    /// Finishes the run without a failed block; the merged scene has a gap where it was.
    func leaveOutBlock(_ block: Int) { retryGate?.decide(block: block, .leaveOut) }

    private var trainer: MetalTrainer?
    private var task: Task<Void, Never>?

    var isActive: Bool {
        switch state { case .preparing, .training, .paused, .finishing: return true; default: return false }
    }
    var fraction: Double { status.totalSteps > 0 ? Double(status.step) / Double(status.totalSteps) : 0 }

    func trainerSnapshot() -> GaussianCloud? { trainer?.snapshot() }

    /// - Parameter resume: an interrupted run of this project: same settings, devices and result file; continues
    ///   from this Mac's checkpoint, reuses finished blocks, or (a single worker job) starts that job again.
    func start(dataset: PreparedDataset, resume: ProjectRecord.Training? = nil) {
        guard !isActive else { return }
        if let r = resume {
            config = r.config; customised = true
            placement = r.placement; deviceIds = r.deviceIds.map(Set.init); workerId = r.workerId
        }
        app?.remoteWatch.stop()   // training this project replaces whatever was being watched
        history.removeAll(); evalHistory.removeAll(); blockRows.removeAll(); retryDevices.removeAll(); retryGate = nil
        pauseControl = PauseController()
        stopping = false
        status = TrainerStatus(totalSteps: config.iterations)
        var cfg = config
        if !dataset.hasDepth { cfg.depthWeight = 0 }
        let name = dataset.root.deletingLastPathComponent().lastPathComponent
        let outDir = dataset.root.deletingLastPathComponent().appendingPathComponent("outputs", isDirectory: true)
        try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
        let stamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
        let project = dataset.root.deletingLastPathComponent()
        let out = resume.map { project.appendingPathComponent($0.output) } ?? outDir.appendingPathComponent("\(name)-\(stamp).ply")
        outputURL = out
        projectDir = project
        resumeBlockCount = resume?.blockCount
        lastRecorded = Date()
        let rel = String(out.path.dropFirst(project.path.count + 1))
        let (pl, devs, wid) = (placement, deviceIds.map { Array($0).sorted() }, workerId)
        ProjectLibrary.shared.update(project) { r in
            var t = ProjectRecord.Training(state: .running, config: cfg, placement: pl, deviceIds: devs, workerId: wid, output: rel,
                                           totalSteps: cfg.iterations)
            if let resume { t.started = resume.started; t.step = resume.checkpointStep ?? 0; t.checkpoint = resume.checkpoint
                t.checkpointStep = resume.checkpointStep; t.blockCount = resume.blockCount; t.blocksDone = resume.blocksDone }
            r.training = t
        }
        var resumeFrom: (URL, Int)? = nil
        if let r = resume, let ck = r.checkpoint, let st = r.checkpointStep, FileManager.default.fileExists(atPath: project.appendingPathComponent(ck).path) {
            resumeFrom = (project.appendingPathComponent(ck), st)
        }
        switch placement {
        case .thisMac: startLocal(dataset: dataset, config: cfg, output: out, resumeFrom: resumeFrom)
        case .remote: startRemote(dataset: dataset, config: cfg, output: out, name: name)
        case .both: startBoth(dataset: dataset, config: cfg, output: out, name: name, forceBlocks: resumeBlockCount != nil)
        }
    }

    private func startLocal(dataset: PreparedDataset, config cfg: TrainingConfig, output: URL, resumeFrom: (URL, Int)? = nil) {
        // Fail fast with what to change, instead of running out of GPU memory partway through loading.
        if MemoryForecast.macProblem(dataset: dataset, config: cfg) != nil {
            // Too big for this Mac in one piece: train it here as blocks that each fit, then merge.
            remoteMessage = "Too large for this Mac's memory in one piece — training it in blocks that fit, one after another."
            deviceIds = ["local"]
            startBoth(dataset: dataset, config: cfg, output: output, name: output.deletingPathExtension().lastPathComponent, forceBlocks: true)
            return
        }
        state = .preparing("Loading images")
        let t = MetalTrainer(dataset: dataset, config: cfg)
        trainer = t
        let preview = livePreview ? previewInterval : 0
        task = Task.detached(priority: .userInitiated) { [weak self] in
            do {
                if let st = RemoteStore.load(dataset.root), !st.local {
                    await MainActor.run { self?.state = .preparing("Downloading the frames from the worker") }
                    try await RemoteImageFetch.shared.ensure(dataset.root)
                }
                try t.prepare { msg in Task { @MainActor in self?.state = .preparing(msg) } }
                if let (url, step) = resumeFrom, let g = try? SplatFileFormat.load(url: url) {
                    t.resume(from: g, step: step)
                    await MainActor.run { self?.remoteMessage = "Continuing from the checkpoint at step \(step.formatted())." }
                }
                await MainActor.run { self?.state = .training; if let s = self { s.app?.setScene(cloud: nil, scene: t.makeScene(), frame: false) } }
                // Checkpoint every 10 minutes (and on Stop, below), so quitting or a crash costs at most that much.
                let saver = Task.detached(priority: .utility) {
                    while !Task.isCancelled {
                        try? await Task.sleep(nanoseconds: 600 * 1_000_000_000)
                        if Task.isCancelled { break }
                        await Self.saveCheckpoint(t, output: output, session: self)
                    }
                }
                defer { saver.cancel() }
                let result = try t.run(previewEvery: preview, onStatus: { s in
                    Task { @MainActor in self?.update(s) }
                }, onPreview: { scene in
                    Task { @MainActor in
                        guard let self, let app = self.app else { return }
                        app.scene = scene; app.renderer.scene = scene; app.refreshCounts()
                    }
                })
                try PLYCodec.writeGaussians(result, to: output, up: dataset.upVectorSIMD)
                let scene = SplatScene(cloud: result)
                await MainActor.run { self?.finish(result: result, scene: scene, output: output) }
            } catch TrainerError.cancelled {
                // Stop keeps the progress: Resume on the project page continues from here.
                if t.checkpoint().step > 0 { await Self.saveCheckpoint(t, output: output, session: self) }
                await MainActor.run { self?.state = .cancelled }
            } catch {
                await MainActor.run { self?.state = .failed(error.localizedDescription) }
            }
        }
    }

    /// Writes the trainer's splats and step to `checkpointURL(for: output)` (atomically) and notes it in the project.
    nonisolated private static func saveCheckpoint(_ t: MetalTrainer, output: URL, session: TrainingSession?) async {
        let (g, step) = t.checkpoint()
        guard step > 0 else { return }
        let url = checkpointURL(for: output)
        let tmp = url.deletingLastPathComponent().appendingPathComponent(".tmp-" + url.lastPathComponent)
        do {
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try PLYCodec.writeGaussians(g, to: tmp)
            try? FileManager.default.removeItem(at: url)
            try FileManager.default.moveItem(at: tmp, to: url)
            for ext in ["sky.png", "sky.json"] {      // the skybox sidecars travel with it
                let a = tmp.deletingPathExtension().appendingPathExtension(ext), b = url.deletingPathExtension().appendingPathExtension(ext)
                if FileManager.default.fileExists(atPath: a.path) { try? FileManager.default.removeItem(at: b); try? FileManager.default.moveItem(at: a, to: b) }
            }
        } catch { return }
        await MainActor.run {
            guard let session, let project = session.projectDir else { return }
            let rel = String(url.path.dropFirst(project.path.count + 1))
            session.recordTraining { $0.checkpoint = rel; $0.checkpointStep = step; $0.step = step }
        }
    }

    /// A training job running on a worker, saved next to the result-to-be (`<output>.remote-job.json`) as soon
    /// as the worker accepts it and removed once the result is saved. If the Mac sleeps, loses the network or
    /// the app quits, reopening the project re-attaches to the job (see `resumePendingRemote`).
    struct PendingRemoteJob: Codable {
        var workerId: String
        var baseURL: String
        var jobId: String
        var datasetId: String
        var name: String
        var output: String
        var startedAt: Date

        static func file(for output: URL) -> URL { URL(fileURLWithPath: output.path + ".remote-job.json") }
        func save() { try? JSONEncoder().encode(self).write(to: Self.file(for: URL(fileURLWithPath: output)), options: .atomic) }
        static func all(in dir: URL) -> [PendingRemoteJob] {
            let names = (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []
            return names.filter { $0.hasSuffix(".remote-job.json") }.compactMap {
                try? JSONDecoder().decode(PendingRemoteJob.self, from: Data(contentsOf: dir.appendingPathComponent($0)))
            }
        }
    }

    /// Updates status / live preview from a worker job's status (shared by new and re-attached jobs).
    private func remoteProgress(client: WorkerClient, start: Date) -> @Sendable (JobStatus) -> Void {
        var lastPreviewStep: Int? = nil, lastPreviewFetch = Date.distantPast, fetching = false
        return { [weak self] js in
            Task { @MainActor in
                guard let self else { return }
                if self.state != .paused { self.state = .training }
                if self.remoteMessage?.hasPrefix("Can't reach") == true { self.remoteMessage = nil }
                // Live preview: pull the worker's compact snapshot at most every 25 s.
                if self.livePreview, !fetching, let ps = js.previewStep, ps != lastPreviewStep,
                   Date().timeIntervalSince(lastPreviewFetch) > 25 {
                    fetching = true; lastPreviewStep = ps; lastPreviewFetch = Date()
                    let jobId = js.jobId
                    Task.detached {
                        let g = try? await client.previewSplat(jobId)
                        await MainActor.run {
                            fetching = false
                            if let g, self.livePreview, self.state == .training {
                                self.app?.setScene(cloud: nil, scene: SplatScene(cloud: g), frame: false)
                            }
                        }
                    }
                }
                var s = self.status
                s.step = js.step ?? s.step; s.totalSteps = js.totalSteps ?? s.totalSteps
                s.loss = Float(js.loss ?? Double(s.loss)); s.psnr = Float(js.psnr ?? Double(s.psnr))
                if let e = js.evalPsnr { s.evalPSNR = Float(e) }
                s.numGaussians = js.numGaussians ?? s.numGaussians
                s.elapsed = js.elapsedSeconds ?? Date().timeIntervalSince(start); s.eta = js.etaSeconds ?? 0
                s.itersPerSecond = s.elapsed > 0 ? Double(s.step) / s.elapsed : 0
                s.phase = js.message ?? "Training on \(client.baseURL.host ?? "worker")"
                self.update(s)
            }
        }
    }

    private func offlineNotice(_ host: String) -> @Sendable (TimeInterval) -> Void {
        { [weak self] gone in
            Task { @MainActor in
                self?.remoteMessage = "Can't reach \(host) for \(gone < 60 ? "\(Int(gone)) s" : "\(Int(gone / 60)) min") — the job keeps running there; retrying. You can close this project and reopen it later to pick the result up."
            }
        }
    }

    private func startRemote(dataset: PreparedDataset, config cfg: TrainingConfig, output: URL, name: String) {
        guard let id = workerId ?? WorkerBrowser.shared.workers.first?.id, let client = WorkerBrowser.shared.client(for: id) else {
            state = .failed("No Linux worker selected. Add one under Compute in the sidebar."); return
        }
        if let info = WorkerBrowser.shared.health[id]?.info,
           MemoryForecast.workerProblem(info: info, dataset: dataset, config: cfg) != nil {
            remoteMessage = "Too large for \(info.name)'s memory in one piece — training it there in blocks that fit, one after another."
            deviceIds = [id]
            startBoth(dataset: dataset, config: cfg, output: output, name: name, forceBlocks: true)
            return
        }
        state = .preparing("Uploading dataset")
        let host = client.baseURL.host ?? "the worker"
        let start = Date()
        let base = remoteProgress(client: client, start: start)
        let pause = pauseControl
        let progress: @Sendable (JobStatus) -> Void = { js in pause?.jobUpdated(js.jobId, state: js.state); base(js) }
        task = Task { [weak self] in
            do {
                let cloud = try await client.train(dataset: dataset, config: cfg, name: name, progress: progress,
                    uploadProgress: { f in Task { @MainActor in self?.state = .preparing("Uploading dataset \(Int(f * 100)) %") } },
                    onJobCreated: { jid, did in
                        pause?.add(job: jid, on: client)
                        PendingRemoteJob(workerId: id, baseURL: client.baseURL.absoluteString, jobId: jid, datasetId: did,
                                         name: name, output: output.path, startedAt: start).save()
                    },
                    onOffline: self?.offlineNotice(host))
                try PLYCodec.writeGaussians(cloud, to: output, up: dataset.upVectorSIMD)
                try? FileManager.default.removeItem(at: PendingRemoteJob.file(for: output))
                let scene = SplatScene(cloud: cloud)
                self?.finish(result: cloud, scene: scene, output: output)
            } catch is CancellationError {
                try? FileManager.default.removeItem(at: PendingRemoteJob.file(for: output))   // Stop also cancelled it there
                self?.state = .cancelled
            } catch {
                self?.state = .failed(error.localizedDescription)
            }
        }
    }

    /// Re-attaches to worker jobs recorded for this project (the Mac slept, lost the network or the app quit
    /// while they ran): resumes progress and live preview, or downloads the finished result straight away.
    func resumePendingRemote(dataset: PreparedDataset, outputsDir: URL) {
        guard !isActive, let p = PendingRemoteJob.all(in: outputsDir).sorted(by: { $0.startedAt > $1.startedAt }).first else { return }
        let output = URL(fileURLWithPath: p.output)
        let host = URL(string: p.baseURL)?.host ?? "the worker"
        workerId = p.workerId
        projectDir = dataset.root.deletingLastPathComponent()
        outputURL = output
        state = .preparing("Re-attaching to “\(p.name)” on \(host)")
        remoteMessage = "Picked up a job that was running on \(host) when this project was last open."
        task = Task { [weak self] in
            do {
                // Right after launch the worker may not be discovered yet (its client carries the access token).
                WorkerBrowser.shared.start()
                var known = WorkerBrowser.shared.client(for: p.workerId)
                for _ in 0..<20 where known == nil {
                    try await Task.sleep(nanoseconds: 500_000_000)
                    known = WorkerBrowser.shared.client(for: p.workerId)
                }
                guard let client = known ?? URL(string: p.baseURL).map({ WorkerClient(baseURL: $0) }) else { return }
                guard let base = self?.remoteProgress(client: client, start: p.startedAt) else { return }
                let pause = PauseController()
                self?.pauseControl = pause
                pause.add(job: p.jobId, on: client)
                let progress: @Sendable (JobStatus) -> Void = { js in pause.jobUpdated(js.jobId, state: js.state); base(js) }
                let cloud = try await client.finishJob(p.jobId, progress: progress, onOffline: self?.offlineNotice(host))
                try PLYCodec.writeGaussians(cloud, to: output, up: dataset.upVectorSIMD)
                try? FileManager.default.removeItem(at: PendingRemoteJob.file(for: output))
                self?.finish(result: cloud, scene: SplatScene(cloud: cloud), output: output)
            } catch is CancellationError {
                self?.state = .cancelled
            } catch {
                // Job gone (worker data cleared) or failed there: forget it so it isn't retried forever.
                if let w = error as? WorkerError, w.status == 404 || w.status == 500 {
                    try? FileManager.default.removeItem(at: PendingRemoteJob.file(for: output))
                }
                self?.state = .failed("Couldn't pick up the worker job: \(error.localizedDescription)")
            }
        }
    }

    /// - Parameter forceBlocks: split even small scenes and even on one device (the job doesn't fit in one piece).
    private func startBoth(dataset: PreparedDataset, config cfg: TrainingConfig, output: URL, name: String, forceBlocks: Bool = false) {
        let browser = WorkerBrowser.shared
        let useLocal = deviceIds?.contains("local") ?? true
        var devices = useLocal ? [ComputeScheduler.localDevice()] : []
        var clients: [String: WorkerClient] = [:]
        for w in browser.workers where deviceIds?.contains(w.id) ?? true {
            if let h = browser.health[w.id], case .online = h.status, let info = h.info, let c = browser.client(for: w.id) {
                devices.append(ComputeScheduler.remoteDevice(w, info: info)); clients[w.id] = c
            }
        }
        if clients.isEmpty && !forceBlocks {
            remoteMessage = "None of the chosen workers is online — training on this Mac only."
            startLocal(dataset: dataset, config: cfg, output: output); return
        }
        // Splitting only pays off for big scenes: below 400 images a block still sees most of the scene,
        // so two half-blocks cost nearly as much as one whole run each. Train it whole on the fastest device.
        if dataset.manifest.cameras.count < 400 && !forceBlocks {
            // Benchmarked machines: compare estimated wall time including the upload; otherwise raw throughput.
            let est = Dictionary(uniqueKeysWithValues: TrainingTimeEstimate.entries(dataset: dataset, config: cfg,
                                    workers: browser.workers.filter { clients[$0.id] != nil }).map { ($0.deviceId, $0.seconds) })
            let best = est.count == devices.count
                ? devices.min { (est[$0.id] ?? .infinity) < (est[$1.id] ?? .infinity) }
                : devices.max { $0.throughput < $1.throughput }
            if let best, case .remote = best.kind {
                remoteMessage = "\(dataset.manifest.cameras.count) images is too small to split — training the whole scene on \(best.name), the faster device."
                workerId = best.id
                startRemote(dataset: dataset, config: cfg, output: output, name: name)
            } else {
                remoteMessage = "\(dataset.manifest.cameras.count) images is too small to split — training the whole scene on this Mac, the faster device."
                startLocal(dataset: dataset, config: cfg, output: output)
            }
            return
        }
        // Blocks: two per device for balance, at least one per 2,000 photos, and more until every block fits on
        // at least one device's memory (a block's photos and its share of the splats).
        let infos = Dictionary(uniqueKeysWithValues: clients.keys.compactMap { id in browser.health[id]?.info.map { (id, $0) } })
        let fits: (String, SceneBlock, [SceneBlock]) -> Bool = { dev, b, all in
            let sub = dataset.restricted(to: Set(b.cameraIds))
            let c = ComputeScheduler.blockConfig(cfg, block: b, blocks: all)
            if dev == "local" { return MemoryForecast.macProblem(dataset: sub, config: c) == nil }
            guard let info = infos[dev] else { return true }
            return MemoryForecast.workerProblem(info: info, dataset: sub, config: c) == nil
        }
        let initialCount = resumeBlockCount ?? max(2, devices.count * 2, Int((Double(dataset.manifest.cameras.count) / 2000).rounded(.up)))
        let resuming = resumeBlockCount != nil
        let store = projectDir.map { Self.blockStore(project: $0, dataset: dataset, config: cfg) }
        state = .preparing("Partitioning scene into blocks")
        task = Task { [weak self] in
            do {
                let pc = dataset.pointCloudURL.flatMap { try? PLYCodec.readPointCloud(url: $0) }
                var blockCount = initialCount
                var blocks = BlockPartitioner(blockCount: blockCount).partition(dataset: dataset, points: pc)
                // More blocks only while that actually shrinks them: when every camera sees most of the scene
                // (gardens, rooms), splitting further keeps nearly all photos in every block.
                func widest(_ bs: [SceneBlock]) -> Int { bs.map(\.cameraIds.count).max() ?? 0 }
                while !resuming, blockCount < 32, !blocks.allSatisfy({ b in devices.contains { fits($0.id, b, blocks) } }) {
                    let next = BlockPartitioner(blockCount: blockCount + max(1, blockCount / 2)).partition(dataset: dataset, points: pc)
                    if Double(widest(next)) > 0.85 * Double(widest(blocks)) { break }
                    blockCount += max(1, blockCount / 2); blocks = next
                }
                self?.remoteMessage = "Training \(blocks.count) blocks on \(devices.count) devices."
                let bc = blocks.count
                self?.recordTraining { $0.placement = .both; $0.blockCount = bc }
                let scheduler = ComputeScheduler(devices: devices, placement: .both)
                let start = Date()
                let total = cfg.iterations * blocks.count
                var latest: [Int: ComputeScheduler.BlockProgress] = [:]
                var rateSamples: [(Date, Int)] = []
                var blockSamples: [Int: [(Date, Int)]] = [:]
                let finalBlocks = blocks
                let nDevices = devices.count
                self?.blockRows = []
                let gate = BlockRetryGate()
                let pause = self?.pauseControl
                self?.retryGate = gate
                self?.retryDevices = devices.filter { $0.capabilities.contains("train") || $0.isLocal }
                    .map { RetryDevice(id: $0.id, name: $0.isLocal ? "This Mac" : $0.name) }
                let cloud = try await scheduler.trainBlockParallel(dataset: dataset, config: cfg, blocks: blocks, name: name,
                                                                   local: useLocal ? MetalTrainingRunner(pause: pause) : nil, clients: clients,
                                                                   fits: { fits($0, $1, finalBlocks) }, retryGate: gate, pause: pause,
                                                                   blockStore: store) { bp in
                    Task { @MainActor in
                        guard let self else { return }
                        if case .preparing = self.state, bp.stage != .waiting { self.state = .training }
                        latest[bp.blockIndex] = bp
                        let now = Date()
                        // Per block: its own rate over the last 20 s.
                        var bs = blockSamples[bp.blockIndex, default: []]
                        if bp.stage == .training { bs.append((now, bp.snapshot.step)) }
                        bs.removeAll { now.timeIntervalSince($0.0) > 20 }
                        blockSamples[bp.blockIndex] = bs
                        let rate: Double = {
                            guard let f = bs.first, let l = bs.last, l.0.timeIntervalSince(f.0) > 1 else { return 0 }
                            return Double(l.1 - f.1) / l.0.timeIntervalSince(f.0)
                        }()
                        let dev = bp.deviceId == "local" ? "This Mac" : (browser.workers.first { $0.id == bp.deviceId }?.name ?? bp.deviceId)
                        var row = BlockRow(id: bp.blockIndex, deviceId: bp.deviceId, device: dev, stage: bp.stage, snapshot: bp.snapshot,
                                           stepsPerSecond: rate)
                        if let i = self.blockRows.firstIndex(where: { $0.id == bp.blockIndex }) {
                            if rate == 0, bp.stage == .training { row.stepsPerSecond = self.blockRows[i].stepsPerSecond }
                            self.blockRows[i] = row
                        } else {
                            self.blockRows.append(row); self.blockRows.sort { $0.id < $1.id }
                        }

                        // Combined: steps summed over blocks, quality averaged over blocks that have trained, splats summed.
                        let steps = latest.values.reduce(0) { $0 + $1.snapshot.step }
                        rateSamples.append((now, steps)); rateSamples.removeAll { now.timeIntervalSince($0.0) > 20 }
                        var s = self.status
                        s.totalSteps = total; s.step = min(total, steps)
                        if let first = rateSamples.first, now.timeIntervalSince(first.0) > 1 {
                            s.itersPerSecond = Double(steps - first.1) / now.timeIntervalSince(first.0)
                        }
                        let psnrs = latest.values.compactMap(\.snapshot.psnr).filter { $0 > 0 }, losses = latest.values.compactMap(\.snapshot.loss)
                        s.psnr = psnrs.isEmpty ? 0 : Float(psnrs.reduce(0, +) / Double(psnrs.count))
                        s.loss = losses.isEmpty ? 0 : Float(losses.reduce(0, +) / Double(losses.count))
                        s.numGaussians = latest.values.reduce(0) { $0 + $1.snapshot.numGaussians }
                        s.elapsed = now.timeIntervalSince(start)
                        // ETA: each device works through its blocks one after another, so the run ends when the slowest
                        // device finishes all of its remaining steps. Unknown until every device has a rate.
                        var deviceETA: [String: TimeInterval] = [:], unknown = false
                        for (devId, rows) in Dictionary(grouping: self.blockRows, by: \.deviceId) {
                            let remaining = rows.reduce(0) { $0 + max(0, $1.snapshot.totalSteps - $1.snapshot.step) }
                            let r = rows.map(\.stepsPerSecond).max() ?? 0
                            if remaining == 0 { deviceETA[devId] = 0 } else if r > 0.1 { deviceETA[devId] = Double(remaining) / r } else { unknown = true }
                        }
                        s.eta = unknown ? 0 : (deviceETA.values.max() ?? 0)
                        let counts = Dictionary(grouping: self.blockRows, by: { $0.stage.summaryKey }).mapValues(\.count)
                        let order = ["training", "uploading", "queued", "loading", "downloading", "waiting", "paused", "offline", "failed", "left out", "done"]
                        let parts = order.compactMap { k in counts[k].map { "\($0) \(k)" } }
                        s.phase = "\(finalBlocks.count) blocks on \(nDevices) device\(nDevices == 1 ? "" : "s"): " + parts.joined(separator: ", ")
                        if bp.stage == .done { let n = counts["done"] ?? 0; self.recordTraining { $0.blocksDone = n } }
                        self.update(s)
                    }
                }
                try PLYCodec.writeGaussians(cloud, to: output, up: dataset.upVectorSIMD)
                if let store { try? FileManager.default.removeItem(at: store) }     // merged: the blocks aren't needed any more
                self?.finish(result: cloud, scene: SplatScene(cloud: cloud), output: output)
            } catch is CancellationError {
                self?.state = .cancelled
            } catch {
                self?.state = .failed(error.localizedDescription)
            }
        }
    }

    private func update(_ s: TrainerStatus) {
        status = s
        if Date().timeIntervalSince(lastRecorded) > 30 {
            lastRecorded = Date()
            let (st, tot) = (s.step, s.totalSteps)
            recordTraining { $0.step = st; $0.totalSteps = tot }
        }
        if state == .preparing("") || { if case .preparing = state { return true }; return false }() { state = .training }
        if history.last?.step != s.step {
            history.append(Sample(id: history.count, step: s.step, loss: Double(s.loss), psnr: Double(s.psnr), gaussians: s.numGaussians))
            if history.count > 4000 { history = stride(from: 0, to: history.count, by: 2).map { history[$0] } }
        }
        if let e = s.evalPSNR, evalHistory.last?.psnr != Double(e) {
            evalHistory.append(Sample(id: evalHistory.count, step: s.step, loss: 0, psnr: Double(e), gaussians: s.numGaussians))
        }
    }

    private func finish(result: GaussianCloud, scene: SplatScene, output: URL) {
        state = .finished
        recordTraining { $0.state = .finished; $0.step = $0.totalSteps; $0.checkpoint = nil; $0.checkpointStep = nil; $0.message = nil }
        let ck = Self.checkpointURL(for: output)
        for u in [ck, ck.deletingPathExtension().appendingPathExtension("sky.png"), ck.deletingPathExtension().appendingPathExtension("sky.json")] {
            try? FileManager.default.removeItem(at: u)
        }
        ProjectLibrary.shared.note(output)
        trainer = nil
        if let app {
            app.setScene(cloud: result, scene: scene, frame: false)
            app.documentURL = output
            app.frameDataset()
        }
        NSSound(named: "Glass")?.play()
    }

    func pause() {
        guard state == .training else { return }
        trainer?.setPaused(true); pauseControl?.set(paused: true); state = .paused
    }
    func resume() {
        guard state == .paused else { return }
        trainer?.setPaused(false); pauseControl?.set(paused: false); state = .training
    }
    /// Stop was pressed and the run is winding down (worker jobs being cancelled, trainers reaching a safe point).
    var stopping = false
    func cancel() {
        if isActive { stopping = true }
        trainer?.cancel(); task?.cancel()     // a paused trainer or worker job is cancelled as it is
        if !isActive { state = .idle }
    }

    /// Save a checkpoint of the current parameters without stopping.
    func saveCheckpoint(to url: URL) throws {
        guard let g = trainer?.snapshot() else { return }
        try PLYCodec.writeGaussians(g, to: url)
    }
}

extension TrainingConfig {
    func withDepth(_ w: Float) -> TrainingConfig { var c = self; c.depthWeight = w; return c }
}

/// Dataset-preparation state for the New Project flow.
@MainActor
@Observable
final class PrepSession {
    weak var app: AppModel?
    var progress = PipelineProgress(stage: "", fraction: 0, message: "")
    var running = false
    var error: String? = nil
    var log: [String] = []
    var thenTrain = false

    // Timing (for elapsed / ETA / heartbeat)
    private(set) var startedAt: Date? = nil
    private(set) var lastUpdate: Date? = nil
    private(set) var stageStarts: [Phase: Date] = [:]
    private(set) var stageEnds: [Phase: Date] = [:]
    /// Last known overall fraction (unknown updates keep it).
    private(set) var overall: Double = 0
    private var stageETA: (seconds: Double, at: Date)? = nil
    private var samples: [(Date, Double)] = []
    private var lastLogged = Date.distantPast
    private var task: Task<Void, Never>?

    /// Coarse phases matching DatasetBuilder's progress ranges.
    enum Phase: Int, CaseIterable, Comparable {
        case images, masks, poses, finalising
        static func < (a: Phase, b: Phase) -> Bool { a.rawValue < b.rawValue }
        var title: String {
            switch self {
            case .images: return "Preparing images"
            case .masks: return "Masks"
            case .poses: return "Camera poses"
            case .finalising: return "Finalising dataset"
            }
        }
        static func of(_ f: Double) -> Phase { f < 0.35 ? .images : f < 0.45 ? .masks : f < 0.97 ? .poses : .finalising }
    }
    var phase: Phase { Phase.of(overall) }

    var elapsed: Double { startedAt.map { Date().timeIntervalSince($0) } ?? 0 }
    var sinceLastUpdate: Double { lastUpdate.map { Date().timeIntervalSince($0) } ?? 0 }

    /// Remaining seconds for the whole preparation: the tool's own estimate for the current stage when
    /// available (plus a rate-based estimate for what follows), otherwise extrapolated from recent progress.
    var eta: Double? {
        if let e = stageETA, phase == .poses {
            let left = max(0, e.seconds - Date().timeIntervalSince(e.at))
            return left + 3
        }
        guard samples.count >= 2, let first = samples.first(where: { Date().timeIntervalSince($0.0) < 120 }) ?? samples.first,
              let last = samples.last, last.1 > first.1 + 0.002 else { return nil }
        let rate = (last.1 - first.1) / max(last.0.timeIntervalSince(first.0), 1)
        return (1 - overall) / rate
    }

    func stageDuration(_ p: Phase) -> Double? {
        guard let s = stageStarts[p] else { return nil }
        return (stageEnds[p] ?? Date()).timeIntervalSince(s)
    }

    private func ingest(_ p: PipelineProgress) {
        let now = Date()
        lastUpdate = now
        if p.fraction >= 0 { overall = max(overall, p.fraction) }
        let ph = phase
        if stageStarts[ph] == nil {
            for prev in Phase.allCases where prev < ph && stageEnds[prev] == nil && stageStarts[prev] != nil { stageEnds[prev] = now }
            stageStarts[ph] = now
        }
        if let e = p.stageETA { stageETA = (e, now) }
        samples.append((now, overall))
        if samples.count > 400 { samples.removeFirst(samples.count - 400) }
        let stamp = Self.clock.string(from: now)
        if progress.stage != p.stage {
            appendLog("\(stamp)  \(p.stage)\(p.message.isEmpty ? "" : "  \(p.message)")")
            lastLogged = now
        } else if !p.message.isEmpty, now.timeIntervalSince(lastLogged) >= 5 {
            appendLog("\(stamp)    \(p.message)\(overall > 0 ? String(format: "  (%.0f %%)", overall * 100) : "")")
            lastLogged = now
        }
        if log.count > 3000 { log.removeFirst(log.count - 3000) }
        var q = p
        if q.fraction < 0 { q.fraction = overall }
        progress = q
        // The project page shows which stage was running if the app quits now.
        let st = Self.stage(of: ph)
        if st != recordedStage, let dir = projectDir, running {
            recordedStage = st
            ProjectLibrary.shared.update(dir) { $0.prep = .init(state: .running, stage: st) }
        }
    }

    /// `<project>/prep.log`: every progress line is also appended here, so a run that dies overnight
    /// (crash, power, memory pressure) still leaves a record of how far it got.
    private var logFile: URL?

    private func appendLog(_ line: String) {
        log.append(line)
        guard let url = logFile, let data = (Self.dayClock.string(from: Date()) + " " + line + "\n").data(using: .utf8) else { return }
        if let h = try? FileHandle(forWritingTo: url) {
            h.seekToEndOfFile(); h.write(data); try? h.close()
        } else {
            try? data.write(to: url)
        }
    }

    private static let dayClock: DateFormatter = { let f = DateFormatter(); f.dateFormat = "yyyy-MM-dd"; return f }()
    private static let clock: DateFormatter = { let f = DateFormatter(); f.dateFormat = "HH:mm:ss"; return f }()

    /// The project being prepared (or last prepared) and how, for Try Again / Resume.
    private(set) var projectDir: URL? = nil
    private(set) var lastRun: (source: SourceDescription, options: PrepOptions, thenTrain: Bool)? = nil
    private var recordedStage: PrepStage? = nil

    private static func stage(of ph: Phase) -> PrepStage {
        switch ph { case .images: return .frames; case .masks: return .masks; case .poses: return .poses; case .finalising: return .done }
    }

    /// Runs the last run again: every stage that finished is reused from its checkpoint.
    func retry() {
        guard let r = lastRun, let dir = projectDir else { return }
        run(source: r.source, options: r.options, projectDir: dir, thenTrain: r.thenTrain)
    }

    /// Continues (or redoes from `redo`) a project's processing from its saved settings. `workerId`: the device
    /// that processes (nil = this Mac).
    func resume(project dir: URL, redo: PrepStage? = nil, workerId: String?, thenTrain: Bool? = nil) {
        guard !running, let rec = ProjectRecord.load(dir), let src = rec.source, var o = rec.options else { return }
        if let redo { PrepCheckpoints(root: dir.appendingPathComponent("dataset")).invalidate(from: redo) }
        o.remoteWorkerId = workerId
        o.compute = workerId == nil ? .thisMac : .remote
        run(source: src, options: o, projectDir: dir, thenTrain: thenTrain ?? rec.trainAfter)
    }

    func run(source: SourceDescription, options: PrepOptions, projectDir: URL, thenTrain: Bool) {
        guard !running else { return }
        running = true; error = nil; log.removeAll(); cancelRequested = false
        self.projectDir = projectDir; lastRun = (source, options, thenTrain); recordedStage = nil
        try? FileManager.default.createDirectory(at: projectDir, withIntermediateDirectories: true)
        ProjectLibrary.shared.update(projectDir) { r in
            r.source = source; r.options = options; r.trainAfter = thenTrain
            r.prep = .init(state: .running, stage: .frames)
        }
        logFile = projectDir.appendingPathComponent("prep.log")
        self.thenTrain = thenTrain
        startedAt = Date(); lastUpdate = startedAt; overall = 0; samples.removeAll(); stageStarts.removeAll(); stageEnds.removeAll(); stageETA = nil
        progress = PipelineProgress(stage: "Starting", fraction: 0, message: "")
        let dsDir = projectDir.appendingPathComponent("dataset", isDirectory: true)
        let workerId = options.remoteWorkerId
        task = Task { [weak self] in
            do {
                // Right after launch the worker may not be discovered yet: wait for it rather than quietly doing the
                // work on this Mac.
                var worker: WorkerClient? = nil
                if let id = workerId {
                    WorkerBrowser.shared.start()
                    for _ in 0..<40 {
                        if let c = WorkerBrowser.shared.client(for: id) { worker = c; break }
                        try await Task.sleep(nanoseconds: 500_000_000)
                    }
                    if worker == nil { self?.ingest(PipelineProgress(stage: "Starting", fraction: 0, message: "worker not found on the network — using this Mac")) }
                    // Training-only workers (an iPhone running Compute) can't split, mask or solve: prepare here.
                    if let w = worker, let info = try? await w.info(timeout: 10), !info.canProcess {
                        worker = nil
                        self?.ingest(PipelineProgress(stage: "Starting", fraction: 0,
                                                      message: "\(info.name) only trains — preparing the dataset on this Mac"))
                    }
                }
                try FileManager.default.createDirectory(at: projectDir, withIntermediateDirectories: true)
                let cp = PrepCheckpoints(root: dsDir)
                let doneFP = PrepCheckpoints.fingerprints(source, options)[.done]!
                var leveled: PreparedDataset
                if cp.isDone(fingerprint: doneFP), let ready = try? PreparedDataset.load(dsDir) {
                    // Finished earlier with these settings (e.g. the app quit before training started).
                    leveled = ready
                    self?.ingest(PipelineProgress(stage: "Finalising dataset", fraction: 0.98, message: "the dataset is already prepared — reusing it"))
                } else {
                    let ds = try await DatasetBuilder.build(source: source, options: options, into: dsDir, worker: worker) { p in
                        Task { @MainActor in self?.ingest(p) }
                    }
                    self?.ingest(PipelineProgress(stage: "Finalising dataset", fraction: 0.98, message: "levelling and centring"))
                    leveled = ds
                    do { try leveled.normalizeOrientation(); self?.appendLog("Levelled scene: up = −Y, centred at origin") }
                    catch { self?.appendLog("Could not level scene: \(error.localizedDescription)") }
                    try? cp.markDone(fingerprint: doneFP, summary: "\(leveled.manifest.cameras.count) cameras, levelled")
                }
                guard let self else { return }
                for ph in Phase.allCases where self.stageEnds[ph] == nil && self.stageStarts[ph] != nil { self.stageEnds[ph] = Date() }
                self.overall = 1
                self.running = false
                self.appendLog("Dataset ready: \(leveled.manifest.cameras.count) cameras in \(formatDuration(self.elapsed))")
                ProjectLibrary.shared.update(projectDir) { $0.prep = .init(state: .finished, stage: .done) }
                self.app?.openDataset(leveled.root)
                self.app?.showNewProject = false
                if thenTrain { self.app?.training.start(dataset: leveled) }
            } catch {
                guard let self else { return }
                self.running = false
                // A cancel is not a failure: don't show "Preparation failed … CancellationError".
                let stage = Self.stage(of: self.phase)
                if self.cancelRequested || error is CancellationError || (error as? URLError)?.code == .cancelled {
                    ProjectLibrary.shared.update(projectDir) { $0.prep = .init(state: .stopped, stage: stage, message: "Stopped") }
                    return
                }
                let message = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
                self.appendLog("FAILED: \(message)")
                self.error = message
                ProjectLibrary.shared.update(projectDir) { $0.prep = .init(state: .failed, stage: stage, message: message) }
            }
        }
    }

    private var cancelRequested = false

    func cancel() { cancelRequested = true; appendLog("Cancelled"); task?.cancel(); running = false; error = nil }

    /// Clears a finished / failed / cancelled run so a new project starts on its settings, not the old result.
    func reset() {
        guard !running else { return }
        error = nil; log.removeAll(); cancelRequested = false; overall = 0
        progress = PipelineProgress(stage: "", fraction: 0, message: "")
        startedAt = nil; lastUpdate = nil; stageStarts.removeAll(); stageEnds.removeAll(); samples.removeAll(); stageETA = nil
    }
}
