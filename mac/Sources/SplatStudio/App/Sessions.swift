import Foundation
import SwiftUI
import Observation
import simd

/// Adapts the Metal trainer to the scheduler's runner protocol (used for local blocks).
struct MetalTrainingRunner: LocalTrainingRunner {
    func train(dataset: PreparedDataset, config: TrainingConfig,
               progress: @escaping @Sendable (TrainingProgressSnapshot) -> Void) async throws -> GaussianCloud {
        try await RemoteImageFetch.shared.ensure(dataset.root)
        return try await Task.detached(priority: .userInitiated) {
            let t = MetalTrainer(dataset: dataset, config: config)
            try t.prepare { _ in }
            return try t.run(previewEvery: 0, onStatus: { s in
                progress(TrainingProgressSnapshot(step: s.step, totalSteps: s.totalSteps, loss: Double(s.loss), psnr: Double(s.psnr),
                                                  numGaussians: s.numGaussians, elapsed: s.elapsed, eta: s.eta))
            })
        }.value
    }
}

/// Live training state for the dashboard.
@MainActor
@Observable
final class TrainingSession {
    weak var app: AppModel?

    enum State: Equatable { case idle, preparing(String), training, paused, finishing, finished, failed(String), cancelled }
    var state: State = .idle
    var preset: TrainingConfig.Preset = .balanced { didSet { if !customised { config = preset.config.withDepth(config.depthWeight) } } }
    var config = TrainingConfig.Preset.balanced.config
    var customised = false
    var placement: ComputePlacement = .thisMac
    var workerId: String? = nil
    var status = TrainerStatus()
    var remoteMessage: String? = nil
    var livePreview = true
    var previewInterval = 250
    var outputURL: URL? = nil

    struct Sample: Identifiable { let id: Int; let step: Int; let loss: Double; let psnr: Double; let gaussians: Int }
    var history: [Sample] = []
    var evalHistory: [Sample] = []

    private var trainer: MetalTrainer?
    private var task: Task<Void, Never>?

    var isActive: Bool {
        switch state { case .preparing, .training, .paused, .finishing: return true; default: return false }
    }
    var fraction: Double { status.totalSteps > 0 ? Double(status.step) / Double(status.totalSteps) : 0 }

    func trainerSnapshot() -> GaussianCloud? { trainer?.snapshot() }

    func start(dataset: PreparedDataset) {
        guard !isActive else { return }
        app?.remoteWatch.stop()   // training this project replaces whatever was being watched
        history.removeAll(); evalHistory.removeAll()
        status = TrainerStatus(totalSteps: config.iterations)
        var cfg = config
        if !dataset.hasDepth { cfg.depthWeight = 0 }
        let name = dataset.root.deletingLastPathComponent().lastPathComponent
        let outDir = dataset.root.deletingLastPathComponent().appendingPathComponent("outputs", isDirectory: true)
        try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
        let stamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
        let out = outDir.appendingPathComponent("\(name)-\(stamp).ply")
        outputURL = out
        switch placement {
        case .thisMac: startLocal(dataset: dataset, config: cfg, output: out)
        case .remote: startRemote(dataset: dataset, config: cfg, output: out, name: name)
        case .both: startBoth(dataset: dataset, config: cfg, output: out, name: name)
        }
    }

    private func startLocal(dataset: PreparedDataset, config cfg: TrainingConfig, output: URL) {
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
                await MainActor.run { self?.state = .training; if let s = self { s.app?.setScene(cloud: nil, scene: t.makeScene(), frame: false) } }
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
                await MainActor.run { self?.state = .cancelled }
            } catch {
                await MainActor.run { self?.state = .failed(error.localizedDescription) }
            }
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
                self.state = .training
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
        state = .preparing("Uploading dataset")
        let host = client.baseURL.host ?? "the worker"
        let start = Date()
        let progress = remoteProgress(client: client, start: start)
        task = Task { [weak self] in
            do {
                let cloud = try await client.train(dataset: dataset, config: cfg, name: name, progress: progress,
                    uploadProgress: { f in Task { @MainActor in self?.state = .preparing("Uploading dataset \(Int(f * 100)) %") } },
                    onJobCreated: { jid, did in
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
                guard let progress = self?.remoteProgress(client: client, start: p.startedAt) else { return }
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

    private func startBoth(dataset: PreparedDataset, config cfg: TrainingConfig, output: URL, name: String) {
        let browser = WorkerBrowser.shared
        var devices = [ComputeScheduler.localDevice()]
        var clients: [String: WorkerClient] = [:]
        for w in browser.workers {
            if let h = browser.health[w.id], case .online = h.status, let info = h.info, let c = browser.client(for: w.id) {
                devices.append(ComputeScheduler.remoteDevice(w, info: info)); clients[w.id] = c
            }
        }
        if clients.isEmpty {
            remoteMessage = "No online worker — training on this Mac only."
            startLocal(dataset: dataset, config: cfg, output: output); return
        }
        // Splitting only pays off for big scenes: below 400 images a block still sees most of the scene,
        // so two half-blocks cost nearly as much as one whole run each. Train it whole on the fastest device.
        if dataset.manifest.cameras.count < 400 {
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
        let blockCount = max(2, devices.count * 2)
        state = .preparing("Partitioning scene into \(blockCount) blocks")
        task = Task { [weak self] in
            do {
                let pc = dataset.pointCloudURL.flatMap { try? PLYCodec.readPointCloud(url: $0) }
                let blocks = BlockPartitioner(blockCount: blockCount).partition(dataset: dataset, points: pc)
                let scheduler = ComputeScheduler(devices: devices, placement: .both)
                let start = Date()
                let total = cfg.iterations * blocks.count
                var latest: [Int: ComputeScheduler.BlockProgress] = [:]
                var rateSamples: [(Date, Int)] = []
                let cloud = try await scheduler.trainBlockParallel(dataset: dataset, config: cfg, blocks: blocks, name: name,
                                                                   local: MetalTrainingRunner(), clients: clients) { bp in
                    Task { @MainActor in
                        guard let self else { return }
                        self.state = .training
                        latest[bp.blockIndex] = bp
                        // Combined view over all blocks: steps summed, quality averaged, splats summed.
                        let steps = latest.values.reduce(0) { $0 + $1.snapshot.step }
                        let now = Date()
                        rateSamples.append((now, steps)); rateSamples.removeAll { now.timeIntervalSince($0.0) > 20 }
                        var s = self.status
                        s.totalSteps = total; s.step = min(total, Int(bp.overall * Double(total)))
                        if let first = rateSamples.first, now.timeIntervalSince(first.0) > 1 {
                            s.itersPerSecond = Double(steps - first.1) / now.timeIntervalSince(first.0)
                        }
                        let psnrs = latest.values.compactMap(\.snapshot.psnr), losses = latest.values.compactMap(\.snapshot.loss)
                        s.psnr = psnrs.isEmpty ? 0 : Float(psnrs.reduce(0, +) / Double(psnrs.count))
                        s.loss = losses.isEmpty ? 0 : Float(losses.reduce(0, +) / Double(losses.count))
                        s.numGaussians = latest.values.reduce(0) { $0 + $1.snapshot.numGaussians }
                        s.elapsed = now.timeIntervalSince(start)
                        s.eta = bp.overall > 0.01 ? s.elapsed * (1 - bp.overall) / bp.overall : 0
                        s.phase = latest.keys.sorted().map { i in
                            let b = latest[i]!
                            let dev = b.deviceId == "local" ? "this Mac" : (browser.workers.first { $0.id == b.deviceId }?.name ?? b.deviceId)
                            return "Block \(i + 1)/\(blocks.count) on \(dev): \(Int(b.snapshot.fraction * 100)) %"
                        }.joined(separator: " · ")
                        self.update(s)
                    }
                }
                try PLYCodec.writeGaussians(cloud, to: output, up: dataset.upVectorSIMD)
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
        trainer = nil
        if let app {
            app.setScene(cloud: result, scene: scene, frame: false)
            app.documentURL = output
            app.frameDataset()
        }
        NSSound(named: "Glass")?.play()
    }

    func pause() { trainer?.setPaused(true); if state == .training { state = .paused } }
    func resume() { trainer?.setPaused(false); if state == .paused { state = .training } }
    func cancel() { trainer?.cancel(); task?.cancel(); if !isActive { state = .idle } }

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

    func run(source: SourceDescription, options: PrepOptions, projectDir: URL, thenTrain: Bool) {
        guard !running else { return }
        running = true; error = nil; log.removeAll(); cancelRequested = false
        logFile = projectDir.appendingPathComponent("prep.log")
        self.thenTrain = thenTrain
        startedAt = Date(); lastUpdate = startedAt; overall = 0; samples.removeAll(); stageStarts.removeAll(); stageEnds.removeAll(); stageETA = nil
        progress = PipelineProgress(stage: "Starting", fraction: 0, message: "")
        let dsDir = projectDir.appendingPathComponent("dataset", isDirectory: true)
        let worker: WorkerClient? = options.remoteWorkerId.flatMap { WorkerBrowser.shared.client(for: $0) }
        task = Task { [weak self] in
            do {
                try FileManager.default.createDirectory(at: projectDir, withIntermediateDirectories: true)
                let ds = try await DatasetBuilder.build(source: source, options: options, into: dsDir, worker: worker) { p in
                    Task { @MainActor in self?.ingest(p) }
                }
                guard let self else { return }
                self.ingest(PipelineProgress(stage: "Finalising dataset", fraction: 0.98, message: "levelling and centring"))
                var leveled = ds
                do { try leveled.normalizeOrientation(); self.appendLog("Levelled scene: up = −Y, centred at origin") }
                catch { self.appendLog("Could not level scene: \(error.localizedDescription)") }
                for ph in Phase.allCases where self.stageEnds[ph] == nil && self.stageStarts[ph] != nil { self.stageEnds[ph] = Date() }
                self.overall = 1
                self.running = false
                self.appendLog("Dataset ready: \(leveled.manifest.cameras.count) cameras in \(formatDuration(self.elapsed))")
                self.app?.openDataset(leveled.root)
                self.app?.showNewProject = false
                if thenTrain { self.app?.training.start(dataset: leveled) }
            } catch {
                guard let self else { return }
                self.running = false
                // A cancel is not a failure: don't show "Preparation failed … CancellationError".
                if self.cancelRequested || error is CancellationError || (error as? URLError)?.code == .cancelled { return }
                let message = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
                self.appendLog("FAILED: \(message)")
                self.error = message
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
