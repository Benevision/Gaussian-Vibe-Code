import Foundation
import Observation

/// Streams the live preview of any training job on a Linux worker into the viewer — including jobs this
/// Mac didn't start (another Mac, the command line, diagnostics). Polls the job every few seconds and
/// pulls its compact `preview.splat` whenever a new one exists; the camera is left where the user put it.
@MainActor
@Observable
final class RemoteJobWatcher {
    struct Watched: Equatable {
        var workerId: String
        var workerName: String
        var jobId: String
        var name: String
        var state: String = "connecting"
        var step = 0, totalSteps = 0
        var psnr: Double? = nil
        var evalPsnr: Double? = nil
        var loss: Double? = nil
        var splats = 0
        var elapsed: Double? = nil
        var eta: Double? = nil
        /// Smoothed from successive status polls (the worker reports elapsed time, not a rate).
        var stepsPerSecond: Double? = nil
        var previewStep: Int? = nil
        var message: String? = nil
    }

    private(set) var watched: Watched?
    /// Same samples as a local `TrainingSession`, so the Train tab draws the same charts. Backfilled from the
    /// job's log when watching starts mid-run.
    private(set) var history: [TrainingSession.Sample] = []
    private(set) var evalHistory: [TrainingSession.Sample] = []
    /// The watched job's cameras (nil = unknown: older worker). While watching, the viewer draws these
    /// instead of the open project's cameras, which belong to a different scene.
    private(set) var cameras: [DatasetCamera]? = nil
    private(set) var cameraExtent: Float? = nil
    @ObservationIgnored weak var app: AppModel?
    @ObservationIgnored private var task: Task<Void, Never>?

    func watch(workerId: String, jobId: String, name: String) {
        stop()
        let wname = WorkerBrowser.shared.workers.first { $0.id == workerId }?.name ?? "worker"
        watched = Watched(workerId: workerId, workerName: wname, jobId: jobId, name: name)
        history = []; evalHistory = []; lastPoll = nil
        app?.inspectorTab = .train
        task = Task { [weak self] in await self?.loop() }
    }

    func stop() {
        task?.cancel(); task = nil; watched = nil; cameras = nil; cameraExtent = nil
        history = []; evalHistory = []
        app?.updateCameraLines()
    }

    @ObservationIgnored private var lastPoll: (Int, Date)? = nil

    private func record(step: Int, psnr: Double?, eval: Double?, loss: Double?, splats: Int) {
        guard step > 0 else { return }
        if let p = psnr, p > 0, step > (history.last?.step ?? -1) {
            history.append(.init(id: history.count, step: step, loss: loss ?? 0, psnr: p, gaussians: splats))
            if history.count > 4000 { history = stride(from: 0, to: history.count, by: 2).enumerated().map { i, k in let h = history[k]; return .init(id: i, step: h.step, loss: h.loss, psnr: h.psnr, gaussians: h.gaussians) } }
        }
        if let e = eval, e > 0, evalHistory.last?.psnr != e {
            evalHistory.append(.init(id: evalHistory.count, step: step, loss: 0, psnr: e, gaussians: splats))
        }
    }

    /// Worker log lines: "step 1200/30000 loss 0.1234 psnr 16.12 gaussians 652122 sh 1" (a few per minute)
    /// and "step 2000: holdout PSNR 16.24 dB".
    private func backfill(_ log: String) {
        let train = try! NSRegularExpression(pattern: #"step (\d+)/\d+ loss ([\d.]+) psnr ([\d.]+) gaussians (\d+)"#)
        let held = try! NSRegularExpression(pattern: #"step (\d+): holdout PSNR ([\d.]+) dB"#)
        var lastSplats = 0
        for line in log.split(separator: "\n") {
            let l = String(line), r = NSRange(l.startIndex..., in: l)
            func g(_ m: NSTextCheckingResult, _ i: Int) -> String { Range(m.range(at: i), in: l).map { String(l[$0]) } ?? "" }
            if let m = train.firstMatch(in: l, range: r) {
                lastSplats = Int(g(m, 4)) ?? lastSplats
                record(step: Int(g(m, 1)) ?? 0, psnr: Double(g(m, 3)), eval: nil, loss: Double(g(m, 2)), splats: lastSplats)
            } else if let m = held.firstMatch(in: l, range: r) {
                record(step: Int(g(m, 1)) ?? 0, psnr: nil, eval: Double(g(m, 2)), loss: nil, splats: lastSplats)
            }
        }
    }

    private func loop() async {
        var shownPreview: Int? = nil
        var first = true
        app?.updateCameraLines()   // hide the open project's cameras straight away
        while !Task.isCancelled, let w = watched {
            guard let client = WorkerBrowser.shared.client(for: w.workerId) else {
                watched?.message = "\(w.workerName) is offline"; try? await Task.sleep(nanoseconds: 5_000_000_000); continue
            }
            if cameras == nil, let jc = try? await client.jobCameras(w.jobId) {
                cameras = jc.cameras; cameraExtent = jc.sceneExtent
                app?.updateCameraLines()
            }
            if first, history.isEmpty, let log = try? await client.log(w.jobId) { backfill(log) }
            if let s = try? await client.job(w.jobId) {
                let now = Date()
                if let (prevStep, prevTime) = lastPoll, let st = s.step, st > prevStep {
                    let rate = Double(st - prevStep) / max(now.timeIntervalSince(prevTime), 0.1)
                    watched?.stepsPerSecond = watched?.stepsPerSecond.map { 0.7 * $0 + 0.3 * rate } ?? rate
                }
                if let st = s.step { lastPoll = (st, now) }
                watched?.state = s.state.rawValue
                watched?.step = s.step ?? 0; watched?.totalSteps = s.totalSteps ?? 0
                watched?.psnr = s.psnr; watched?.evalPsnr = s.evalPsnr; watched?.loss = s.loss
                watched?.splats = s.numGaussians ?? 0
                watched?.elapsed = s.elapsedSeconds; watched?.eta = s.etaSeconds
                watched?.message = s.message
                record(step: s.step ?? 0, psnr: s.psnr, eval: s.evalPsnr, loss: s.loss, splats: s.numGaussians ?? 0)
                if let ps = s.previewStep, ps != shownPreview, let g = try? await client.previewSplat(w.jobId) {
                    shownPreview = ps
                    watched?.previewStep = ps
                    app?.setScene(cloud: nil, scene: SplatScene(cloud: g), frame: first)
                    first = false
                }
                if s.state.isTerminal { return }   // keep the banner (final numbers) until the user closes it
            }
            try? await Task.sleep(nanoseconds: 4_000_000_000)
        }
    }
}
