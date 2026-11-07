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
        var splats = 0
        var previewStep: Int? = nil
        var message: String? = nil
    }

    private(set) var watched: Watched?
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
        task = Task { [weak self] in await self?.loop() }
    }

    func stop() {
        task?.cancel(); task = nil; watched = nil; cameras = nil; cameraExtent = nil
        app?.updateCameraLines()
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
            if let s = try? await client.job(w.jobId) {
                watched?.state = s.state.rawValue
                watched?.step = s.step ?? 0; watched?.totalSteps = s.totalSteps ?? 0
                watched?.psnr = s.psnr; watched?.splats = s.numGaussians ?? 0
                watched?.message = s.message
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
