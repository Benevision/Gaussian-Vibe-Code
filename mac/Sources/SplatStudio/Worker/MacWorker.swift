import Foundation
import Network
import Observation

/// "Act as a worker": this Mac takes training jobs from other devices (an iPhone running SplatWork Capture, another
/// Mac) over the SplatWork Worker API — the same subset the iPhone's Compute screen serves (Worker/WorkerRouter.swift:
/// info, dataset upload, train jobs, result.ply, log, cameras). Advertised as `_splatworker._tcp` with
/// `kind=mac`; processing (`/v1/process`: learned masks, camera solving) is left to Linux workers.
@MainActor
@Observable
final class MacWorker {
    static let shared = MacWorker()
    /// Not 7878, so a Linux worker (or `python -m splatworker serve`) can run on the same Mac.
    static let port: UInt16 = 7879
    static let defaultsKey = "actAsWorker"

    private(set) var isEnabled = false
    private(set) var serverState: HTTPServer.State = .stopped
    private(set) var jobs: [WorkerJobStatus] = []
    private(set) var advertisedName = ""
    private var engine: WorkerEngine?
    private var server: HTTPServer?
    private var activity: NSObjectProtocol?
    private var refreshPending = false

    var activeJob: WorkerJobStatus? { jobs.last { $0.isRunning } ?? jobs.first { $0.state == "queued" } }
    var isTraining: Bool { jobs.contains { !$0.isTerminal } }

    /// Restores the switch from the last launch.
    func restore() { if UserDefaults.standard.bool(forKey: Self.defaultsKey) { setEnabled(true) } }

    func setEnabled(_ on: Bool) {
        UserDefaults.standard.set(on, forKey: Self.defaultsKey)
        guard on != isEnabled else { return }
        on ? start() : stop()
    }

    private func start() {
        let root = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("SplatStudio/Worker", isDirectory: true)
        let name = LocalDeviceName.current
        let e = WorkerEngine(name: name, root: root)
        e.onChange = { [weak self] in Task { @MainActor in self?.scheduleRefresh() } }
        let txt = NWTXTRecord(["version": "1", "gpu": e.gpuName, "auth": "0", "kind": "mac"])
        let service = NWListener.Service(name: name, type: WorkerBrowser.serviceType, domain: nil, txtRecord: txt)
        let s = HTTPServer(port: Self.port, service: service, tempDir: e.tempDir,
                           policy: { [e] m, p, len in e.bodyPolicy(method: m, path: p, contentLength: len) },
                           onState: { [weak self] st in Task { @MainActor in self?.serverState = st } },
                           handler: { [e] r in e.handle(r) })
        engine = e; server = s
        advertisedName = name
        WorkerBrowser.shared.ownNames.insert(name)   // don't list ourselves as a remote device
        isEnabled = true
        s.start()
    }

    private func stop() {
        engine?.failAllActive(message: "the Mac stopped acting as a worker")
        server?.stop()
        server = nil; engine = nil
        isEnabled = false
        serverState = .stopped
        WorkerBrowser.shared.ownNames.remove(advertisedName)
        refresh()
    }

    func cancel(_ id: String) { engine?.cancel(id, state: "cancelled", message: "cancelled on the Mac") }

    private func scheduleRefresh() {
        guard !refreshPending else { return }
        refreshPending = true
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 250_000_000)
            refreshPending = false
            refresh()
        }
    }

    private func refresh() {
        jobs = engine?.snapshot() ?? jobs.map { j in
            var j = j
            if !j.isTerminal { j.state = "failed"; j.message = "worker stopped" }
            return j
        }
        // Keep the Mac awake (no idle sleep / App Nap) while someone else's job trains here.
        if isTraining, activity == nil {
            activity = ProcessInfo.processInfo.beginActivity(options: [.idleSystemSleepDisabled, .userInitiated], reason: "training a worker job")
        } else if !isTraining, let a = activity {
            ProcessInfo.processInfo.endActivity(a); activity = nil
        }
    }
}
