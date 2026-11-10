import Foundation
import Network
import Observation
import UIKit

/// App-side owner of the training worker: the switch on the Compute screen, the HTTP server + Bonjour
/// advertisement (`_splatworker._tcp`, PROTOCOL.md §4), screen-awake handling and background behaviour.
@MainActor
@Observable
final class ComputeWorker {
    static let port: UInt16 = 7878
    static let serviceType = "_splatworker._tcp"

    private(set) var isEnabled = false
    private(set) var serverState: HTTPServer.State = .stopped
    private(set) var jobs: [WorkerJobStatus] = []
    private(set) var addresses: [String] = []
    private(set) var gpuName = ""
    private(set) var lastLog: [String] = []

    private var engine: WorkerEngine?
    private var server: HTTPServer?
    private var refreshPending = false
    private var backgroundTask: UIBackgroundTaskIdentifier = .invalid

    var activeJob: WorkerJobStatus? { jobs.last { $0.isRunning } ?? jobs.first { $0.state == "queued" } }
    var isTraining: Bool { jobs.contains { !$0.isTerminal } }

    func setEnabled(_ on: Bool) {
        guard on != isEnabled else { return }
        on ? start() : stop()
    }

    private func start() {
        let root = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("SplatWorker", isDirectory: true)
        let name = DeviceInfo.modelName
        let e = WorkerEngine(name: name, root: root, token: nil)   // no access token (advertised auth=0)
        e.onChange = { [weak self] in
            Task { @MainActor in self?.scheduleRefresh() }
        }
        gpuName = e.gpuName
        let txt = NWTXTRecord(["version": "1", "gpu": e.gpuName, "auth": "0", "kind": "iphone"])
        let service = NWListener.Service(name: name, type: Self.serviceType, domain: nil, txtRecord: txt)
        let s = HTTPServer(port: Self.port, service: service, tempDir: e.tempDir,
                           policy: { [e] m, p, h, len in e.bodyPolicy(method: m, path: p, headers: h, contentLength: len) },
                           onState: { [weak self] st in Task { @MainActor in self?.serverState = st } },
                           handler: { [e] r in e.handle(r) })
        engine = e
        server = s
        isEnabled = true
        addresses = Self.lanAddresses()
        s.start()
    }

    private func stop() {
        engine?.failAllActive(message: "training was switched off on the iPhone")
        server?.stop()
        server = nil
        engine = nil
        isEnabled = false
        serverState = .stopped
        refresh()
    }

    func cancel(_ id: String) { engine?.cancel(id, state: "cancelled", message: "cancelled on the iPhone") }

    // MARK: lifecycle

    /// iOS suspends GPU work in the background: end the running job cleanly so the Mac learns why.
    func appDidEnterBackground() {
        guard let engine, engine.hasActiveJob else { return }
        // A few seconds of background time to let the current step finish and the trainer unwind.
        backgroundTask = UIApplication.shared.beginBackgroundTask(withName: "stop training") { [weak self] in
            MainActor.assumeIsolated { self?.endBackgroundTask() }
        }
        engine.failAllActive(message: "app moved to background")
        refresh()
        Task { @MainActor in
            try? await Task.sleep(for: .seconds(3))
            endBackgroundTask()
        }
    }

    func appDidBecomeActive() {
        guard isEnabled else { return }
        addresses = Self.lanAddresses()
        // The listener may have been torn down while suspended.
        if case .failed = serverState { server?.stop(); server?.start() }
        if serverState == .stopped { server?.start() }
        refresh()
    }

    private func endBackgroundTask() {
        if backgroundTask != .invalid { UIApplication.shared.endBackgroundTask(backgroundTask); backgroundTask = .invalid }
    }

    // MARK: state

    private func scheduleRefresh() {
        guard !refreshPending else { return }
        refreshPending = true
        Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(250))
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
        if let id = activeJob?.jobId ?? jobs.last?.jobId { lastLog = Array((engine?.logLines(id) ?? []).suffix(6)) }
        // Keep the phone awake while a job runs (the screen locking would background the app and stop it).
        UIApplication.shared.isIdleTimerDisabled = isTraining
    }

    /// IPv4 addresses of Wi-Fi / Ethernet interfaces, for display ("192.168.0.42:7878").
    static func lanAddresses() -> [String] {
        var out: [String] = []
        var ifap: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&ifap) == 0, let first = ifap else { return [] }
        defer { freeifaddrs(ifap) }
        for p in sequence(first: first, next: { $0.pointee.ifa_next }) {
            let i = p.pointee
            guard let sa = i.ifa_addr, sa.pointee.sa_family == UInt8(AF_INET) else { continue }
            let name = String(cString: i.ifa_name)
            guard name.hasPrefix("en") else { continue }
            var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            if getnameinfo(sa, socklen_t(sa.pointee.sa_len), &host, socklen_t(host.count), nil, 0, NI_NUMERICHOST) == 0 {
                out.append(String(decoding: host.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self))
            }
        }
        return out
    }
}
