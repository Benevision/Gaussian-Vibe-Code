import Foundation
import Metal
import os
#if canImport(SplatTrainKit)
import SplatTrainKit   // iPhone app; on the Mac TrainKit.swift is part of the app itself
#endif

/// `JobStatus` as SplatWork's WorkerClient decodes it (docs/PROTOCOL.md §4).
struct WorkerJobStatus: Codable, Sendable, Equatable, Identifiable {
    var jobId: String
    var kind = "train"
    var name: String
    var state: String               // queued | loading | training | completed | failed | cancelled
    var step: Int = 0
    var totalSteps: Int
    var loss: Double? = nil
    var psnr: Double? = nil
    var evalPsnr: Double? = nil
    var numGaussians: Int? = nil
    var elapsedSeconds: Double? = nil
    var etaSeconds: Double? = nil
    var previewStep: Int? = nil
    var message: String? = nil
    var datasetId: String
    var createdAt: String
    var id: String { jobId }

    static let terminal: Set<String> = ["completed", "failed", "cancelled"]
    var isTerminal: Bool { Self.terminal.contains(state) }
    var isRunning: Bool { state == "loading" || state == "training" }
    var fraction: Double { totalSteps > 0 ? min(1, Double(step) / Double(totalSteps)) : (state == "completed" ? 1 : 0) }
}

/// The worker's state and request handling. Thread-safe (one lock); training runs one job at a time on a
/// private serial queue. Knows nothing about UIKit / AppKit — the app layer drives it: `ComputeWorker` on the
/// iPhone (Compute screen), `MacWorker` in SplatWork ("act as a worker"). Shared by both apps.
final class WorkerEngine: @unchecked Sendable {
    #if os(iOS)
    static let workerVersion = "ios-1.0"
    static let kind = "iphone"
    static let backend = "Metal (SplatWork Capture, iPhone)"
    static let deviceLabel = "the iPhone"
    #else
    static let workerVersion = "mac-1.0"
    static let kind = "mac"
    static let backend = "Metal (SplatWork, Mac)"
    static let deviceLabel = "the Mac"
    #endif
    static let logSubsystem = Bundle.main.bundleIdentifier ?? "SplatWorker"
    static let log = Logger(subsystem: WorkerEngine.logSubsystem, category: "worker")

    struct Info: Encodable {
        struct GPU: Encodable { var name: String; var memoryMB: Int; var freeMemoryMB: Int }
        var name: String
        var version = 1
        var workerVersion = WorkerEngine.workerVersion
        var gpus: [GPU]
        var backend = WorkerEngine.backend
        var busy: Bool
        var queueLength: Int
        var capabilities = ["train"]
        var models: [String] = []
        var auth = false
        var kind = WorkerEngine.kind
        /// Memory this app may still use (os_proc_available_memory on iOS), so a Mac can size jobs before sending.
        var memory: [String: Int] = ["availableMB": Int(TrainKit.availableMemory / 1_048_576)]
    }

    private final class Job {
        var status: WorkerJobStatus
        let configJSON: Data?
        let dir: URL
        var trainer: TrainKitJob?
        var log: [String] = []
        var started: Date?
        init(status: WorkerJobStatus, configJSON: Data?, dir: URL) { self.status = status; self.configJSON = configJSON; self.dir = dir }
    }

    let name: String
    let gpuName: String
    let root: URL
    /// Bearer token every request must carry (`Authorization: Bearer <token>`, PROTOCOL.md §4 Auth); nil = no auth.
    private let token: String?
    private var datasetsDir: URL { root.appendingPathComponent("datasets") }
    private var jobsDir: URL { root.appendingPathComponent("jobs") }
    var tempDir: URL { root.appendingPathComponent("tmp") }

    private let lock = NSLock()
    private var jobs: [String: Job] = [:]
    private var order: [String] = []
    /// datasetId → dataset root (directory containing dataset.json)
    private var datasets: [String: URL] = [:]
    private let trainQueue = DispatchQueue(label: "splatworker.train", qos: .userInitiated)
    /// Called (on any thread) whenever a job changes state or progress.
    var onChange: (@Sendable () -> Void)?

    /// - token: the access token clients must send on every request, or nil to serve anyone (advertise `auth=0`).
    init(name: String, root: URL, token: String?) {
        self.name = name
        self.root = root
        self.token = token
        gpuName = TrainKit.gpuName
        let fm = FileManager.default
        // Jobs and uploads don't survive a restart of the worker (the Mac re-uploads): start clean.
        try? fm.removeItem(at: root)
        for d in [datasetsDir, jobsDir, tempDir] { try? fm.createDirectory(at: d, withIntermediateDirectories: true) }
        var rv = URLResourceValues(); rv.isExcludedFromBackup = true
        var r = root; try? r.setResourceValues(rv)
    }

    // MARK: snapshots

    func snapshot() -> [WorkerJobStatus] {
        lock.lock(); defer { lock.unlock() }
        return order.compactMap { jobs[$0]?.status }
    }

    func logLines(_ id: String) -> [String] {
        lock.lock(); defer { lock.unlock() }
        return jobs[id]?.log ?? []
    }

    var hasActiveJob: Bool { snapshot().contains { !$0.isTerminal } }

    func info() -> Info {
        let s = snapshot()
        let total = Int(ProcessInfo.processInfo.physicalMemory / 1_048_576)
        let free = Int(TrainKit.availableMemory / 1_048_576)
        return Info(name: name, gpus: [.init(name: gpuName, memoryMB: total, freeMemoryMB: free)],
                    busy: s.contains { $0.isRunning }, queueLength: s.filter { $0.state == "queued" }.count,
                    auth: token != nil)
    }

    // MARK: HTTP

    /// Whether the request carries `Authorization: Bearer <token>` (always true without a token). The comparison
    /// takes the same time wherever the values differ, like the Linux worker's `hmac.compare_digest`.
    private func isAuthorized(_ headers: [String: String]) -> Bool {
        guard let token else { return true }
        let header = headers["authorization"] ?? ""
        guard let sp = header.firstIndex(of: " "), header[..<sp].lowercased() == "bearer" else { return false }
        let given = Array(header[header.index(after: sp)...].trimmingCharacters(in: .whitespaces).utf8)
        let want = Array(token.utf8)
        var diff: UInt8 = given.count == want.count ? 0 : 1
        for i in want.indices { diff |= want[i] ^ (i < given.count ? given[i] : 0) }
        return diff == 0
    }

    private static var unauthorized: HTTPResponse {
        var r = HTTPResponse.error(401, "missing or invalid bearer token")
        r.headers["WWW-Authenticate"] = "Bearer"
        return r
    }

    /// Decides how to read a request body before it arrives.
    func bodyPolicy(method: String, path: String, headers: [String: String], contentLength: UInt64?) -> HTTPBodyPolicy {
        // without the token nothing is read: no unauthenticated upload reaches memory or the disk
        guard isAuthorized(headers) else { return HTTPBodyPolicy(toFile: false, maxBytes: 0, reject: Self.unauthorized) }
        guard WorkerRoute.match(method: method, path: path) == .uploadDataset else {
            return HTTPBodyPolicy(toFile: false, maxBytes: 8 << 20)
        }
        guard let len = contentLength, len > 0 else {
            return HTTPBodyPolicy(toFile: true, maxBytes: 0, reject: .error(411, "Content-Length required"))
        }
        // the zip and its extracted copy exist side by side for a moment
        let free = Self.freeDiskBytes(root)
        let needed = len * 2 + 300 << 20
        if free < needed {
            return HTTPBodyPolicy(toFile: true, maxBytes: 0,
                                  reject: .error(507, "not enough storage on \(Self.deviceLabel): upload needs ~\(needed >> 20) MB, \(free >> 20) MB free"))
        }
        return HTTPBodyPolicy(toFile: true, maxBytes: len)
    }

    func handle(_ r: HTTPRequest) -> HTTPResponse {
        guard isAuthorized(r.headers) else {
            if let f = r.bodyFile { try? FileManager.default.removeItem(at: f) }
            return Self.unauthorized
        }
        let route = WorkerRoute.match(method: r.method, path: r.path)
        if route != .uploadDataset, let f = r.bodyFile { try? FileManager.default.removeItem(at: f) }
        switch route {
        case .info: return .json(info())
        case .uploadDataset: return uploadDataset(r)
        case .listDatasets:
            lock.lock(); let ids = datasets.keys.sorted(); lock.unlock()
            return .json(ids.map { ["datasetId": $0] })
        case .deleteDataset(let id):
            lock.lock(); let u = datasets.removeValue(forKey: id); lock.unlock()
            guard u != nil else { return .error(404, "dataset \(id) not found") }
            try? FileManager.default.removeItem(at: datasetsDir.appendingPathComponent(id))
            return .noContent
        case .createJob: return createJob(r)
        case .listJobs: return .json(snapshot())
        case .job(let id):
            lock.lock(); let s = jobs[id]?.status; lock.unlock()
            return s.map { .json($0) } ?? .error(404, "job \(id) not found")
        case .cancelJob(let id):
            return cancel(id, state: "cancelled", message: "cancelled").map { .json($0) } ?? .error(404, "job \(id) not found")
        case .deleteJob(let id):
            _ = cancel(id, state: "cancelled", message: "deleted")
            lock.lock()
            let j = jobs.removeValue(forKey: id)
            order.removeAll { $0 == id }
            lock.unlock()
            guard let j else { return .error(404, "job \(id) not found") }
            try? FileManager.default.removeItem(at: j.dir)
            onChange?()
            return .noContent
        case .result(let id):
            lock.lock(); let j = jobs[id]; let st = j?.status.state; lock.unlock()
            guard let j else { return .error(404, "job \(id) not found") }
            let u = j.dir.appendingPathComponent("result.ply")
            guard st == "completed", FileManager.default.fileExists(atPath: u.path) else {
                return .error(404, "result not available (job is \(st ?? "?"))")
            }
            return HTTPResponse(status: 200, contentType: "application/octet-stream",
                                headers: ["Content-Disposition": "attachment; filename=\"\(id).ply\""], body: .file(u, remove: false))
        case .log(let id):
            lock.lock(); let l = jobs[id]?.log; lock.unlock()
            return l.map { .text($0.joined(separator: "\n") + "\n") } ?? .error(404, "job \(id) not found")
        case .cameras(let id):
            lock.lock(); let did = jobs[id]?.status.datasetId; let root = did.flatMap { datasets[$0] }; lock.unlock()
            guard let root, let d = try? TrainKit.camerasJSON(root: root) else { return .error(404, "job has no dataset cameras") }
            return HTTPResponse(status: 200, body: .data(d))
        case .preview: return .error(404, "no preview on this worker")
        case .methodNotAllowed: return .error(405, "method not allowed")
        case .notFound: return .error(404, "not found: \(r.method) \(r.path)")
        }
    }

    // MARK: datasets

    private func uploadDataset(_ r: HTTPRequest) -> HTTPResponse {
        guard let zip = r.bodyFile else { return .error(400, "empty body") }
        defer { try? FileManager.default.removeItem(at: zip) }
        let ctype = (r.headers["content-type"] ?? "application/zip").split(separator: ";").first.map { $0.trimmingCharacters(in: .whitespaces).lowercased() } ?? ""
        guard ["application/zip", "application/x-zip-compressed", "application/octet-stream", "multipart/x-zip", ""].contains(ctype) else {
            return .error(415, "body must be a zip archive (Content-Type: application/zip)")
        }
        purgeUnusedDatasets()
        let id = "d" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(11).lowercased()
        let dir = datasetsDir.appendingPathComponent(id)
        let files = dir.appendingPathComponent("files")
        let t0 = Date()
        do {
            let free = Self.freeDiskBytes(root)
            let bytes = try ZipReader.extract(zip, to: files, maxBytes: free > 200 << 20 ? free - 200 << 20 : 0)
            let dsRoot = ZipReader.uploadRoot(files)
            let cams = try TrainKit.validateDataset(root: dsRoot)
            lock.lock(); datasets[id] = dsRoot; lock.unlock()
            Self.log.info("dataset \(id, privacy: .public): \(cams) cameras, \(bytes >> 20) MB in \(Date().timeIntervalSince(t0), format: .fixed(precision: 1)) s")
            return .json(["datasetId": id])
        } catch {
            try? FileManager.default.removeItem(at: dir)
            Self.log.error("dataset upload rejected: \(error.localizedDescription, privacy: .public)")
            if case ZipReader.ZipError.tooLarge = error { return .error(507, error.localizedDescription) }
            if error is ZipReader.ZipError { return .error(400, "rejected archive: \(error.localizedDescription)") }
            return .error(400, error.localizedDescription)
        }
    }

    /// Phone storage is tight: a new upload replaces datasets no unfinished job uses. (SplatWork uploads
    /// once per device and reuses the id for every block it sends, so older ones are done with.)
    private func purgeUnusedDatasets() {
        lock.lock()
        let inUse = Set(jobs.values.filter { !$0.status.isTerminal }.map(\.status.datasetId))
        let drop = datasets.keys.filter { !inUse.contains($0) }
        for d in drop { datasets.removeValue(forKey: d) }
        lock.unlock()
        for d in drop { try? FileManager.default.removeItem(at: datasetsDir.appendingPathComponent(d)) }
    }

    static func freeDiskBytes(_ url: URL) -> UInt64 {
        let v = try? url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return UInt64(max(0, v?.volumeAvailableCapacityForImportantUsage ?? 0))
    }

    // MARK: jobs

    private func createJob(_ r: HTTPRequest) -> HTTPResponse {
        guard let obj = (try? JSONSerialization.jsonObject(with: r.body)) as? [String: Any] else {
            return .error(400, "body must be a JSON object")
        }
        guard let did = obj["datasetId"] as? String else { return .error(400, "datasetId is required") }
        lock.lock(); let dsRoot = datasets[did]; lock.unlock()
        guard dsRoot != nil else { return .error(404, "dataset \(did) not found") }
        let name = String((obj["name"] as? String ?? did).prefix(200))
        var configJSON: Data? = nil
        if let c = obj["config"], !(c is NSNull) {
            guard JSONSerialization.isValidJSONObject(c), let d = try? JSONSerialization.data(withJSONObject: c) else {
                return .error(400, "invalid config: must be an object")
            }
            configJSON = d
        }
        let id = "j" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(11).lowercased()
        let status = WorkerJobStatus(jobId: id, name: name, state: "queued", totalSteps: TrainKit.iterations(configJSON: configJSON),
                                     message: "queued", datasetId: did, createdAt: ISO8601DateFormatter().string(from: Date()))
        let job = Job(status: status, configJSON: configJSON, dir: jobsDir.appendingPathComponent(id))
        try? FileManager.default.createDirectory(at: job.dir, withIntermediateDirectories: true)
        lock.lock(); jobs[id] = job; order.append(id); lock.unlock()
        appendLog(id, "job \(id) \"\(name)\" on dataset \(did)")
        onChange?()
        trainQueue.async { [weak self] in self?.run(id) }
        return .json(["jobId": id])
    }

    /// Ends a job that isn't finished yet with `state` (cancel, or "failed" when the app leaves the
    /// foreground). Returns the job's status, nil if unknown.
    @discardableResult
    func cancel(_ id: String, state: String, message: String) -> WorkerJobStatus? {
        lock.lock()
        guard let j = jobs[id] else { lock.unlock(); return nil }
        var trainer: TrainKitJob? = nil
        if !j.status.isTerminal {
            j.status.state = state
            j.status.message = message
            j.status.etaSeconds = nil
            if let s = j.started { j.status.elapsedSeconds = Date().timeIntervalSince(s) }
            trainer = j.trainer
        }
        let s = j.status
        lock.unlock()
        trainer?.cancel()
        if s.state == state { appendLog(id, "job \(state): \(message)") }
        onChange?()
        return s
    }

    /// Ends every unfinished job (used when iOS is about to suspend the app: GPU work isn't allowed in the background).
    func failAllActive(message: String) {
        for s in snapshot() where !s.isTerminal { cancel(s.jobId, state: "failed", message: message) }
    }

    private func appendLog(_ id: String, _ line: String) {
        let stamped = "[\(ISO8601DateFormatter().string(from: Date()))] \(line)"
        Self.log.info("\(id, privacy: .public): \(line, privacy: .public)")
        lock.lock(); jobs[id]?.log.append(stamped); lock.unlock()
    }

    private func update(_ id: String, _ f: (inout WorkerJobStatus) -> Void) {
        lock.lock()
        if let j = jobs[id], !j.status.isTerminal { f(&j.status) }
        lock.unlock()
        onChange?()
    }

    /// Runs on `trainQueue` (one job at a time).
    private func run(_ id: String) {
        lock.lock()
        guard let j = jobs[id], j.status.state == "queued", let dsRoot = datasets[j.status.datasetId] else {
            if let j = jobs[id], !j.status.isTerminal { j.status.state = "failed"; j.status.message = "dataset is gone" }
            lock.unlock(); onChange?(); return
        }
        j.status.state = "loading"
        j.status.message = "loading images"
        j.started = Date()
        let configJSON = j.configJSON, dir = j.dir
        lock.unlock()
        onChange?()

        let available = TrainKit.availableMemory
        let trainer: TrainKitJob
        do {
            trainer = try TrainKitJob(datasetRoot: dsRoot, configJSON: configJSON, availableBytes: available)
        } catch {
            appendLog(id, "rejected: \(error.localizedDescription)")
            update(id) { $0.state = "failed"; $0.message = error.localizedDescription }
            return
        }
        appendLog(id, trainer.budget.summary)
        Self.log.notice("\(trainer.budget.summary, privacy: .public)")
        lock.lock()
        j.trainer = trainer
        j.status.totalSteps = trainer.totalSteps
        let alreadyEnded = j.status.isTerminal
        lock.unlock()
        if alreadyEnded { trainer.cancel(); return }

        let started = Date()
        var lastNotify = Date.distantPast
        do {
            let count = try trainer.run(output: dir.appendingPathComponent("result.ply"), onPhase: { [weak self] phase in
                self?.appendLog(id, phase)
                self?.update(id) { s in
                    s.message = phase
                    if phase == "Training" { s.state = "training" }
                }
            }, onProgress: { [weak self] p in
                guard let self else { return }
                let now = Date()
                // the trainer reports ~10×/s; that's plenty for both the Mac's poll and the screen
                guard now.timeIntervalSince(lastNotify) > 0.25 || p.step == p.totalSteps else { return }
                lastNotify = now
                self.update(id) { s in
                    s.state = "training"
                    s.step = p.step; s.totalSteps = p.totalSteps
                    s.loss = p.loss; s.psnr = p.evalPsnr ?? p.psnr; s.evalPsnr = p.evalPsnr
                    s.numGaussians = p.numGaussians
                    s.elapsedSeconds = now.timeIntervalSince(started); s.etaSeconds = p.eta
                    s.message = p.phase
                }
            })
            appendLog(id, "completed: \(count) splats in \(Int(Date().timeIntervalSince(started))) s")
            update(id) { s in
                s.state = "completed"; s.message = "completed"; s.step = s.totalSteps
                s.numGaussians = count; s.etaSeconds = 0; s.elapsedSeconds = Date().timeIntervalSince(started)
            }
        } catch TrainKitError.cancelled {
            appendLog(id, "stopped")
            update(id) { $0.state = "cancelled"; $0.message = "cancelled" }   // no-op when cancel() already set it
        } catch {
            appendLog(id, "failed: \(error.localizedDescription)")
            update(id) { $0.state = "failed"; $0.message = error.localizedDescription }
        }
        lock.lock(); j.trainer = nil; lock.unlock()
        onChange?()
    }
}
