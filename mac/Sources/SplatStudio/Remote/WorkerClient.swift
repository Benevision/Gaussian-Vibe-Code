import Foundation

// MARK: - API models (PROTOCOL.md §4, §6)

/// `GET /v1/info`
struct WorkerInfo: Codable, Hashable, Sendable {
    struct GPU: Codable, Hashable, Sendable { var name: String; var memoryMB: Int?; var freeMemoryMB: Int? }
    var name: String
    var version: Int?
    var gpus: [GPU]?
    var backend: String?
    var busy: Bool?
    var queueLength: Int?
    /// e.g. ["train", "process"]
    var capabilities: [String]?
    /// Importable splatprep models, e.g. ["superpoint_lightglue", …]
    var models: [String]?
    /// Worker ≥ 1.3.10: LAN addresses, preferred first (`kind` "ethernet" or "wifi").
    struct Address: Codable, Hashable, Sendable { var address: String; var kind: String? }
    var addresses: [Address]?
    /// Host RAM (worker ≥ 1.3.14).
    struct Memory: Codable, Hashable, Sendable { var totalMB: Int?; var availableMB: Int? }
    var memory: Memory?
    /// Worker ≥ 1.3.13: training images are streamed from disk when they don't fit in RAM.
    var streamsImages: Bool { capabilities?.contains("stream") ?? false }

    var canTrain: Bool { capabilities?.contains("train") ?? true }
    var canProcess: Bool { capabilities?.contains("process") ?? false }
    /// Worker ≥ 1.3 with PyAV: splits videos (incl. 360°) itself (`task: frames`).
    var canExtractFrames: Bool { capabilities?.contains("frames") ?? false }
    /// Worker ≥ 1.3.8: processing outputs stay on the worker as datasets (`outputDatasetId`), datasets can be
    /// composed (`/v1/datasets/compose`) and fetched selectively (`archive.zip?only=/skip=`).
    var canChain: Bool { capabilities?.contains("chain") ?? false }
}

/// Job state (tolerant: unknown values are preserved).
struct JobState: RawRepresentable, Codable, Hashable, Sendable, CustomStringConvertible {
    var rawValue: String
    init(rawValue: String) { self.rawValue = rawValue }
    init(from decoder: Decoder) throws { rawValue = try decoder.singleValueContainer().decode(String.self) }
    func encode(to encoder: Encoder) throws { var c = encoder.singleValueContainer(); try c.encode(rawValue) }

    static let queued = JobState(rawValue: "queued"), loading = JobState(rawValue: "loading"), training = JobState(rawValue: "training")
    static let running = JobState(rawValue: "running"), completed = JobState(rawValue: "completed")
    static let failed = JobState(rawValue: "failed"), cancelled = JobState(rawValue: "cancelled")
    var isTerminal: Bool { self == .completed || self == .failed || self == .cancelled }
    var description: String { rawValue }
}

/// `JobStatus` (training or processing job).
struct JobStatus: Codable, Hashable, Sendable, Identifiable {
    var jobId: String
    var name: String?
    /// "train" (default) | "process"
    var kind: String?
    var state: JobState
    var step: Int?
    var totalSteps: Int?
    var loss: Double?
    var psnr: Double?
    /// Held-out PSNR (worker ≥ 1.1.7); older workers put it in `psnr` once available.
    var evalPsnr: Double?
    var numGaussians: Int?
    var elapsedSeconds: Double?
    var etaSeconds: Double?
    var previewStep: Int?
    var message: String?
    /// Processing jobs on workers ≥ 1.3.8: the output, kept on the worker as a dataset for the next step.
    var outputDatasetId: String?
    var id: String { jobId }

    var fraction: Double {
        guard let s = step, let t = totalSteps, t > 0 else { return state == .completed ? 1 : 0 }
        return min(1, Double(s) / Double(t))
    }
}

struct WorkerError: LocalizedError, Sendable {
    var status: Int
    var message: String
    var errorDescription: String? { status == 401 ? "Worker rejected the access token (401)" : "Worker error \(status): \(message)" }
}

// MARK: - Client

/// HTTP client for a SplatWork Node (PROTOCOL.md §4 training API and §6 processing API).
final class WorkerClient: @unchecked Sendable {
    let baseURL: URL
    let token: String?
    private let session: URLSession

    /// - Parameters:
    ///   - baseURL: e.g. `http://192.168.1.20:7878`
    ///   - token: bearer token if the worker requires auth.
    init(baseURL: URL, token: String? = nil, session: URLSession? = nil) {
        self.baseURL = baseURL
        self.token = token?.isEmpty == true ? nil : token
        if let session { self.session = session } else {
            let cfg = URLSessionConfiguration.default
            cfg.timeoutIntervalForRequest = 60
            cfg.timeoutIntervalForResource = 24 * 3600
            cfg.waitsForConnectivity = false
            self.session = URLSession(configuration: cfg)
        }
    }

    convenience init(endpoint: WorkerEndpoint, token: String? = nil) {
        self.init(baseURL: endpoint.baseURL, token: token)
    }

    private func request(_ path: String, method: String = "GET", contentType: String? = nil, timeout: TimeInterval? = nil) -> URLRequest {
        var r = URLRequest(url: baseURL.appendingPathComponent(path))
        r.httpMethod = method
        if let token { r.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        if let contentType { r.setValue(contentType, forHTTPHeaderField: "Content-Type") }
        if let timeout { r.timeoutInterval = timeout }
        return r
    }

    private func check(_ data: Data, _ resp: URLResponse) throws {
        guard let h = resp as? HTTPURLResponse else { throw WorkerError(status: -1, message: "no HTTP response") }
        guard (200..<300).contains(h.statusCode) else {
            let msg = (try? JSONDecoder().decode([String: String].self, from: data))?["error"] ?? String(decoding: data.prefix(500), as: UTF8.self)
            throw WorkerError(status: h.statusCode, message: msg)
        }
    }

    private func json<T: Decodable>(_ r: URLRequest, as: T.Type = T.self) async throws -> T {
        let (d, resp) = try await session.data(for: r)
        try check(d, resp)
        return try JSONDecoder().decode(T.self, from: d)
    }

    private func jsonBody<B: Encodable>(_ path: String, _ body: B) throws -> URLRequest {
        var r = request(path, method: "POST", contentType: "application/json")
        r.httpBody = try JSONEncoder().encode(body)
        return r
    }

    // Remote install of optional processing components (worker ≥ 1.1.1)

    struct RuntimeStatus: Codable, Sendable {
        var state: String            // idle | running | done | failed
        var component: String?
        var step: String?
        var fraction: Double?
        var message: String?
        var log: [String]?
        var installed: [String: Bool]?
    }

    /// `GET /v1/runtime` — optional components and the progress of a remote install.
    func runtimeStatus() async throws -> RuntimeStatus { try await json(request("v1/runtime")) }

    /// `POST /v1/runtime/install` — installs a fixed component ("processing": COLMAP mapper, LightGlue,
    /// OWLv2, SAM 2) into the worker's GPU runtime; poll `runtimeStatus()` for progress.
    func installRuntime(_ component: String = "processing") async throws -> RuntimeStatus {
        try await json(jsonBody("v1/runtime/install", ["component": component]))
    }

    // §4

    /// `GET /v1/info` (also the health check).
    func info(timeout: TimeInterval = 5) async throws -> WorkerInfo { try await json(request("v1/info", timeout: timeout)) }

    /// `POST /v1/datasets` with a zip file. Returns `datasetId`.
    func uploadDataset(zip: URL, progress: (@Sendable (Double) -> Void)? = nil) async throws -> String {
        var r = request("v1/datasets", method: "POST", contentType: "application/zip")
        r.timeoutInterval = 3600
        let (d, resp) = try await session.upload(for: r, fromFile: zip, delegate: UploadProgressDelegate(progress))
        try check(d, resp)
        guard let id = try JSONDecoder().decode([String: String].self, from: d)["datasetId"] else { throw WorkerError(status: 200, message: "no datasetId") }
        return id
    }

    /// `POST /v1/datasets/file?name=…`: streams one file (e.g. a multi-GB video) as a dataset — no zip, no copy.
    func uploadFile(_ file: URL, name: String, progress: (@Sendable (Double) -> Void)? = nil) async throws -> String {
        let safe = String(name.map { $0.isLetter || $0.isNumber || "._- ".contains($0) ? $0 : "_" }).trimmingCharacters(in: CharacterSet(charactersIn: "."))
        var r = request("v1/datasets/file", method: "POST", contentType: "application/octet-stream")
        if let u = r.url, var comps = URLComponents(url: u, resolvingAgainstBaseURL: false) {
            comps.queryItems = [URLQueryItem(name: "name", value: safe.isEmpty ? "upload.bin" : safe)]
            r.url = comps.url
        }
        r.timeoutInterval = 3 * 3600
        let (d, resp) = try await session.upload(for: r, fromFile: file, delegate: UploadProgressDelegate(progress))
        try check(d, resp)
        guard let id = try JSONDecoder().decode([String: String].self, from: d)["datasetId"] else { throw WorkerError(status: 200, message: "no datasetId") }
        return id
    }

    /// Zips a directory (`ditto -c -k --keepParent`) and uploads it. Returns `datasetId`.
    func uploadDataset(directory: URL, progress: (@Sendable (Double) -> Void)? = nil) async throws -> String {
        let zip = FileManager.default.temporaryDirectory.appendingPathComponent("splat-upload-\(UUID().uuidString).zip")
        defer { try? FileManager.default.removeItem(at: zip) }
        try await ZipUtil.zip(directory: directory, to: zip)
        return try await uploadDataset(zip: zip, progress: progress)
    }

    struct CreateJob: Encodable { var datasetId: String; var name: String; var config: TrainingConfig }
    /// `POST /v1/jobs`. Returns `jobId`.
    func createJob(datasetId: String, name: String, config: TrainingConfig) async throws -> String {
        let r: [String: String] = try await json(jsonBody("v1/jobs", CreateJob(datasetId: datasetId, name: name, config: config)))
        guard let id = r["jobId"] else { throw WorkerError(status: 200, message: "no jobId") }
        return id
    }

    struct JobCameras: Decodable { var cameras: [DatasetCamera]; var sceneExtent: Float?; var upVector: [Float]? }
    /// `GET /v1/jobs/{id}/cameras` (worker ≥ 1.3.1): the training job's cameras, in its splats' coordinates.
    func jobCameras(_ id: String) async throws -> JobCameras { try await json(request("v1/jobs/\(id)/cameras")) }

    /// `GET /v1/jobs`
    func jobs() async throws -> [JobStatus] { try await json(request("v1/jobs")) }
    /// `GET /v1/jobs/{id}`
    func job(_ id: String) async throws -> JobStatus { try await json(request("v1/jobs/\(id)", timeout: 15)) }
    /// `POST /v1/jobs/{id}/cancel`
    @discardableResult func cancelJob(_ id: String) async throws -> JobStatus { try await json(request("v1/jobs/\(id)/cancel", method: "POST")) }
    /// `DELETE /v1/jobs/{id}`
    func deleteJob(_ id: String) async throws {
        let (d, resp) = try await session.data(for: request("v1/jobs/\(id)", method: "DELETE"))
        try check(d, resp)
    }
    /// `GET /v1/jobs/{id}/log`
    func log(_ id: String) async throws -> String {
        let (d, resp) = try await session.data(for: request("v1/jobs/\(id)/log"))
        try check(d, resp)
        return String(decoding: d, as: UTF8.self)
    }

    /// Downloads a job file to `destination`. Returns false on 404.
    @discardableResult
    func download(_ path: String, query: [URLQueryItem] = [], to destination: URL,
                  progress: (@Sendable (Double) -> Void)? = nil) async throws -> Bool {
        var r = request(path)
        if !query.isEmpty, let u = r.url, var comps = URLComponents(url: u, resolvingAgainstBaseURL: false) {
            comps.queryItems = query
            r.url = comps.url
        }
        r.timeoutInterval = 3600
        let (tmp, resp) = try await session.download(for: r, delegate: DownloadProgressDelegate(progress))
        if let h = resp as? HTTPURLResponse, h.statusCode == 404 { try? FileManager.default.removeItem(at: tmp); return false }
        if let h = resp as? HTTPURLResponse, !(200..<300).contains(h.statusCode) {
            let d = (try? Data(contentsOf: tmp)) ?? Data()
            try? FileManager.default.removeItem(at: tmp)
            try check(d, resp)
        }
        try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? FileManager.default.removeItem(at: destination)
        try FileManager.default.moveItem(at: tmp, to: destination)
        return true
    }

    /// `GET /v1/jobs/{id}/preview.ply` → GaussianCloud, nil until the first preview exists.
    func preview(_ id: String) async throws -> GaussianCloud? {
        let dst = FileManager.default.temporaryDirectory.appendingPathComponent("preview-\(id)-\(UUID().uuidString.prefix(6)).ply")
        defer { try? FileManager.default.removeItem(at: dst) }
        guard try await download("v1/jobs/\(id)/preview.ply", to: dst) else { return nil }
        return try PLYCodec.readGaussians(url: dst)
    }

    /// `GET /v1/jobs/{id}/preview.splat` → compact live preview (≤400 k splats, no SH), nil until one exists
    /// (or on workers older than 1.1 that only write the full preview.ply).
    func previewSplat(_ id: String) async throws -> GaussianCloud? {
        let dst = FileManager.default.temporaryDirectory.appendingPathComponent("preview-\(id)-\(UUID().uuidString.prefix(6)).splat")
        defer { try? FileManager.default.removeItem(at: dst) }
        guard try await download("v1/jobs/\(id)/preview.splat", to: dst) else { return nil }
        return try DotSplatCodec.read(url: dst)
    }

    /// `GET /v1/jobs/{id}/result.ply` saved to `destination`.
    func downloadResult(_ id: String, to destination: URL, progress: (@Sendable (Double) -> Void)? = nil) async throws {
        guard try await download("v1/jobs/\(id)/result.ply", to: destination, progress: progress) else {
            throw WorkerError(status: 404, message: "result not available")
        }
    }

    // §6

    struct ProcessBody: Encodable { var datasetId: String; var request: ProcessingRequest }
    /// `POST /v1/process` — returns the processing `jobId`.
    func process(datasetId: String, request: ProcessingRequest) async throws -> String {
        let r: [String: String] = try await json(jsonBody("v1/process", ProcessBody(datasetId: datasetId, request: request)))
        guard let id = r["jobId"] else { throw WorkerError(status: 200, message: "no jobId") }
        return id
    }

    /// `GET /v1/jobs/{id}/output.zip` (optionally without some top-level folders), unzipped into `directory`.
    func downloadProcessOutput(_ id: String, into directory: URL, skip: [String] = [],
                               progress: (@Sendable (Double) -> Void)? = nil) async throws {
        let zip = FileManager.default.temporaryDirectory.appendingPathComponent("output-\(id)-\(UUID().uuidString.prefix(6)).zip")
        defer { try? FileManager.default.removeItem(at: zip) }
        let q = skip.isEmpty ? [] : [URLQueryItem(name: "skip", value: skip.joined(separator: ","))]
        guard try await download("v1/jobs/\(id)/output.zip", query: q, to: zip, progress: progress) else {
            throw WorkerError(status: 404, message: "output not available")
        }
        try await ZipUtil.unzip(zip, to: directory)
    }

    /// `GET /v1/datasets/{id}/archive.zip?only=…` (worker ≥ 1.3.8), unzipped into `directory`.
    func downloadDatasetFiles(_ datasetId: String, only: [String], into directory: URL,
                              progress: (@Sendable (Double) -> Void)? = nil) async throws {
        let zip = FileManager.default.temporaryDirectory.appendingPathComponent("dataset-\(datasetId)-\(UUID().uuidString.prefix(6)).zip")
        defer { try? FileManager.default.removeItem(at: zip) }
        guard try await download("v1/datasets/\(datasetId)/archive.zip", query: [URLQueryItem(name: "only", value: only.joined(separator: ","))],
                                 to: zip, progress: progress) else {
            throw WorkerError(status: 404, message: "dataset \(datasetId) is no longer on the worker")
        }
        try await ZipUtil.unzip(zip, to: directory)
    }

    struct ComposeLayer: Encodable { var datasetId: String; var prefix: String? = nil }
    private struct ComposeBody: Encodable { var layers: [ComposeLayer] }
    private struct ComposeReply: Decodable { var datasetId: String }
    /// `POST /v1/datasets/compose` (worker ≥ 1.3.8): a new dataset from existing ones (later layers win).
    func compose(_ layers: [ComposeLayer]) async throws -> String {
        let r: ComposeReply = try await json(jsonBody("v1/datasets/compose", ComposeBody(layers: layers)))
        return r.datasetId
    }

    /// Uploads `directory` without the folders a worker already has (`store`), then composes the two there.
    /// Falls back to a full upload when the store isn't usable (other worker, or gone from it).
    func uploadReusing(_ store: RemoteStore?, directory: URL, progress: (@Sendable (Double) -> Void)? = nil) async throws -> String {
        if let store, store.worker == baseURL.absoluteString {
            let staging = FileManager.default.temporaryDirectory.appendingPathComponent("splat-delta-\(UUID().uuidString.prefix(8))/dataset")
            defer { try? FileManager.default.removeItem(at: staging.deletingLastPathComponent()) }
            try FileManager.default.ensureDirectory(staging)
            let needed = Self.referencedTopLevel(directory)
            for c in try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
            where !store.dirs.contains(c.lastPathComponent) && (needed == nil || needed!.contains(c.lastPathComponent)) {
                try FileManager.default.linkOrCopyTree(c, to: staging.appendingPathComponent(c.lastPathComponent))
            }
            do {
                let delta = try await uploadDataset(directory: staging, progress: progress)
                return try await compose([ComposeLayer(datasetId: store.datasetId), ComposeLayer(datasetId: delta)])
            } catch let e as WorkerError where e.status == 404 {
                // the worker no longer has the images: send everything (needs them here)
                try await RemoteImageFetch.shared.ensure(directory)
            }
        }
        return try await uploadDataset(directory: directory, progress: progress)
    }

    /// For a dataset folder with `dataset.json`: the top-level files and folders it refers to (images, masks,
    /// depth, points…) plus `dataset.json`. Intermediates next to it — the camera solve's work folder with its own
    /// copy of the images, raw mask outputs, frames.json — are left out of uploads. nil = no dataset.json.
    static func referencedTopLevel(_ directory: URL) -> Set<String>? {
        guard let d = try? Data(contentsOf: directory.appendingPathComponent("dataset.json")),
              let j = try? JSONSerialization.jsonObject(with: d) else { return nil }
        var out: Set<String> = ["dataset.json"]
        let fm = FileManager.default
        func walk(_ v: Any) {
            if let s = v as? String, !s.isEmpty, !s.hasPrefix("/"), s.count < 512,
               let first = s.split(separator: "/").first.map(String.init), first != "..",
               fm.fileExists(atPath: directory.appendingPathComponent(s).path) { out.insert(first) }
            else if let a = v as? [Any] { a.forEach(walk) }
            else if let o = v as? [String: Any] { o.values.forEach(walk) }
        }
        walk(j)
        return out
    }

    // MARK: High level

    /// Polls a job until it reaches a terminal state. Cancelling the Swift task cancels the remote job.
    /// Errors that mean "can't reach the worker right now" (Wi-Fi drop, Mac asleep, worker restarting) rather
    /// than "the job is gone". Waiting rides these out; the job keeps running on the worker regardless.
    static func isTransient(_ error: Error) -> Bool {
        if let u = error as? URLError {
            switch u.code {
            case .notConnectedToInternet, .networkConnectionLost, .timedOut, .cannotConnectToHost, .cannotFindHost,
                 .dnsLookupFailed, .dataNotAllowed, .internationalRoamingOff, .callIsActive, .secureConnectionFailed,
                 .resourceUnavailable, .backgroundSessionWasDisconnected, .badServerResponse:
                return true
            default: return false
            }
        }
        if let w = error as? WorkerError { return w.status >= 500 && w.status != 501 }   // proxy / restart hiccups
        let ns = error as NSError
        return ns.domain == NSPOSIXErrorDomain   // e.g. ECONNRESET
    }

    /// Polls until the job ends. Network trouble is retried (gaps growing to 30 s) for up to `maxOffline`, with
    /// `onOffline` told how long the worker has been unreachable; only the worker saying the job failed, the job
    /// disappearing (404) or that long an outage ends the wait with an error. Cancelling the task cancels the
    /// job on the worker.
    func waitForJob(_ id: String, pollInterval: TimeInterval = 2, maxOffline: TimeInterval = 12 * 3600,
                    onUpdate: (@Sendable (JobStatus) -> Void)? = nil,
                    onOffline: (@Sendable (TimeInterval) -> Void)? = nil) async throws -> JobStatus {
        var offlineSince: Date? = nil
        var backoff = pollInterval
        do {
            while true {
                do {
                    let s = try await job(id)
                    offlineSince = nil; backoff = pollInterval
                    onUpdate?(s)
                    if s.state.isTerminal { return s }
                    try await Task.sleep(nanoseconds: UInt64(pollInterval * 1e9))
                } catch {
                    if error is CancellationError || Task.isCancelled || (error as? URLError)?.code == .cancelled { throw error }
                    guard Self.isTransient(error) else { throw error }
                    let since = offlineSince ?? Date(); offlineSince = since
                    let gone = Date().timeIntervalSince(since)
                    if gone > maxOffline { throw WorkerError(status: 503, message: "worker unreachable for \(Int(gone / 60)) min: \(error.localizedDescription)") }
                    onOffline?(gone)
                    try await Task.sleep(nanoseconds: UInt64(backoff * 1e9))
                    backoff = min(backoff * 2, 30)
                }
            }
        } catch {
            // Cancelling surfaces as CancellationError or as URLError.cancelled from the in-flight poll.
            // Tell the worker from a detached task: requests made inside the cancelled task are aborted too.
            guard error is CancellationError || Task.isCancelled || (error as? URLError)?.code == .cancelled else { throw error }
            await Task.detached { [self] in _ = try? await self.cancelJob(id) }.value
            throw CancellationError()
        }
    }

    /// `downloadResult` that rides out network trouble (up to `maxOffline`).
    func downloadResultRetrying(_ id: String, to destination: URL, maxOffline: TimeInterval = 3600,
                                onOffline: (@Sendable (TimeInterval) -> Void)? = nil) async throws {
        let start = Date()
        var backoff: TimeInterval = 2
        while true {
            do { try await downloadResult(id, to: destination); return } catch {
                guard Self.isTransient(error), !Task.isCancelled else { throw error }
                let gone = Date().timeIntervalSince(start)
                if gone > maxOffline { throw error }
                onOffline?(gone)
                try await Task.sleep(nanoseconds: UInt64(backoff * 1e9)); backoff = min(backoff * 2, 30)
            }
        }
    }

    /// Uploads a prepared dataset, trains, and returns the result. `datasetId` may be reused for several jobs.
    func train(dataset: PreparedDataset, config: TrainingConfig, name: String, datasetId existing: String? = nil,
               progress: (@Sendable (JobStatus) -> Void)? = nil, uploadProgress: (@Sendable (Double) -> Void)? = nil,
               onJobCreated: (@Sendable (_ jobId: String, _ datasetId: String) -> Void)? = nil,
               onOffline: (@Sendable (TimeInterval) -> Void)? = nil) async throws -> GaussianCloud {
        let did: String
        if let existing { did = existing }
        else { did = try await uploadReusing(RemoteStore.usable(at: dataset.root, by: self), directory: dataset.root, progress: uploadProgress) }
        let jid = try await createJob(datasetId: did, name: name, config: config)
        onJobCreated?(jid, did)
        return try await finishJob(jid, progress: progress, onOffline: onOffline)
    }

    /// Waits for a training job (riding out disconnects) and downloads its result — also used to re-attach to
    /// a job after the Mac slept, lost the network or the app was relaunched.
    func finishJob(_ jid: String, progress: (@Sendable (JobStatus) -> Void)? = nil,
                   onOffline: (@Sendable (TimeInterval) -> Void)? = nil) async throws -> GaussianCloud {
        let final = try await waitForJob(jid, onUpdate: progress, onOffline: onOffline)
        guard final.state == .completed else { throw WorkerError(status: 500, message: "job \(final.state): \(final.message ?? "")") }
        let dst = FileManager.default.temporaryDirectory.appendingPathComponent("result-\(jid).ply")
        defer { try? FileManager.default.removeItem(at: dst) }
        try await downloadResultRetrying(jid, to: dst, onOffline: onOffline)
        var cloud = try PLYCodec.readGaussians(url: dst)
        // learned skybox (worker ≥ 1.3.15), when the run had one
        let skyPNG = dst.deletingPathExtension().appendingPathExtension("sky.png")
        let skyJSON = dst.deletingPathExtension().appendingPathExtension("sky.json")
        defer { try? FileManager.default.removeItem(at: skyPNG); try? FileManager.default.removeItem(at: skyJSON) }
        if (try? await download("v1/jobs/\(jid)/sky.png", to: skyPNG, progress: nil)) == true,
           (try? await download("v1/jobs/\(jid)/sky.json", to: skyJSON, progress: nil)) == true,
           let png = try? Data(contentsOf: skyPNG), let meta = try? Data(contentsOf: skyJSON) {
            cloud.skybox = GaussianCloud.Skybox(png: png, meta: meta)
        }
        return cloud
    }

    /// Runs a splatprep request remotely: uploads `directory` (containing `images/`, optional `priors.json`,
    /// `masks/`), waits, and unzips the output into `outputDirectory`.
    func runProcessing(directory: URL, request: ProcessingRequest, outputDirectory: URL,
                       progress: PipelineProgressHandler? = nil) async throws {
        try await runProcessing(upload: { cb in try await self.uploadDataset(directory: directory, progress: cb) },
                                request: request, outputDirectory: outputDirectory, progress: progress)
    }

    /// Same as `runProcessing(directory:…)` with a caller-chosen upload (e.g. `uploadFile` for a video, or an
    /// existing worker dataset). `skip`: top-level output folders not to download (they stay on the worker,
    /// in the returned status's `outputDatasetId`).
    @discardableResult
    func runProcessing(upload: (@escaping @Sendable (Double) -> Void) async throws -> String, request: ProcessingRequest,
                       outputDirectory: URL, skip: [String] = [], progress: PipelineProgressHandler? = nil,
                       existingJob: String? = nil, onJobCreated: ((String) -> Void)? = nil) async throws -> JobStatus {
        // Phases on the overall bar: upload 0–15 %, remote work 15–95 %, download 95–100 %.
        let host = baseURL.host ?? "worker"
        let task = request.task.rawValue
        let jid: String
        if let existingJob {
            // The same step already ran (or is running) on this worker: wait for / fetch that job.
            progress?(PipelineProgress(stage: "On \(host): \(task)", fraction: 0.15, message: "picking up the earlier \(task) job"))
            jid = existingJob
        } else {
            progress?(PipelineProgress(stage: "Uploading to \(host)", fraction: 0, message: "preparing \(task) job"))
            let did = try await upload { f in
                progress?(PipelineProgress(stage: "Uploading to \(host)", fraction: 0.15 * f, message: String(format: "%.0f %%", f * 100)))
            }
            jid = try await process(datasetId: did, request: request)
            onJobCreated?(jid)
        }
        let final = try await waitForJob(jid) { s in
            // Worker ≥ 1.1.2 reports "stage: detail" (e.g. "masks: SAM 2 forward: frame 120/649").
            let msg = s.message ?? ""
            let parts = msg.split(separator: ":", maxSplits: 1).map { $0.trimmingCharacters(in: .whitespaces) }
            let stage = parts.first.map { $0.isEmpty ? task : $0 } ?? task
            let detail = parts.count > 1 ? parts[1] : (s.state == .queued ? "waiting for the GPU" : s.state == .loading ? "starting" : "")
            progress?(PipelineProgress(stage: "On \(host): \(stage)", fraction: 0.15 + 0.8 * s.fraction, message: detail))
        }
        guard final.state == .completed else { throw PipelineError.processFailed("remote processing \(final.state): \(final.message ?? "")") }
        progress?(PipelineProgress(stage: "Downloading from \(host)", fraction: 0.95, message: "results of \(task)"))
        // Retry through network drops and sleep (the result stays on the worker): up to an hour offline.
        let start = Date()
        var backoff: TimeInterval = 2
        while true {
            do {
                try await downloadProcessOutput(jid, into: outputDirectory, skip: skip) { f in
                    progress?(PipelineProgress(stage: "Downloading from \(host)", fraction: 0.95 + 0.05 * f,
                                               message: String(format: "results of %@ — %.0f %%", task, f * 100)))
                }
                break
            } catch {
                guard Self.isTransient(error), !Task.isCancelled, Date().timeIntervalSince(start) < 3600 else { throw error }
                progress?(PipelineProgress(stage: "Downloading from \(host)", fraction: 0.95,
                                           message: "connection lost — retrying (the result is safe on the worker)"))
                try await Task.sleep(nanoseconds: UInt64(backoff * 1e9)); backoff = min(backoff * 2, 30)
            }
        }
        progress?(PipelineProgress(stage: "Downloading from \(host)", fraction: 1, message: "done"))
        return final
    }
}

/// splatprep SfM on a remote worker (`/v1/process`); same request as the local solver.
struct RemoteSplatPrepPoseSolver: PoseSolver {
    var client: WorkerClient
    var local: SplatPrepPoseSolver

    func solve(_ frames: FrameSet, progress: PipelineProgressHandler?) async throws -> PoseSolution {
        let req = local.request(for: frames)
        // Upload only what the worker needs: images/, masks/, priors — and not the images when the worker
        // already has them (it split the video: `RemoteStore`).
        let fm = FileManager.default
        let store = RemoteStore.usable(at: frames.root, by: client)
        if store == nil { try await FrameMasker.waitForFrames(frames.root, progress: progress) }
        let staging = frames.root.appendingPathComponent(".remote-upload/dataset")
        try? fm.removeItem(at: staging.deletingLastPathComponent())
        try fm.ensureDirectory(staging)
        defer { try? fm.removeItem(at: staging.deletingLastPathComponent()) }
        for sub in ["images", "masks"] + [req.priorPoses, req.skyMasksDir.map { _ in "splatprep-sky" }].compactMap({ $0 }) {
            let src = frames.root.appendingPathComponent(sub)
            try fm.linkOrCopyTree(src, to: staging.appendingPathComponent(sub))
        }
        let outDir = frames.root.appendingPathComponent(req.outputDir)
        let marker = outDir.appendingPathComponent(".remote-job")
        let unpack = frames.root.appendingPathComponent(".remote-output")
        try? fm.removeItem(at: unpack)
        // Undistorted output images (non-pinhole models) replace the originals, so they're needed here.
        let keepImagesRemote = store != nil && local.cameraModel == "pinhole"
        let worker = client.baseURL.absoluteString
        let fp = RemoteJobLedger.fingerprint(req, images: frames.frames.map(\.image))
        var reuse: String? = nil
        if let id = RemoteJobLedger.lookup(frames.root, request: req, fingerprint: fp, worker: worker),
           let st = try? await client.job(id), !(st.state == .failed || st.state == .cancelled) {
            reuse = id
        }
        let root = frames.root
        // The reused job's result is already here (a rerun after training failed, say): nothing to download.
        if let reuse, (try? String(contentsOf: marker, encoding: .utf8)) == reuse,
           fm.fileExists(atPath: outDir.appendingPathComponent("dataset.json").path) {
            progress?(PipelineProgress(stage: "Camera positions", fraction: 1, message: "already downloaded from the earlier job"))
            return try SplatPrepPoseSolver.readSolution(sfmDir: outDir, frames: frames, prefix: req.outputDir,
                                                        preferOutputImages: local.cameraModel != "pinhole")
        }
        try? fm.removeItem(at: outDir)
        var jobId = reuse
        try await client.runProcessing(upload: { cb in try await client.uploadReusing(store, directory: staging, progress: cb) },
                                       request: req, outputDirectory: unpack,
                                       // the solve's own copy of the images too (nested: workers ≥ 1.3.12)
                                       skip: keepImagesRemote ? ["images", "\(req.outputDir)/images"] : [],
                                       progress: progress, existingJob: reuse,
                                       onJobCreated: { jobId = $0; RemoteJobLedger.record(root, request: req, fingerprint: fp, worker: worker, jobId: $0) })
        // The zip may contain outputDir itself or its contents.
        let inner = [unpack.appendingPathComponent(req.outputDir), unpack].first { fm.fileExists(atPath: $0.appendingPathComponent("dataset.json").path) }
        guard let inner else { throw PipelineError.processFailed("remote output has no dataset.json") }
        try fm.moveItem(at: inner, to: outDir)
        try? fm.removeItem(at: unpack)
        if let jobId { try? jobId.write(to: marker, atomically: true, encoding: .utf8) }
        return try SplatPrepPoseSolver.readSolution(sfmDir: outDir, frames: frames, prefix: req.outputDir,
                                                    preferOutputImages: local.cameraModel != "pinhole")
    }
}

final class UploadProgressDelegate: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    let cb: (@Sendable (Double) -> Void)?
    init(_ cb: (@Sendable (Double) -> Void)?) { self.cb = cb }
    func urlSession(_ session: URLSession, task: URLSessionTask, didSendBodyData bytesSent: Int64, totalBytesSent: Int64, totalBytesExpectedToSend: Int64) {
        guard totalBytesExpectedToSend > 0 else { return }
        cb?(Double(totalBytesSent) / Double(totalBytesExpectedToSend))
    }
}

final class DownloadProgressDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    let cb: (@Sendable (Double) -> Void)?
    init(_ cb: (@Sendable (Double) -> Void)?) { self.cb = cb }
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        guard totalBytesExpectedToWrite > 0 else { return }
        cb?(Double(totalBytesWritten) / Double(totalBytesExpectedToWrite))
    }
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {}
}
