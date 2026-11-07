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

    var canTrain: Bool { capabilities?.contains("train") ?? true }
    var canProcess: Bool { capabilities?.contains("process") ?? false }
    /// Worker ≥ 1.3 with PyAV: splits videos (incl. 360°) itself (`task: frames`).
    var canExtractFrames: Bool { capabilities?.contains("frames") ?? false }
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

/// HTTP client for a Splat Worker (PROTOCOL.md §4 training API and §6 processing API).
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

    struct JobCameras: Decodable { var cameras: [DatasetCamera]; var sceneExtent: Float? }
    /// `GET /v1/jobs/{id}/cameras` (worker ≥ 1.3.1): the training job's cameras, in its splats' coordinates.
    func jobCameras(_ id: String) async throws -> JobCameras { try await json(request("v1/jobs/\(id)/cameras")) }

    /// `GET /v1/jobs`
    func jobs() async throws -> [JobStatus] { try await json(request("v1/jobs")) }
    /// `GET /v1/jobs/{id}`
    func job(_ id: String) async throws -> JobStatus { try await json(request("v1/jobs/\(id)")) }
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
    func download(_ path: String, to destination: URL, progress: (@Sendable (Double) -> Void)? = nil) async throws -> Bool {
        var r = request(path)
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

    /// `GET /v1/jobs/{id}/output.zip`, unzipped into `directory`.
    func downloadProcessOutput(_ id: String, into directory: URL) async throws {
        let zip = FileManager.default.temporaryDirectory.appendingPathComponent("output-\(id)-\(UUID().uuidString.prefix(6)).zip")
        defer { try? FileManager.default.removeItem(at: zip) }
        guard try await download("v1/jobs/\(id)/output.zip", to: zip) else { throw WorkerError(status: 404, message: "output not available") }
        try await ZipUtil.unzip(zip, to: directory)
    }

    // MARK: High level

    /// Polls a job until it reaches a terminal state. Cancelling the Swift task cancels the remote job.
    func waitForJob(_ id: String, pollInterval: TimeInterval = 2, onUpdate: (@Sendable (JobStatus) -> Void)? = nil) async throws -> JobStatus {
        do {
            while true {
                let s = try await job(id)
                onUpdate?(s)
                if s.state.isTerminal { return s }
                try await Task.sleep(nanoseconds: UInt64(pollInterval * 1e9))
            }
        } catch {
            // Cancelling surfaces as CancellationError or as URLError.cancelled from the in-flight poll.
            // Tell the worker from a detached task: requests made inside the cancelled task are aborted too.
            guard error is CancellationError || Task.isCancelled || (error as? URLError)?.code == .cancelled else { throw error }
            await Task.detached { [self] in _ = try? await self.cancelJob(id) }.value
            throw CancellationError()
        }
    }

    /// Uploads a prepared dataset, trains, and returns the result. `datasetId` may be reused for several jobs.
    func train(dataset: PreparedDataset, config: TrainingConfig, name: String, datasetId existing: String? = nil,
               progress: (@Sendable (JobStatus) -> Void)? = nil, uploadProgress: (@Sendable (Double) -> Void)? = nil) async throws -> GaussianCloud {
        let did: String
        if let existing { did = existing } else { did = try await uploadDataset(directory: dataset.root, progress: uploadProgress) }
        let jid = try await createJob(datasetId: did, name: name, config: config)
        let final = try await waitForJob(jid, onUpdate: progress)
        guard final.state == .completed else { throw WorkerError(status: 500, message: "job \(final.state): \(final.message ?? "")") }
        let dst = FileManager.default.temporaryDirectory.appendingPathComponent("result-\(jid).ply")
        defer { try? FileManager.default.removeItem(at: dst) }
        try await downloadResult(jid, to: dst)
        return try PLYCodec.readGaussians(url: dst)
    }

    /// Runs a splatprep request remotely: uploads `directory` (containing `images/`, optional `priors.json`,
    /// `masks/`), waits, and unzips the output into `outputDirectory`.
    func runProcessing(directory: URL, request: ProcessingRequest, outputDirectory: URL,
                       progress: PipelineProgressHandler? = nil) async throws {
        try await runProcessing(upload: { cb in try await self.uploadDataset(directory: directory, progress: cb) },
                                request: request, outputDirectory: outputDirectory, progress: progress)
    }

    /// Same as `runProcessing(directory:…)` with a caller-chosen upload (e.g. `uploadFile` for a video).
    func runProcessing(upload: (@escaping @Sendable (Double) -> Void) async throws -> String, request: ProcessingRequest,
                       outputDirectory: URL, progress: PipelineProgressHandler? = nil) async throws {
        // Phases on the overall bar: upload 0–15 %, remote work 15–95 %, download 95–100 %.
        let host = baseURL.host ?? "worker"
        let task = request.task.rawValue
        progress?(PipelineProgress(stage: "Uploading to \(host)", fraction: 0, message: "preparing \(task) job"))
        let did = try await upload { f in
            progress?(PipelineProgress(stage: "Uploading to \(host)", fraction: 0.15 * f, message: String(format: "%.0f %%", f * 100)))
        }
        let jid = try await process(datasetId: did, request: request)
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
        try await downloadProcessOutput(jid, into: outputDirectory)
        progress?(PipelineProgress(stage: "Downloading from \(host)", fraction: 1, message: "done"))
    }
}

/// splatprep SfM on a remote worker (`/v1/process`); same request as the local solver.
struct RemoteSplatPrepPoseSolver: PoseSolver {
    var client: WorkerClient
    var local: SplatPrepPoseSolver

    func solve(_ frames: FrameSet, progress: PipelineProgressHandler?) async throws -> PoseSolution {
        let req = local.request(for: frames)
        // Upload only what the worker needs: images/, masks/, priors.
        let fm = FileManager.default
        let staging = frames.root.appendingPathComponent(".remote-upload/dataset")
        try? fm.removeItem(at: staging.deletingLastPathComponent())
        try fm.ensureDirectory(staging)
        defer { try? fm.removeItem(at: staging.deletingLastPathComponent()) }
        for sub in ["images", "masks"] + [req.priorPoses].compactMap({ $0 }) {
            let src = frames.root.appendingPathComponent(sub)
            try fm.linkOrCopyTree(src, to: staging.appendingPathComponent(sub))
        }
        let outDir = frames.root.appendingPathComponent(req.outputDir)
        try? fm.removeItem(at: outDir)
        let unpack = frames.root.appendingPathComponent(".remote-output")
        try? fm.removeItem(at: unpack)
        try await client.runProcessing(directory: staging, request: req, outputDirectory: unpack, progress: progress)
        // The zip may contain outputDir itself or its contents.
        let inner = [unpack.appendingPathComponent(req.outputDir), unpack].first { fm.fileExists(atPath: $0.appendingPathComponent("dataset.json").path) }
        guard let inner else { throw PipelineError.processFailed("remote output has no dataset.json") }
        try fm.moveItem(at: inner, to: outDir)
        try? fm.removeItem(at: unpack)
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
