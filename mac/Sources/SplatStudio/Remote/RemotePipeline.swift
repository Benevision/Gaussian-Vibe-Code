import Foundation

/// A whole-pipeline job of a project (`<project>/.pipeline-job.json`), written as soon as the worker accepts it and
/// removed once its dataset is here. While it exists, Resume Processing (and launching the app) re-attaches to the
/// job instead of starting the work again: the Mac may have slept, lost the network or quit meanwhile.
struct PipelineJobRecord: Codable, Sendable {
    var workerId: String
    var baseURL: String
    var workerName: String
    var jobId: String
    var name: String
    var startedAt: Date
    /// `PrepCheckpoints.fingerprints` of the run (keyed by `PrepStage.rawValue`): a record of other settings is stale.
    var fingerprints: [String: String]
    /// The images are already in the project (ingested here); a video split on the worker leaves them there.
    var imagesLocal: Bool
    var trains: Bool
    var config: TrainingConfig?

    static let fileName = ".pipeline-job.json"
    static func url(_ project: URL) -> URL { project.appendingPathComponent(fileName) }
    static func load(_ project: URL) -> PipelineJobRecord? {
        guard let d = try? Data(contentsOf: url(project)) else { return nil }
        return try? JSONDecoder.iso.decode(PipelineJobRecord.self, from: d)
    }
    func save(_ project: URL) { try? JSONEncoder.iso.encode(self).write(to: Self.url(project), options: .atomic) }
    static func remove(_ project: URL) { try? FileManager.default.removeItem(at: url(project)) }
}

/// Processing and training of a project as one job on a SplatWork for Linux worker: the source and every setting go
/// up once, the worker runs frames → masks → sky → camera solve → dataset → training by itself, and this Mac only
/// fetches the prepared dataset (and the result) — whenever it is back online.
enum RemotePipeline {
    /// Fingerprints by stage name, as `PipelineJobRecord` and the worker keep them.
    static func fingerprintStrings(_ source: SourceDescription, _ o: PrepOptions) -> [String: String] {
        Dictionary(uniqueKeysWithValues: PrepCheckpoints.fingerprints(source, o).map { ($0.key.rawValue, $0.value) })
    }

    /// Why this run can't be one pipeline job (nil = it can). Processing and training must both go to this worker,
    /// and every stage still to run must be one the worker can do (no Apple Vision masks, no Object Capture, no
    /// stereo or posed sources).
    static func ineligibility(source: SourceDescription, options: PrepOptions, dir: URL, info: WorkerInfo,
                              trainOnSameWorker: Bool) -> String? {
        guard info.canRunPipeline else { return "the worker can't run whole pipelines (update it to SplatWork for Linux 1.6)" }
        guard trainOnSameWorker else { return "training runs elsewhere" }
        switch source {
        case .photos, .photoFiles, .video, .video360, .photos360: break
        default: return "this kind of source is prepared step by step"
        }
        let o = options.resolved
        let fps = PrepCheckpoints.fingerprints(source, o)
        let cp = PrepCheckpoints(root: dir)
        if cp.entry(.poses)?.fingerprint == fps[.poses] { return "the cameras are already solved" }
        if case .objectCapture = o.poseSolver { return "Object Capture runs on this Mac" }
        let masksDone = cp.entry(.masks)?.fingerprint == fps[.masks]
        if !masksDone {
            switch o.masking.model {
            case .appleVisionForeground, .appleVisionPerson, .appleVisionDynamic: return "Apple Vision masks run on this Mac"
            default: break
            }
        }
        return nil
    }

    /// The solver request (without what only the frames tell: the worker adds masks, sky, rig and fisheye).
    static func solveRequest(_ o: PrepOptions, source: SourceDescription) -> (ProcessingRequest, sequential: Bool) {
        let sequential: Bool = { switch source { case .video, .video360: return true; default: return false } }()
        let multiView: Bool = { switch source { case .video360, .photos360: return true; default: return false } }()
        var r = ProcessingRequest(task: .sfm)
        switch o.poseSolver {
        case .splatprep(let f, let m, let mp): r.features = f; r.matching = m; r.mapper = mp
        default:
            // Automatic, with every stage on the worker: its learned SfM (Object Capture needs this Mac).
            r.features = "superpoint_lightglue"; r.matching = multiView ? "auto" : (sequential ? "sequential" : "auto"); r.mapper = "auto"
        }
        r.imagesDir = "images"; r.outputDir = "sfm"
        r.cameraModel = o.sfmCameraModel
        r.resizeMax = o.maxImageSize
        return (r, sequential)
    }

    static func masksRequest(_ o: PrepOptions) -> ProcessingRequest? {
        guard case .splatprep(let id) = o.masking.model else { return nil }
        var r = ProcessingRequest(task: .masks)
        r.imagesDir = "images"; r.outputDir = "splatprep-masks"; r.maskModel = id; r.invertMask = o.masking.invert
        if id == "dynamic" {
            r.maskClasses = o.masking.dynamicClasses ?? MaskingOptions.defaultDynamicClasses
            r.maskThreshold = 0.35
        }
        return r
    }

    static func skyRequest() -> ProcessingRequest {
        var r = ProcessingRequest(task: .masks)
        r.imagesDir = "images"; r.outputDir = "splatprep-sky"; r.maskModel = "sky"; r.invertMask = true
        return r
    }

    /// Frames ingested on this Mac (photos, 360° photos, a video when the worker shouldn't split it).
    private static func ingest(_ source: SourceDescription, _ o: PrepOptions, into dir: URL,
                               progress: @escaping PipelineProgressHandler) async throws -> FrameSet {
        switch source {
        case .photos(let folder): return try await PhotoIngest.ingest(folder: folder, options: o.photos, into: dir, progress: progress)
        case .photoFiles(let files): return try await PhotoIngest.ingest(files: files, options: o.photos, into: dir, progress: progress)
        case .video(let u): return try await VideoFrameExtractor.extract(video: u, options: o.video, into: dir, progress: progress)
        case .video360(let u): return try await Equirect360.extract(video: u, video: o.video, options: o.equirect, into: dir, progress: progress)
        case .photos360(let folder):
            return try await Equirect360.extract(photos: PhotoIngest.listImages(in: folder, recursive: false), options: o.equirect,
                                                 encoding: o.photos.encoding, into: dir, progress: progress)
        default: throw PipelineError.noInput("this source can't be sent as one pipeline job")
        }
    }

    /// Uploads the source, starts the job and records it. Stages whose checkpoints match are not sent again.
    static func start(source: SourceDescription, options: PrepOptions, dir: URL, project: URL, worker: WorkerClient,
                      workerId: String, info: WorkerInfo, train: TrainingConfig?,
                      progress: @escaping PipelineProgressHandler) async throws -> PipelineJobRecord {
        let o = options.resolved
        let fm = FileManager.default
        try fm.ensureDirectory(dir)
        let cp = PrepCheckpoints(root: dir)
        let fps = PrepCheckpoints.fingerprints(source, o)
        let host = worker.baseURL.host ?? info.name
        let uploadP = PipelineProgress.scaled(progress, 0.05...0.15)
        var frames: FrameSet? = nil
        var masksDone = false
        if let m = cp.frames(.masks, fingerprint: fps[.masks]!) {
            frames = m; masksDone = true
            progress(PipelineProgress(stage: "Masks", fraction: 0.02, message: "reusing the frames and masks from the last run"))
        } else if let f = cp.frames(.frames, fingerprint: fps[.frames]!) {
            frames = f
            progress(PipelineProgress(stage: "Preparing images", fraction: 0.02, message: "reusing the \(f.frames.count) frames from the last run"))
        }
        var videoURL: URL? = nil
        if frames == nil {
            var video: URL? = nil
            switch source { case .video(let u), .video360(let u): video = u; default: break }
            if let video, o.videoOnWorker, info.canExtractFrames {
                videoURL = video
            } else {
                let f = try await ingest(source, o, into: dir, progress: PipelineProgress.scaled(progress, 0...0.05))
                try cp.save(f, as: .frames, fingerprint: fps[.frames]!,
                            summary: "\(f.frames.count) frames" + (f.rejected.isEmpty ? "" : ", \(f.rejected.count) left out"))
                frames = f
            }
        }
        // Upload: the video as it is, or the frames (images, frames.json and the masks it names).
        let datasetId: String
        var spec = PipelineSpec(frames: nil, staticRegions: !masksDone && o.maskStaticRegions, fisheyeMasks: !masksDone,
                                masks: masksDone ? nil : masksRequest(o).map { .init(request: $0, dilatePixels: o.masking.dilatePixels) },
                                sky: masksDone || !o.masking.sky ? nil : .init(request: skyRequest()),
                                sfm: { let (r, seq) = solveRequest(o, source: source); return .init(request: r, sequential: seq, enforceRig: o.enforceRigConstraints) }(),
                                holdoutInterval: o.holdoutInterval, train: train.map { .init(config: $0) },
                                checkpoints: fingerprintStrings(source, o))
        if let u = videoURL {
            var req = ProcessingRequest(task: .frames)
            req.features = nil; req.matching = nil; req.mapper = nil; req.cameraModel = nil
            req.outputDir = "frames"
            req.video = "video." + (u.pathExtension.isEmpty ? "mp4" : u.pathExtension.lowercased())
            req.videoOptions = o.video
            if case .video360 = source { req.equirect = o.equirect }
            spec.frames = .init(request: req)
            progress(PipelineProgress(stage: "Uploading to \(host)", fraction: 0.05, message: "the video, once — every step runs there"))
            datasetId = try await worker.uploadFile(u, name: req.video!) { f in
                uploadP(PipelineProgress(stage: "Uploading to \(host)", fraction: f, message: String(format: "video %.0f %%", f * 100)))
            }
        } else {
            guard let frames else { throw PipelineError.noInput("no frames") }
            try await FrameMasker.waitForFrames(dir, progress: progress)
            let staging = dir.appendingPathComponent(".remote-upload/pipeline")
            try? fm.removeItem(at: staging.deletingLastPathComponent())
            try fm.ensureDirectory(staging)
            defer { try? fm.removeItem(at: staging.deletingLastPathComponent()) }
            try fm.linkOrCopyTree(dir.appendingPathComponent("images"), to: staging.appendingPathComponent("images"))
            for rel in Set(frames.frames.flatMap { [$0.mask, $0.sky].compactMap { $0 } }) {
                let src = dir.appendingPathComponent(rel), dst = staging.appendingPathComponent(rel)
                guard fm.fileExists(atPath: src.path) else { continue }
                try fm.ensureDirectory(dst.deletingLastPathComponent())
                try? fm.linkItem(at: src, to: dst)
            }
            var up = frames; up.root = staging
            try up.save()
            progress(PipelineProgress(stage: "Uploading to \(host)", fraction: 0.05, message: "\(frames.frames.count) frames, once — every step runs there"))
            datasetId = try await worker.uploadDataset(directory: staging) { f in
                uploadP(PipelineProgress(stage: "Uploading to \(host)", fraction: f, message: String(format: "frames %.0f %%", f * 100)))
            }
        }
        let name = project.lastPathComponent
        let jobId = try await worker.createPipeline(datasetId: datasetId, name: name, pipeline: spec)
        let rec = PipelineJobRecord(workerId: workerId, baseURL: worker.baseURL.absoluteString, workerName: info.name, jobId: jobId,
                                    name: name, startedAt: Date(), fingerprints: spec.checkpoints, imagesLocal: videoURL == nil,
                                    trains: train != nil, config: train)
        rec.save(project)
        progress(PipelineProgress(stage: "On \(host)", fraction: 0.15,
                                  message: "every step now runs on \(info.name) — this Mac can sleep or go offline; reopen the project to pick it up"))
        return rec
    }

    /// Where a pipeline job stands, for the overall bar (DatasetBuilder's ranges: frames ‹ 0.35, masks ‹ 0.45,
    /// cameras ‹ 0.97).
    static func overall(_ s: JobStatus) -> Double {
        guard let stages = s.stages, let cur = s.stage else { return 0.15 }
        let prepIds = stages.filter { $0.id != "train" }.map(\.id)
        if cur == "train" { return 0.98 }
        let ranges: [String: ClosedRange<Double>] = ["frames": 0.15...0.35, "static": 0.35...0.37, "masks": 0.37...0.42,
                                                     "sky": 0.42...0.45, "sfm": 0.45...0.96, "level": 0.96...0.97]
        guard prepIds.contains(cur), let r = ranges[cur] else { return 0.15 }
        return r.lowerBound + (r.upperBound - r.lowerBound) * min(1, max(0, s.stageProgress ?? 0))
    }

    struct Ready { var dataset: PreparedDataset; var trainJobId: String?; var status: JobStatus }

    /// Follows the job until its dataset is ready (or the whole job is done), riding out the worker being unreachable
    /// for up to `maxOffline`, then fetches the dataset into `dir` with the checkpoints a local run leaves.
    /// Cancelling the task cancels the job on the worker.
    static func follow(_ rec: PipelineJobRecord, client: WorkerClient, source: SourceDescription, options: PrepOptions,
                       dir: URL, project: URL, maxOffline: TimeInterval = 24 * 3600,
                       progress: @escaping PipelineProgressHandler) async throws -> Ready {
        let host = client.baseURL.host ?? rec.workerName
        var offlineSince: Date? = nil
        var backoff: TimeInterval = 2
        var final: JobStatus
        var lastStage = ""
        do {
            while true {
                do {
                    let s = try await client.job(rec.jobId)
                    offlineSince = nil; backoff = 2
                    let st = s.stages?.first { $0.id == s.stage }
                    let title = st?.title ?? s.stage ?? "Working"
                    if title != lastStage, !lastStage.isEmpty,
                       let prev = s.stages?.first(where: { ($0.title ?? $0.id) == lastStage }) {
                        progress(PipelineProgress(stage: "On \(host): \(lastStage)", fraction: -1,
                                                  message: prev.state == "skipped" ? (prev.message ?? "skipped") : "done"))
                    }
                    lastStage = title
                    let detail = s.state == .paused ? "paused on \(rec.workerName)" : (s.message ?? "").split(separator: ":", maxSplits: 1).dropFirst().first.map { $0.trimmingCharacters(in: .whitespaces) } ?? ""
                    progress(PipelineProgress(stage: "On \(host): \(title)", fraction: overall(s), message: detail))
                    if s.state == .failed || s.state == .cancelled {
                        PipelineJobRecord.remove(project)
                        throw PipelineError.processFailed("\(rec.workerName): \(s.message ?? s.state.rawValue)")
                    }
                    if s.state == .completed || (s.outputDatasetId != nil && s.stage == "train") { final = s; break }
                    try await Task.sleep(nanoseconds: 2_000_000_000)
                } catch let e as PipelineError {
                    throw e
                } catch {
                    if error is CancellationError || Task.isCancelled || (error as? URLError)?.code == .cancelled { throw error }
                    if let w = error as? WorkerError, w.status == 404 {
                        PipelineJobRecord.remove(project)
                        throw PipelineError.processFailed("\(rec.workerName) no longer has this project's job (its data was cleared?) — Resume starts it again")
                    }
                    guard WorkerClient.isTransient(error) else { throw error }
                    let since = offlineSince ?? Date(); offlineSince = since
                    let gone = Date().timeIntervalSince(since)
                    if gone > maxOffline { throw WorkerError(status: 503, message: "\(rec.workerName) unreachable for \(Int(gone / 60)) min") }
                    progress(PipelineProgress(stage: "On \(host)", fraction: -1,
                                              message: "can't reach \(rec.workerName) for \(gone < 60 ? "\(Int(gone)) s" : "\(Int(gone / 60)) min") — the job keeps running there; retrying"))
                    try await Task.sleep(nanoseconds: UInt64(backoff * 1e9)); backoff = min(backoff * 2, 30)
                }
            }
        } catch {
            guard error is CancellationError || Task.isCancelled || (error as? URLError)?.code == .cancelled else { throw error }
            await Task.detached { _ = try? await client.cancelJob(rec.jobId) }.value
            PipelineJobRecord.remove(project)
            throw CancellationError()
        }
        guard let datasetId = final.outputDatasetId else { throw PipelineError.processFailed("the worker kept no dataset") }
        let ds = try await adopt(rec, client: client, status: final, datasetId: datasetId, source: source, options: options, dir: dir, progress: progress)
        PipelineJobRecord.remove(project)
        return Ready(dataset: ds, trainJobId: rec.trains ? final.trainJobId : nil, status: final)
    }

    /// Downloads the prepared dataset (without the images: they are here, or follow in the background) and
    /// records the frames / masks / poses / done checkpoints from the worker's `.pipeline/` snapshots.
    static func adopt(_ rec: PipelineJobRecord, client: WorkerClient, status: JobStatus, datasetId: String,
                      source: SourceDescription, options: PrepOptions, dir: URL,
                      progress: @escaping PipelineProgressHandler) async throws -> PreparedDataset {
        let fm = FileManager.default
        let host = client.baseURL.host ?? rec.workerName
        let tmp = dir.appendingPathComponent(".remote-output")
        try? fm.removeItem(at: tmp)
        defer { try? fm.removeItem(at: tmp) }
        progress(PipelineProgress(stage: "Downloading from \(host)", fraction: 0.97, message: "the prepared dataset"))
        let start = Date()
        var backoff: TimeInterval = 2
        while true {
            do {
                try await client.downloadDatasetFiles(datasetId, skip: ["images"], into: tmp) { f in
                    progress(PipelineProgress(stage: "Downloading from \(host)", fraction: 0.97, message: String(format: "the prepared dataset — %.0f %%", f * 100)))
                }
                break
            } catch {
                guard WorkerClient.isTransient(error), !Task.isCancelled, Date().timeIntervalSince(start) < 3600 else { throw error }
                try await Task.sleep(nanoseconds: UInt64(backoff * 1e9)); backoff = min(backoff * 2, 30)
            }
        }
        // Into the project: every file the worker made replaces the one here (frames.json, masks, sky, sfm, dataset).
        let snaps = tmp.appendingPathComponent(".pipeline")
        try moveTree(tmp, into: dir, except: ".pipeline/")
        if !rec.imagesLocal {
            // A video split on the worker: its frames stay there (the next steps ran on them) and come down in the
            // background, as after a step-by-step split.
            let store = RemoteStore(worker: client.baseURL.absoluteString, datasetId: datasetId, dirs: ["images"])
            try store.save(dir)
            await RemoteImageFetch.shared.start(root: dir, client: client, store: store)
        }
        try writeCheckpoints(snapshots: snaps, status: status, source: source, options: options, dir: dir)
        var ds = try PreparedDataset.load(dir)
        ds.root = dir
        progress(PipelineProgress(stage: "Finalising dataset", fraction: 0.98, message: "\(ds.manifest.cameras.count) cameras, levelled on \(rec.workerName)"))
        return ds
    }

    /// The checkpoints a local run would have left (same fingerprints, same files), so Redo from… and changing a
    /// later stage's settings reuse what the worker did.
    static func writeCheckpoints(snapshots snaps: URL, status: JobStatus, source: SourceDescription, options: PrepOptions, dir: URL) throws {
        let fm = FileManager.default
        let o = options.resolved
        let cp = PrepCheckpoints(root: dir)
        let fps = PrepCheckpoints.fingerprints(source, o)
        func frameSet(_ name: String) -> FrameSet? {
            guard let d = try? Data(contentsOf: snaps.appendingPathComponent("\(name).frames.json")),
                  var fs = try? JSONDecoder().decode(FrameSet.self, from: d) else { return nil }
            fs.root = dir
            for i in fs.frames.indices { fs.frames[i].source = fs.frames[i].source ?? sourcePath(source) }
            return fs
        }
        if cp.entry(.frames)?.fingerprint != fps[.frames], let f = frameSet("frames") {
            try cp.save(f, as: .frames, fingerprint: fps[.frames]!,
                        summary: "\(f.frames.count) frames" + (f.rejected.isEmpty ? "" : ", \(f.rejected.count) left out"),
                        filesFrom: snaps.appendingPathComponent("frames.files"))
        }
        let skySkipped = status.stages?.contains { $0.id == "sky" && $0.state == "skipped" } ?? false
        if cp.entry(.masks)?.fingerprint != fps[.masks], let m = frameSet("masks") {
            try cp.save(m, as: .masks, fingerprint: fps[.masks]!, summary: DatasetBuilder.maskSummary(o, skySkipped: skySkipped),
                        filesFrom: snaps.appendingPathComponent("masks.files"))
            try? m.save()
        }
        // poses: the solved dataset before levelling — put it in place for a moment, save it, then the levelled one back
        let live = dir.appendingPathComponent(PreparedDataset.fileName)
        let posesManifest = snaps.appendingPathComponent("poses.dataset.json")
        if fm.fileExists(atPath: posesManifest.path), let levelled = try? Data(contentsOf: live) {
            let points = dir.appendingPathComponent("points3d.ply")
            let levelledPoints = try? Data(contentsOf: points)
            defer {
                try? levelled.write(to: live, options: .atomic)
                if let levelledPoints { try? levelledPoints.write(to: points, options: .atomic) }
            }
            try Data(contentsOf: posesManifest).write(to: live, options: .atomic)
            let pp = snaps.appendingPathComponent("poses.points.ply")
            if fm.fileExists(atPath: pp.path) { try Data(contentsOf: pp).write(to: points, options: .atomic) }
            let solved = try PreparedDataset.load(dir)
            let n = frameSet("masks")?.frames.count ?? frameSet("frames")?.frames.count ?? solved.manifest.cameras.count
            try cp.save(solved, as: .poses, fingerprint: fps[.poses]!, summary: "\(solved.manifest.cameras.count) of \(n) frames placed")
        }
        let ready = try PreparedDataset.load(dir)
        try cp.markDone(fingerprint: fps[.done]!, summary: "\(ready.manifest.cameras.count) cameras, levelled")
    }

    /// Moves every file below `src` to the same place below `dst` (replacing), except those under `except`.
    private static func moveTree(_ src: URL, into dst: URL, except: String) throws {
        let fm = FileManager.default
        guard let e = fm.enumerator(at: src, includingPropertiesForKeys: [.isRegularFileKey]) else { return }
        let base = src.standardizedFileURL.path
        for case let f as URL in e where (try? f.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true {
            let rel = String(f.standardizedFileURL.path.dropFirst(base.count + 1))
            guard !rel.hasPrefix(except) else { continue }
            let to = dst.appendingPathComponent(rel)
            try fm.ensureDirectory(to.deletingLastPathComponent())
            try? fm.removeItem(at: to)
            try fm.moveItem(at: f, to: to)
        }
    }

    private static func sourcePath(_ s: SourceDescription) -> String? {
        switch s {
        case .video(let u), .video360(let u): return u.path
        default: return nil
        }
    }
}
