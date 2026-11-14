import Foundation

/// Old home video (VHS, Video8, Hi8 … digitised to a file): splatprep's `vhs` restoration finds the clips, picks frames
/// for sharpness and camera movement, deinterlaces, repairs tracking dropouts, crops borders and the head-switching
/// band, resamples to square pixels, denoises over neighbouring frames and evens out each clip's colour
/// (`tools/splatprep/vhs.py`, `docs/dev/VHS.md`). Every frame records its clip and camera group, so the camera
/// solve gives each clip (or each zoom setting within it) its own lens.
struct HomeVideoOptions: Codable, Hashable, Sendable {
    /// Clips to use (ids from the analysis); nil = the recommended clip (the best usable one).
    var clips: [Int]? = nil
    /// Frames to pick over all chosen clips.
    var frameCount: Int = 400
    /// Output height: tape holds no more than PAL's 576 lines.
    var maxHeight: Int = 576
    var denoise = true
    var colour = true
    var repairDropouts = true

    init() {}
    init(from decoder: Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        clips = c(.clips, clips); frameCount = c(.frameCount, frameCount); maxHeight = c(.maxHeight, maxHeight)
        denoise = c(.denoise, denoise); colour = c(.colour, colour); repairDropouts = c(.repairDropouts, repairDropouts)
    }

    var request: ProcessingRequest.VHS {
        .init(clips: clips, maxHeight: maxHeight, denoise: denoise, colour: colour, repairDropouts: repairDropouts, totalFrames: frameCount)
    }
}

/// One clip of `vhs.json`.
struct HomeVideoClip: Codable, Hashable, Identifiable, Sendable {
    var id: Int
    var start: Double
    var end: Double
    var duration: Double
    /// good | rotationOnly | static | blurry | moving | short | noSignal
    var verdict: String
    var reason: String
    var thumbnail: String? = nil
    var parallax: Double? = nil
    var rigidity: Double? = nil
    var cameraTravel: Double? = nil
    var relativeSharpness: Double? = nil
    var zooms: Bool? = nil
    var overlay: Bool? = nil
    var score: Double? = nil
    var selected: Int? = nil
    var cameras: Int? = nil

    var usable: Bool { verdict == "good" }
    /// Worth offering at all (the rest can't make a splat).
    var offerable: Bool { verdict != "noSignal" && verdict != "short" }
    var verdictTitle: String {
        switch verdict {
        case "good": return "Good: the camera moves"
        case "rotationOnly": return "Only turns: little depth"
        case "static": return "Camera hardly moves"
        case "blurry": return "Mostly blurred"
        case "moving": return "Mostly moving things"
        case "short": return "Too short"
        case "noSignal": return "No picture"
        default: return verdict
        }
    }
}

/// `vhs.json`: what splatprep found in the file.
struct HomeVideoAnalysis: Codable, Hashable, Sendable {
    var width: Int
    var height: Int
    var pixelAspect: Double
    var uniqueFps: Double
    var repeatFactor: Double
    var duration: Double
    var combedFrames: Int
    var noSignalFrames: Int
    var dropoutFrames: Int
    var outputHeight: Int
    var clips: [HomeVideoClip]
    var used: [Int]? = nil
    /// The clip used unless others are ticked (the best "good" one).
    var recommended: Int? = nil

    static let fileName = "vhs.json"
    static func load(_ dir: URL) -> HomeVideoAnalysis? {
        guard let d = try? Data(contentsOf: dir.appendingPathComponent(fileName)) else { return nil }
        return try? JSONDecoder().decode(HomeVideoAnalysis.self, from: d)
    }

    /// Plain-words summary of the tape's problems, for the sheet and the log.
    var summary: String {
        var parts = ["\(clips.filter(\.offerable).count) clips", String(format: "%.0f unique frames/s", uniqueFps)]
        if repeatFactor > 1.2 { parts.append(String(format: "each frame stored %.0f×", repeatFactor)) }
        if abs(pixelAspect - 1) > 0.01 { parts.append("non-square pixels") }
        if combedFrames > 0 { parts.append("\(combedFrames) interlaced or damaged frames") }
        if dropoutFrames > 0 { parts.append("\(dropoutFrames) frames with tracking lines") }
        if noSignalFrames > 0 { parts.append("blank stretches") }
        return parts.joined(separator: ", ")
    }
}

enum HomeVideoIngest {
    /// Analyses are kept per video (path, size, date) so the sheet's clip list and the project's frame step decode the
    /// file once.
    static func cacheDirectory(for video: URL) -> URL {
        let a = (try? FileManager.default.attributesOfItem(atPath: video.path)) ?? [:]
        let key = "\(video.path)|\((a[.size] as? NSNumber)?.int64Value ?? 0)|\((a[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0)"
        return FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("SplatStudio/HomeVideo/\(sha256Hex(Data(key.utf8)).prefix(16))", isDirectory: true)
    }

    /// Finds the clips (one pass over the file; a minute or two for a long tape). Cached.
    static func analyse(video: URL, runner: SplatPrepRunner = SplatPrepRunner(),
                        progress: PipelineProgressHandler? = nil) async throws -> (HomeVideoAnalysis, URL) {
        let dir = cacheDirectory(for: video)
        if let a = HomeVideoAnalysis.load(dir), FileManager.default.fileExists(atPath: dir.appendingPathComponent("vhs-analysis.npz").path) {
            return (a, dir)
        }
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        var req = ProcessingRequest(task: .frames)
        req.features = nil; req.matching = nil; req.mapper = nil; req.cameraModel = nil
        req.video = video.path; req.outputDir = "."
        req.vhs = .init(analyseOnly: true)
        try await runner.run(req, workDir: dir, progress: progress)
        guard let a = HomeVideoAnalysis.load(dir) else { throw PipelineError.processFailed("the video analysis wrote no \(HomeVideoAnalysis.fileName)") }
        return (a, dir)
    }

    /// Frames for a project: restoration on this Mac (splatprep) or on a worker (`task: frames` with `vhs`).
    static func ingest(video: URL, options: HomeVideoOptions, into dir: URL, worker: WorkerClient?, onWorker: Bool,
                       progress: PipelineProgressHandler?) async throws -> FrameSet {
        let fm = FileManager.default
        try fm.ensureDirectory(dir)
        if onWorker, let worker, let info = try? await worker.info(timeout: 10), info.canRestoreHomeVideo {
            do {
                return try await RemoteFrameExtraction.extract(video: video, options: VideoExtractionOptions(), equirect: nil, vhs: options.request,
                                                               into: dir, worker: worker, progress: progress)
            } catch let e as CancellationError {
                throw e
            } catch {
                if Task.isCancelled { throw CancellationError() }
                progress?(PipelineProgress(stage: "Restoring home video", fraction: -1,
                                           message: "the worker couldn't restore the video (\(error.localizedDescription)) — doing it on this Mac"))
            }
        }
        // reuse the analysis the New Project sheet made
        let cache = cacheDirectory(for: video).appendingPathComponent("vhs-analysis.npz")
        let here = dir.appendingPathComponent("vhs-analysis.npz")
        if fm.fileExists(atPath: cache.path), !fm.fileExists(atPath: here.path) { try? fm.copyItem(at: cache, to: here) }
        for sub in ["images", "masks"] { try? fm.removeItem(at: dir.appendingPathComponent(sub)) }
        var req = ProcessingRequest(task: .frames)
        req.features = nil; req.matching = nil; req.mapper = nil; req.cameraModel = nil
        req.video = video.path; req.outputDir = "."
        req.vhs = options.request
        _ = try await SplatPrepRunner().run(req, workDir: dir, progress: { p in
            progress?(PipelineProgress(stage: "Restoring home video", fraction: p.fraction, message: p.message))
        })
        var set = try FrameSet.load(dir)
        for i in set.frames.indices { set.frames[i].source = video.path }
        guard !set.frames.isEmpty else { throw PipelineError.noInput("no frames from \(video.lastPathComponent)") }
        try set.save()
        return set
    }

    /// Training settings for restored tape footage, unless the user changed them: colour from one tape generation
    /// to the next carries no view-dependent detail (SH degree 1), people and pets walk through most shots
    /// (transient masking), and the frames are SD.
    static func adjustTraining(_ c: TrainingConfig) -> TrainingConfig {
        var c = c
        c.shDegree = min(c.shDegree, 1)
        c.transientMasking = true
        c.maxImageSize = min(c.maxImageSize, 1024)
        return c
    }

    /// vhs.json in the dataset folder, or frames that carry their clip (datasets restored on a worker before the
    /// worker's vhs.json came back with the frames).
    static func isHomeVideoDataset(_ root: URL) -> Bool {
        if FileManager.default.fileExists(atPath: root.appendingPathComponent(HomeVideoAnalysis.fileName).path) { return true }
        guard let fs = try? FrameSet.load(root) else { return false }
        return fs.frames.contains { $0.clip != nil }
    }

    /// Names that say "old home video" (for the CLI and drag-and-drop): vhs, hi8, video8, betamax, camcorder.
    static func looksLikeHomeVideo(_ url: URL) -> Bool {
        let n = url.lastPathComponent.lowercased()
        return ["vhs", "hi8", "video8", "betamax", "camcorder", "home video", "homevideo"].contains { n.contains($0) }
    }
}
