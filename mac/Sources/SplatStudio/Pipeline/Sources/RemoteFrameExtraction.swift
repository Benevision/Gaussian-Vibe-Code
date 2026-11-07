import Foundation

/// Video → frames on a Linux worker (`task: frames`): the video is streamed up as-is, the worker decodes it,
/// picks the sharpest frame per window and (for 360° footage) cuts the views — the same algorithm and
/// file layout as `VideoFrameExtractor` / `Equirect360` here — and sends back `images/`, `masks/`, `frames.json`.
enum RemoteFrameExtraction {
    static func extract(video url: URL, options vo: VideoExtractionOptions, equirect: Equirect360Options?, into dir: URL,
                        worker: WorkerClient, progress: PipelineProgressHandler?) async throws -> FrameSet {
        let fm = FileManager.default
        try fm.ensureDirectory(dir)
        var req = ProcessingRequest(task: .frames)
        req.features = nil; req.matching = nil; req.mapper = nil; req.cameraModel = nil
        req.outputDir = "frames"
        req.video = "video." + (url.pathExtension.isEmpty ? "mp4" : url.pathExtension.lowercased())
        req.videoOptions = vo
        req.equirect = equirect
        let unpack = dir.appendingPathComponent(".remote-output")
        try? fm.removeItem(at: unpack)
        defer { try? fm.removeItem(at: unpack) }
        let name = req.video!
        try await worker.runProcessing(upload: { cb in try await worker.uploadFile(url, name: name, progress: cb) },
                                       request: req, outputDirectory: unpack, progress: progress)
        let inner = [unpack.appendingPathComponent(req.outputDir), unpack]
            .first { fm.fileExists(atPath: $0.appendingPathComponent(FrameSet.fileName).path) }
        guard let inner else { throw PipelineError.processFailed("the worker returned no \(FrameSet.fileName)") }
        for sub in ["images", "masks"] {
            let src = inner.appendingPathComponent(sub), dst = dir.appendingPathComponent(sub)
            guard fm.fileExists(atPath: src.path) else { continue }
            if fm.fileExists(atPath: dst.path) {
                for f in try fm.contentsOfDirectory(atPath: src.path) {
                    let d = dst.appendingPathComponent(f)
                    try? fm.removeItem(at: d)
                    try fm.moveItem(at: src.appendingPathComponent(f), to: d)
                }
            } else {
                try fm.moveItem(at: src, to: dst)
            }
        }
        var set = try JSONDecoder().decode(FrameSet.self, from: Data(contentsOf: inner.appendingPathComponent(FrameSet.fileName)))
        set.root = dir
        for i in set.frames.indices { set.frames[i].source = url.path }
        guard !set.frames.isEmpty else { throw PipelineError.noInput("no frames extracted from \(url.lastPathComponent)") }
        try set.save()
        return set
    }
}
