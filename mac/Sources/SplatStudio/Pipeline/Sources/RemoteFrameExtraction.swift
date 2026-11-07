import Foundation

/// Video → frames on a Linux worker (`task: frames`): the video is streamed up as-is, the worker decodes it,
/// picks the sharpest frame per window and (for 360° footage) cuts the views — the same algorithm and
/// file layout as `VideoFrameExtractor` / `Equirect360` here — and sends back `masks/`, `frames.json` (and `images/`,
/// in the background on workers ≥ 1.3.8, which keep them for the next steps: see `RemoteStore`).
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
        // Workers ≥ 1.3.8 keep the frames: only frames.json and masks come back now, the images follow in the
        // background while masks / camera poses / training already run on the worker's copy.
        let chain = (try? await worker.info(timeout: 10))?.canChain ?? false
        let final = try await worker.runProcessing(upload: { cb in try await worker.uploadFile(url, name: name, progress: cb) },
                                                   request: req, outputDirectory: unpack, skip: chain ? ["images"] : [],
                                                   progress: progress)
        let kept = chain ? final.outputDatasetId : nil
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
        if let kept {
            let store = RemoteStore(worker: worker.baseURL.absoluteString, datasetId: kept, dirs: ["images"])
            try store.save(dir)
            await RemoteImageFetch.shared.start(root: dir, client: worker, store: store)
            progress?(PipelineProgress(stage: "Preparing images", fraction: -1,
                                       message: "frames stay on \(worker.baseURL.host ?? "the worker") for the next steps; downloading them here in the background"))
        } else if chain {
            throw PipelineError.processFailed("the worker didn't keep the frames (no outputDatasetId)")
        }
        for i in set.frames.indices { set.frames[i].source = url.path }
        guard !set.frames.isEmpty else { throw PipelineError.noInput("no frames extracted from \(url.lastPathComponent)") }
        try set.save()
        return set
    }
}
