import Foundation
import Vision
import ImageIO
import CoreVideo
import CoreGraphics
import Accelerate

/// Which model produces per-image keep-masks (255 = keep, 0 = ignored by the loss).
enum MaskingModel: Codable, Hashable, Sendable {
    /// No masks (360 nadir / undistortion masks are still kept).
    case none
    /// Apple Vision salient foreground instances (`VNGenerateForegroundInstanceMaskRequest`) — keeps the object(s).
    case appleVisionForeground
    /// Apple Vision person segmentation (`VNGeneratePersonSegmentationRequest`) — removes people (operator, passers-by).
    case appleVisionPerson
    /// Apple Vision people + animals (cats, dogs): person segmentation plus animal detection refined to the
    /// animal's outline with foreground instance masks. Removes the usual moving things, on-device.
    case appleVisionDynamic
    /// splatprep mask model (`sam2_auto`, `sky`, `person`, …).
    case splatprep(maskModelId: String)
}

/// Options for mask generation.
struct MaskingOptions: Codable, Hashable, Sendable {
    var model: MaskingModel = .none
    /// Invert the model's semantics (e.g. keep only people, or remove the foreground object).
    var invert: Bool = false
    /// Grow the removed region by this many pixels (hides halos around removed objects).
    var dilatePixels: Int = 4
    /// Object classes for the `dynamic` model (nil = defaults below).
    var dynamicClasses: [String]? = nil
    /// Also find the sky (a separate layer, stacked on the masks above): camera solving ignores it and training
    /// draws it with a learned skybox at infinity instead of splats (no more haze and shards in the sky).
    var sky: Bool = false
    static let defaultDynamicClasses = ["person", "dog", "cat", "bird", "car", "truck", "bus", "bicycle", "motorcycle"]

    init() {}
    init(from decoder: Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        model = c(.model, model); invert = c(.invert, invert); dilatePixels = c(.dilatePixels, dilatePixels); dynamicClasses = c(.dynamicClasses, dynamicClasses)
        sky = c(.sky, sky)
    }
}

/// Generates masks for a frame set and records them in `FrameSet.frames[i].mask` (combined with existing masks).
enum FrameMasker {
    /// - Parameter worker: when set, learned (splatprep) masks are computed on that Linux worker; Apple Vision
    ///   masks always run on this Mac. Falls back to this Mac if the worker can't run the model.
    static func apply(_ options: MaskingOptions, to frames: inout FrameSet, runner: SplatPrepRunner = .init(),
                      worker: WorkerClient? = nil, progress: PipelineProgressHandler? = nil) async throws {
        switch options.model {
        case .none: return
        case .appleVisionForeground, .appleVisionPerson, .appleVisionDynamic:
            try await applyVision(options, to: &frames, progress: progress)
        case .splatprep(let id):
            try await applySplatPrep(modelId: id, options: options, to: &frames, runner: runner, worker: worker, progress: progress)
        }
        try frames.save()
    }

    static func applyVision(_ options: MaskingOptions, to frames: inout FrameSet, progress: PipelineProgressHandler?) async throws {
        let stage = "Masking (Apple Vision)"
        try await waitForFrames(frames.root, progress: progress)
        let root = frames.root
        try FileManager.default.ensureDirectory(root.appendingPathComponent("masks"))
        let list = frames.frames
        let counter = ProgressCounter(total: list.count)
        let model = options.model
        let results: [String?] = try await parallelMap(list.count, maxConcurrent: min(4, imageDecodeConcurrency)) { i in
            let f = list[i]
            defer { let n = counter.increment(); progress?(PipelineProgress(stage: stage, fraction: Double(n) / Double(list.count), message: f.image)) }
            let url = root.appendingPathComponent(f.image)
            let made: [UInt8]?
            switch model {
            case .appleVisionPerson: made = try personKeepMask(url, width: f.width, height: f.height)
            case .appleVisionDynamic: made = try dynamicKeepMask(url, width: f.width, height: f.height)
            default: made = try foregroundKeepMask(url, width: f.width, height: f.height)
            }
            guard var keep = made else { return f.mask }
            if options.invert { keep = keep.map { 255 - $0 } }
            if options.dilatePixels > 0 { keep = erode(keep, width: f.width, height: f.height, radius: options.dilatePixels) }
            return try combineAndWrite(keep, frame: f, root: root)
        }
        for i in frames.frames.indices { frames.frames[i].mask = results[i] }
    }

    /// Sky layer: splatprep's sky model (SegFormer, or a heuristic without it) on this Mac or the worker, written as
    /// `splatprep-sky/masks/<image>.png` (255 = sky) and recorded in `frames[i].sky`. Frames with almost no sky
    /// get no sky mask.
    static func applySky(to frames: inout FrameSet, runner: SplatPrepRunner = .init(), worker: WorkerClient? = nil,
                         progress: PipelineProgressHandler?) async throws {
        var r = ProcessingRequest(task: .masks)
        r.imagesDir = "images"; r.outputDir = "splatprep-sky"; r.maskModel = "sky"; r.invertMask = true
        var ranRemotely = false
        if let worker {
            do { try await runRemote(r, frames: frames, worker: worker, progress: progress); ranRemotely = true } catch {
                #if !os(macOS)
                throw error   // no local splatprep on iOS
                #endif
                progress?(PipelineProgress(stage: "Finding the sky", fraction: -1,
                                           message: "worker couldn't run the sky model (\(error.localizedDescription)) — on this Mac"))
            }
        }
        if !ranRemotely {
            try await waitForFrames(frames.root, progress: progress)
            try await runner.run(r, workDir: frames.root, progress: progress)
        }
        let dir = frames.root.appendingPathComponent("splatprep-sky/masks")
        var withSky = 0
        for i in frames.frames.indices {
            let f = frames.frames[i]
            let base = ((f.image as NSString).lastPathComponent as NSString).deletingPathExtension
            let url = dir.appendingPathComponent(base + ".png")
            guard FileManager.default.fileExists(atPath: url.path),
                  let (px, _, _) = try? ImageUtil.readGray8(url, width: 64, height: max(8, 64 * f.height / max(f.width, 1))) else { continue }
            let sky = px.reduce(0) { $0 + ($1 >= 128 ? 1 : 0) }
            guard Double(sky) > 0.002 * Double(px.count) else { continue }     // no sky in this photo
            frames.frames[i].sky = "splatprep-sky/masks/\(base).png"
            withSky += 1
        }
        progress?(PipelineProgress(stage: "Finding the sky", fraction: 1, message: "sky in \(withSky) of \(frames.frames.count) photos"))
        try frames.save()
    }

    static func applySplatPrep(modelId: String, options: MaskingOptions, to frames: inout FrameSet, runner: SplatPrepRunner,
                               worker: WorkerClient? = nil, progress: PipelineProgressHandler?) async throws {
        var r = ProcessingRequest(task: .masks)
        r.imagesDir = "images"; r.outputDir = "splatprep-masks"; r.maskModel = modelId; r.invertMask = options.invert
        if modelId == "dynamic" {
            // Things that actually move. Static look-alikes (bags, umbrellas, strollers) were the main
            // false positives in testing, so they're left to the mask editor.
            r.maskClasses = options.dynamicClasses ?? MaskingOptions.defaultDynamicClasses
            r.maskThreshold = 0.35
        }
        var ranRemotely = false
        if let worker {
            do {
                try await runRemote(r, frames: frames, worker: worker, progress: progress)
                ranRemotely = true
            } catch {
                // e.g. the worker hasn't installed the processing models yet: do it here instead.
                #if !os(macOS)
                throw error   // no local splatprep on iOS
                #endif
                progress?(PipelineProgress(stage: "Masking", fraction: -1,
                                           message: "worker couldn't run \(modelId) (\(error.localizedDescription)) — masking on this Mac"))
            }
        }
        if !ranRemotely {
            try await waitForFrames(frames.root, progress: progress)
            try await runner.run(r, workDir: frames.root, progress: progress)
        }
        let dir = frames.root.appendingPathComponent("splatprep-masks/masks")
        try FileManager.default.ensureDirectory(frames.root.appendingPathComponent("masks"))
        // A mask that keeps (almost) nothing means the model lost its object (or failed): applying it would
        // silently remove that whole photo from training. Leave such frames unmasked, and refuse outright
        // when it's most of them — that's the wrong mask mode for this capture, not a few bad frames.
        func maskURL(_ f: IngestedFrame) -> URL {
            dir.appendingPathComponent((((f.image as NSString).lastPathComponent as NSString).deletingPathExtension) + ".png")
        }
        var emptyFrames = 0, seen = 0
        var empty = Set<Int>()
        for i in frames.frames.indices {     // pass 1: count (no masks kept in memory)
            let f = frames.frames[i], m = maskURL(f)
            guard FileManager.default.fileExists(atPath: m.path) else { continue }
            let (keep, _, _) = try ImageUtil.readGray8(m, width: f.width, height: f.height)
            seen += 1
            let kept = keep.reduce(0) { $0 + ($1 >= 128 ? 1 : 0) }
            if Double(kept) < 0.01 * Double(max(keep.count, 1)) { emptyFrames += 1; empty.insert(i) }
        }
        if seen > 0 && Double(emptyFrames) > 0.5 * Double(seen) {
            throw PipelineError.processFailed("Masking with \(modelId) kept almost nothing in \(emptyFrames) of \(seen) frames, so it was not applied. "
                + (modelId == "sam2_auto" ? "“Isolate one object” keeps only a single central object — for rooms and scenes choose “Remove moving things” or no masks."
                                          : "Try a different mask mode or no masks."))
        }
        if emptyFrames > 0 {
            progress?(PipelineProgress(stage: "Masking", fraction: -1, message: "\(emptyFrames) frames had empty masks and were left unmasked"))
        }
        for i in frames.frames.indices where !empty.contains(i) {     // pass 2: apply
            let f = frames.frames[i], m = maskURL(f)
            guard FileManager.default.fileExists(atPath: m.path) else { continue }
            var (keep, _, _) = try ImageUtil.readGray8(m, width: f.width, height: f.height)
            keep = keep.map { $0 >= 128 ? 255 : 0 }
            if options.dilatePixels > 0 { keep = erode(keep, width: f.width, height: f.height, radius: options.dilatePixels) }
            frames.frames[i].mask = try combineAndWrite(keep, frame: f, root: frames.root)
        }
    }

    /// Waits for frames still downloading from the worker that split the video (see `RemoteStore`).
    static func waitForFrames(_ root: URL, progress: PipelineProgressHandler?) async throws {
        guard let s = RemoteStore.load(root), !s.local else { return }
        progress?(PipelineProgress(stage: "Preparing images", fraction: -1, message: "waiting for the frames to finish downloading from the worker"))
        try await RemoteImageFetch.shared.ensure(root)
    }

    /// Runs the masks request on the worker and unpacks `splatprep-masks/` here. The images are uploaded —
    /// unless the worker already has them (it split the video: `RemoteStore`), then nothing is.
    static func runRemote(_ r: ProcessingRequest, frames: FrameSet, worker: WorkerClient, progress: PipelineProgressHandler?) async throws {
        let fm = FileManager.default
        if let store = RemoteStore.usable(at: frames.root, by: worker), store.dirs.contains("images") {
            let unpack = frames.root.appendingPathComponent(".remote-output")
            try? fm.removeItem(at: unpack)
            defer { try? fm.removeItem(at: unpack) }
            let w = worker.baseURL.absoluteString, root = frames.root
            let fp = RemoteJobLedger.fingerprint(r, images: frames.frames.map(\.image))
            var reuse: String? = nil
            if let id = RemoteJobLedger.lookup(root, request: r, fingerprint: fp, worker: w),
               let st = try? await worker.job(id), !(st.state == .failed || st.state == .cancelled) { reuse = id }
            try await worker.runProcessing(upload: { _ in store.datasetId }, request: r, outputDirectory: unpack,
                                           skip: ["images"], progress: progress, existingJob: reuse,
                                           onJobCreated: { RemoteJobLedger.record(root, request: r, fingerprint: fp, worker: w, jobId: $0) })
            let inner = [unpack.appendingPathComponent(r.outputDir), unpack].first { fm.fileExists(atPath: $0.appendingPathComponent("masks").path) }
            guard let inner else { throw PipelineError.processFailed("remote masking returned no masks/") }
            let outDir = frames.root.appendingPathComponent(r.outputDir)
            try? fm.removeItem(at: outDir)
            try fm.moveItem(at: inner, to: outDir)
            return
        }
        try await waitForFrames(frames.root, progress: progress)
        let staging = frames.root.appendingPathComponent(".remote-upload/dataset")
        try? fm.removeItem(at: staging.deletingLastPathComponent())
        try fm.ensureDirectory(staging)
        defer { try? fm.removeItem(at: staging.deletingLastPathComponent()) }
        try fm.linkOrCopyTree(frames.root.appendingPathComponent("images"), to: staging.appendingPathComponent("images"))
        let unpack = frames.root.appendingPathComponent(".remote-output")
        try? fm.removeItem(at: unpack)
        defer { try? fm.removeItem(at: unpack) }
        try await worker.runProcessing(directory: staging, request: r, outputDirectory: unpack, progress: progress)
        // The zip holds either outputDir itself or its contents.
        let inner = [unpack.appendingPathComponent(r.outputDir), unpack].first { fm.fileExists(atPath: $0.appendingPathComponent("masks").path) }
        guard let inner else { throw PipelineError.processFailed("remote masking returned no masks/") }
        let outDir = frames.root.appendingPathComponent(r.outputDir)
        try? fm.removeItem(at: outDir)
        try fm.moveItem(at: inner, to: outDir)
    }

    /// ANDs with an existing mask of the frame and writes `masks/<stem>.png`.
    static func combineAndWrite(_ keep: [UInt8], frame f: IngestedFrame, root: URL) throws -> String {
        var k = keep
        if let old = f.mask, let (o, _, _) = try? ImageUtil.readGray8(root.appendingPathComponent(old), width: f.width, height: f.height) {
            for j in k.indices { k[j] = min(k[j], o[j]) }
        }
        let rel = "masks/" + ((f.image as NSString).lastPathComponent as NSString).deletingPathExtension + ".png"
        try ImageUtil.writeGray8(k, width: f.width, height: f.height, to: root.appendingPathComponent(rel))
        return rel
    }

    /// Keep = salient foreground instances.
    static func foregroundKeepMask(_ url: URL, width: Int, height: Int) throws -> [UInt8]? {
        let handler = VNImageRequestHandler(url: url, options: [:])
        let req = VNGenerateForegroundInstanceMaskRequest()
        try handler.perform([req])
        guard let obs = req.results?.first else { return [UInt8](repeating: 0, count: width * height) }
        let pb = try obs.generateScaledMaskForImage(forInstances: obs.allInstances, from: handler)
        return resample(pb, width: width, height: height)
    }

    /// Keep = everything except people.
    static func personKeepMask(_ url: URL, width: Int, height: Int) throws -> [UInt8]? {
        let handler = VNImageRequestHandler(url: url, options: [:])
        let req = VNGeneratePersonSegmentationRequest()
        req.qualityLevel = .accurate
        req.outputPixelFormat = kCVPixelFormatType_OneComponent8
        try handler.perform([req])
        guard let obs = req.results?.first else { return [UInt8](repeating: 255, count: width * height) }
        guard let person = resample(obs.pixelBuffer, width: width, height: height) else { return nil }
        return person.map { 255 - $0 }
    }

    /// Keep = everything except people and animals.
    ///
    /// Frames from phones/headsets are often stored in sensor orientation (sideways), and Vision's
    /// detectors expect upright images, so the frame is tried upright and rotated ±90°; the orientation
    /// with the most confident person/animal detections is segmented and the mask rotated back.
    /// People: person segmentation (only when a person is actually detected). Animals:
    /// `VNRecognizeAnimalsRequest` boxes refined to foreground-instance outlines (box fallback).
    static func dynamicKeepMask(_ url: URL, width: Int, height: Int) throws -> [UInt8]? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let original = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return nil }
        // 1. choose orientation
        var best = (turns: 0, score: Float(0), humans: false)
        for turns in [0, 1, 3] {
            guard let img = turns == 0 ? original : rotate(original, quarterTurnsCW: turns) else { continue }
            let h = VNImageRequestHandler(cgImage: img, options: [:])
            let humans = VNDetectHumanRectanglesRequest(), animals = VNRecognizeAnimalsRequest()
            try? h.perform([humans, animals])
            let hs = (humans.results ?? []).map(\.confidence), as_ = (animals.results ?? []).map(\.confidence)
            let score = (hs + as_).reduce(0, +) + (turns == 0 ? 0.05 : 0)   // slight preference for as-stored
            if score > best.score { best = (turns, score, !hs.filter { $0 > 0.5 }.isEmpty) }
        }
        guard best.score > 0.1 else { return [UInt8](repeating: 255, count: width * height) }
        guard let img = best.turns == 0 ? original : rotate(original, quarterTurnsCW: best.turns) else { return nil }
        let rw = img.width, rh = img.height
        // 2. segment in that orientation
        var keep = [UInt8](repeating: 255, count: rw * rh)
        let handler = VNImageRequestHandler(cgImage: img, options: [:])
        let animals = VNRecognizeAnimalsRequest()
        let fg = VNGenerateForegroundInstanceMaskRequest()
        let person = VNGeneratePersonSegmentationRequest()
        person.qualityLevel = .accurate
        person.outputPixelFormat = kCVPixelFormatType_OneComponent8
        try handler.perform(best.humans ? [animals, fg, person] : [animals, fg])
        if best.humans, let pm = person.results?.first.flatMap({ resample($0.pixelBuffer, width: rw, height: rh) }) {
            for j in pm.indices where pm[j] != 0 { keep[j] = 0 }
        }
        let boxes = (animals.results ?? []).filter { $0.confidence > 0.3 }.map { $0.boundingBox }
        if !boxes.isEmpty {
            func pixelRect(_ b: CGRect, grow: CGFloat) -> (Int, Int, Int, Int) {   // Vision: normalised, bottom-left origin
                let gx = b.width * grow, gy = b.height * grow
                return (max(0, Int((b.minX - gx) * CGFloat(rw))), max(0, Int((1 - b.maxY - gy) * CGFloat(rh))),
                        min(rw, Int((b.maxX + gx) * CGFloat(rw))), min(rh, Int((1 - b.minY + gy) * CGFloat(rh))))
            }
            let rects = boxes.map { pixelRect($0, grow: 0.1) }
            var matched = [Bool](repeating: false, count: boxes.count)
            if let obs = fg.results?.first {
                for inst in obs.allInstances {
                    guard let pb = try? obs.generateScaledMaskForImage(forInstances: IndexSet(integer: inst), from: handler),
                          let m = resample(pb, width: rw, height: rh) else { continue }
                    var total = 0
                    var inside = [Int](repeating: 0, count: boxes.count)
                    for y in stride(from: 0, to: rh, by: 2) {
                        for x in stride(from: 0, to: rw, by: 2) where m[y * rw + x] != 0 {
                            total += 1
                            for (bi, r) in rects.enumerated() where x >= r.0 && x < r.2 && y >= r.1 && y < r.3 { inside[bi] += 1 }
                        }
                    }
                    guard total > 0, let b = inside.indices.max(by: { inside[$0] < inside[$1] }), Double(inside[b]) / Double(total) > 0.6 else { continue }
                    matched[b] = true
                    for j in m.indices where m[j] != 0 { keep[j] = 0 }
                }
            }
            for (bi, b) in boxes.enumerated() where !matched[bi] {
                let r = pixelRect(b, grow: 0.05)
                if r.1 < r.3 && r.0 < r.2 { for y in r.1..<r.3 { for x in r.0..<r.2 { keep[y * rw + x] = 0 } } }
            }
        }
        // 3. back to the stored orientation and requested size
        var back = keep, bw = rw, bh = rh
        if best.turns != 0 { (back, bw, bh) = rotateMask(keep, width: rw, height: rh, quarterTurnsCW: 4 - best.turns) }
        if bw == width && bh == height { return back }
        var out = [UInt8](repeating: 255, count: width * height)
        for y in 0..<height { for x in 0..<width { out[y * width + x] = back[min(bh - 1, y * bh / height) * bw + min(bw - 1, x * bw / width)] } }
        return out
    }

    /// Rotates an image by 90° steps clockwise.
    static func rotate(_ img: CGImage, quarterTurnsCW turns: Int) -> CGImage? {
        let t = ((turns % 4) + 4) % 4
        let w = img.width, h = img.height
        let (ow, oh) = t % 2 == 0 ? (w, h) : (h, w)
        guard let ctx = CGContext(data: nil, width: ow, height: oh, bitsPerComponent: 8, bytesPerRow: ow * 4,
                                  space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return nil }
        // CoreGraphics has a bottom-left origin; rotating by −90° there is a clockwise turn on screen.
        ctx.translateBy(x: CGFloat(ow) / 2, y: CGFloat(oh) / 2)
        ctx.rotate(by: -CGFloat(t) * .pi / 2)
        ctx.draw(img, in: CGRect(x: -CGFloat(w) / 2, y: -CGFloat(h) / 2, width: CGFloat(w), height: CGFloat(h)))
        return ctx.makeImage()
    }

    /// Rotates a top-left-origin mask by 90° steps clockwise.
    static func rotateMask(_ m: [UInt8], width w: Int, height h: Int, quarterTurnsCW turns: Int) -> ([UInt8], Int, Int) {
        let t = ((turns % 4) + 4) % 4
        if t == 0 { return (m, w, h) }
        let (ow, oh) = t % 2 == 0 ? (w, h) : (h, w)
        var out = [UInt8](repeating: 255, count: ow * oh)
        for y in 0..<h {
            for x in 0..<w {
                let (nx, ny): (Int, Int)
                switch t {
                case 1: (nx, ny) = (h - 1 - y, x)
                case 2: (nx, ny) = (w - 1 - x, h - 1 - y)
                default: (nx, ny) = (y, w - 1 - x)
                }
                out[ny * ow + nx] = m[y * w + x]
            }
        }
        return (out, ow, oh)
    }

    /// Converts a one-component 8-bit or float mask buffer to a binary 8-bit mask of the given size.
    static func resample(_ pb: CVPixelBuffer, width: Int, height: Int) -> [UInt8]? {
        CVPixelBufferLockBaseAddress(pb, .readOnly); defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb), bpr = CVPixelBufferGetBytesPerRow(pb)
        let isFloat = CVPixelBufferGetPixelFormatType(pb) == kCVPixelFormatType_OneComponent32Float
        var out = [UInt8](repeating: 0, count: width * height)
        for y in 0..<height {
            let sy = min(h - 1, y * h / height)
            let row = base + sy * bpr
            for x in 0..<width {
                let sx = min(w - 1, x * w / width)
                let v: Float = isFloat ? row.load(fromByteOffset: sx * 4, as: Float.self) : Float(row.load(fromByteOffset: sx, as: UInt8.self)) / 255
                out[y * width + x] = v >= 0.5 ? 255 : 0
            }
        }
        return out
    }

    /// Morphological erosion of the keep region (= dilation of the removed region).
    static func erode(_ m: [UInt8], width: Int, height: Int, radius: Int) -> [UInt8] {
        var src = m, dst = [UInt8](repeating: 0, count: m.count)
        let k = 2 * radius + 1
        src.withUnsafeMutableBytes { sp in dst.withUnsafeMutableBytes { dp in
            var s = vImage_Buffer(data: sp.baseAddress, height: vImagePixelCount(height), width: vImagePixelCount(width), rowBytes: width)
            var d = vImage_Buffer(data: dp.baseAddress, height: vImagePixelCount(height), width: vImagePixelCount(width), rowBytes: width)
            _ = vImageMin_Planar8(&s, &d, nil, 0, 0, vImagePixelCount(k), vImagePixelCount(k), vImage_Flags(kvImageEdgeExtend))
        } }
        return dst
    }
}
