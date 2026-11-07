import Foundation
import CoreGraphics

/// Options for importing a folder of photos.
struct PhotoIngestOptions: Codable, Hashable, Sendable {
    /// Longest side of the written images in pixels (nil = keep full resolution).
    var maxImageSize: Int? = 1600
    /// Drop images whose sharpness is far below the set's median (motion blur / missed focus).
    var rejectBlurry: Bool = true
    /// An image is blurry if its variance-of-Laplacian is below this fraction of the median.
    var blurRelativeThreshold: Double = 0.35
    /// Optional absolute variance-of-Laplacian threshold (in 0…255² units); applied in addition.
    var blurAbsoluteThreshold: Double? = nil
    /// Drop near-identical images (burst shots, accidental double taps).
    var rejectDuplicates: Bool = true
    /// Maximum dHash Hamming distance (0…64) for two images to count as duplicates.
    var duplicateHammingThreshold: Int = 2
    /// Also look into sub-folders.
    var recursive: Bool = false
    /// Output encoding of the written images.
    var encoding: ImageUtil.Encoding = .jpeg

    init() {}
    init(from decoder: Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        maxImageSize = c(.maxImageSize, maxImageSize); rejectBlurry = c(.rejectBlurry, rejectBlurry)
        blurRelativeThreshold = c(.blurRelativeThreshold, blurRelativeThreshold); blurAbsoluteThreshold = c(.blurAbsoluteThreshold, blurAbsoluteThreshold)
        rejectDuplicates = c(.rejectDuplicates, rejectDuplicates); duplicateHammingThreshold = c(.duplicateHammingThreshold, duplicateHammingThreshold)
        recursive = c(.recursive, recursive); encoding = c(.encoding, encoding)
    }
}

/// Imports a photo folder (JPEG / HEIC / PNG / TIFF / RAW) into a `FrameSet`:
/// EXIF orientation applied, HDR tone-mapped, optionally downscaled, blurry and duplicate shots rejected.
enum PhotoIngest {
    /// Lists image files (sorted by name) in a folder.
    static func listImages(in folder: URL, recursive: Bool) -> [URL] {
        let fm = FileManager.default
        var urls: [URL] = []
        if recursive {
            let e = fm.enumerator(at: folder, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
            while let u = e?.nextObject() as? URL { if ImageUtil.isImageFile(u) { urls.append(u) } }
        } else {
            urls = ((try? fm.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])) ?? [])
                .filter(ImageUtil.isImageFile)
        }
        // When RAW+JPEG pairs exist prefer the RAW (better dynamic range) and drop the JPEG twin.
        let rawStems = Set(urls.filter { ImageUtil.rawExtensions.contains($0.pathExtension.lowercased()) }.map { $0.deletingPathExtension().path })
        urls = urls.filter { ImageUtil.rawExtensions.contains($0.pathExtension.lowercased()) || !rawStems.contains($0.deletingPathExtension().path) }
        return urls.sorted { $0.path.localizedStandardCompare($1.path) == .orderedAscending }
    }

    /// Ingests the given files (or all images in `folder`) into `outputRoot/images/`.
    static func ingest(folder: URL? = nil, files: [URL]? = nil, options: PhotoIngestOptions = .init(), into outputRoot: URL,
                       progress: PipelineProgressHandler? = nil) async throws -> FrameSet {
        let inputs = files ?? folder.map { listImages(in: $0, recursive: options.recursive) } ?? []
        guard !inputs.isEmpty else { throw PipelineError.noInput("no images found") }
        let fm = FileManager.default
        let imagesDir = try fm.ensureDirectory(outputRoot.appendingPathComponent("images"))
        let ext = options.encoding == .jpeg ? "jpg" : "png"
        progress?(PipelineProgress(stage: "Importing photos", fraction: 0, message: "\(inputs.count) images"))

        struct Item: Sendable { var frame: IngestedFrame?; var hash: UInt64; var error: String? }
        let counter = ProgressCounter(total: inputs.count)
        let items: [Item] = try await parallelMap(inputs.count) { i in
            let src = inputs[i]
            defer { let n = counter.increment(); progress?(PipelineProgress(stage: "Importing photos", fraction: Double(n) / Double(inputs.count), message: src.lastPathComponent)) }
            do {
                let img = try ImageUtil.load(src, maxSize: options.maxImageSize)
                let name = String(format: "%06d.%@", i, ext)
                try ImageUtil.write(img, to: imagesDir.appendingPathComponent(name), encoding: options.encoding)
                var f = IngestedFrame(image: "images/\(name)", width: img.width, height: img.height)
                f.sharpness = ImageMetrics.sharpness(img)
                f.source = src.path
                f.focalPx = ImageUtil.exifFocalPixels(src, width: img.width, height: img.height)
                return Item(frame: f, hash: ImageMetrics.dHash(img), error: nil)
            } catch {
                return Item(frame: nil, hash: 0, error: error.localizedDescription)
            }
        }

        var set = FrameSet(root: outputRoot)
        for (i, it) in items.enumerated() where it.frame == nil {
            set.rejected.append(.init(item: inputs[i].lastPathComponent, reason: "unreadable: \(it.error ?? "?")"))
        }
        var candidates = items.enumerated().compactMap { (i, it) in it.frame.map { (i, $0, it.hash) } }

        // Duplicate rejection: keep the sharper image of each near-identical pair.
        if options.rejectDuplicates {
            var kept: [(Int, IngestedFrame, UInt64)] = []
            for c in candidates {
                if let j = kept.firstIndex(where: { ImageMetrics.hamming($0.2, c.2) <= options.duplicateHammingThreshold }) {
                    if c.1.sharpness > kept[j].1.sharpness {
                        set.rejected.append(.init(item: inputs[kept[j].0].lastPathComponent, reason: "duplicate of \(inputs[c.0].lastPathComponent)"))
                        try? fm.removeItem(at: outputRoot.appendingPathComponent(kept[j].1.image))
                        kept[j] = c
                    } else {
                        set.rejected.append(.init(item: inputs[c.0].lastPathComponent, reason: "duplicate of \(inputs[kept[j].0].lastPathComponent)"))
                        try? fm.removeItem(at: outputRoot.appendingPathComponent(c.1.image))
                    }
                } else { kept.append(c) }
            }
            candidates = kept.sorted { $0.0 < $1.0 }
        }

        // Blur rejection relative to the median sharpness (keeps at least 60 % of images).
        if options.rejectBlurry, candidates.count >= 4 {
            let sorted = candidates.map { $0.1.sharpness }.sorted()
            let median = sorted[sorted.count / 2]
            var thr = median * options.blurRelativeThreshold
            if let a = options.blurAbsoluteThreshold { thr = max(thr, a) }
            thr = min(thr, sorted[Int(Double(sorted.count) * 0.4)])
            candidates = candidates.filter { c in
                if c.1.sharpness < thr {
                    set.rejected.append(.init(item: inputs[c.0].lastPathComponent, reason: String(format: "blurry (%.0f < %.0f)", c.1.sharpness, thr)))
                    try? fm.removeItem(at: outputRoot.appendingPathComponent(c.1.image))
                    return false
                }
                return true
            }
        }
        set.frames = candidates.map { $0.1 }
        guard !set.frames.isEmpty else { throw PipelineError.noInput("all images were rejected") }
        try set.save()
        progress?(PipelineProgress(stage: "Importing photos", fraction: 1, message: "\(set.frames.count) kept, \(set.rejected.count) rejected"))
        return set
    }
}

/// Thread-safe counter for progress reporting from parallel work.
final class ProgressCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var n = 0
    let total: Int
    init(total: Int) { self.total = total }
    func increment() -> Int { lock.lock(); defer { lock.unlock() }; n += 1; return n }
}
