import Foundation
import AVFoundation
import CoreImage
import CoreMedia
import CoreVideo
import VideoToolbox
import ImageIO

/// Options for turning a video into training frames.
struct VideoExtractionOptions: Codable, Hashable, Sendable {
    /// Frames per second to keep (one per window of 1/fps seconds). Ignored when `targetFrameCount` is set.
    var targetFPS: Double = 2
    /// Keep exactly this many frames, evenly spread over the (trimmed) video.
    var targetFrameCount: Int? = nil
    /// Longest side of the written images (nil = native resolution).
    var maxImageSize: Int? = 1600
    /// Pick the sharpest decoded frame in each window (otherwise the first one).
    var pickSharpest: Bool = true
    /// Only analyse every Nth decoded frame inside a window (speeds up high-fps footage).
    var analysisStride: Int = 1
    /// Skip windows that straddle a hard scene cut (edited footage), and restart windowing after the cut.
    var skipSceneCuts: Bool = false
    /// Mean absolute luma difference (0…1) between consecutive analysed frames that counts as a cut.
    var sceneCutThreshold: Double = 0.3
    /// Skip frames nearly identical to the previously kept one (camera not moving).
    var skipStaticFrames: Bool = true
    /// Minimum mean absolute luma difference (0…1) to the previous kept frame.
    var staticThreshold: Double = 0.01
    /// Trim: start / end time in seconds.
    var startTime: Double? = nil
    var endTime: Double? = nil
    /// Tone-map 10-bit HLG/PQ footage to SDR (always recommended for training).
    var toneMapHDR: Bool = true
    var encoding: ImageUtil.Encoding = .jpeg

    init() {}
    init(from decoder: Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        targetFPS = c(.targetFPS, targetFPS); targetFrameCount = c(.targetFrameCount, targetFrameCount)
        maxImageSize = c(.maxImageSize, maxImageSize); pickSharpest = c(.pickSharpest, pickSharpest)
        analysisStride = c(.analysisStride, analysisStride); skipSceneCuts = c(.skipSceneCuts, skipSceneCuts)
        sceneCutThreshold = c(.sceneCutThreshold, sceneCutThreshold); skipStaticFrames = c(.skipStaticFrames, skipStaticFrames)
        staticThreshold = c(.staticThreshold, staticThreshold); startTime = c(.startTime, startTime); endTime = c(.endTime, endTime)
        toneMapHDR = c(.toneMapHDR, toneMapHDR); encoding = c(.encoding, encoding)
    }
}

/// One decoded video sample: a single frame, or several views (MV-HEVC left/right layers).
struct DecodedVideoSample {
    var time: Double
    /// Pixel buffers: one for normal video; `[left, right]` for MV-HEVC stereo.
    var buffers: [CVPixelBuffer]
}

/// Thin wrapper around `AVAssetReader` producing pixel buffers (8-bit 4:2:0 for SDR, 10-bit 4:2:0 for HDR),
/// optionally decoding both MV-HEVC layers (Apple spatial video, macOS 14+ APIs).
final class VideoFrameReader {
    let url: URL
    let asset: AVURLAsset
    let track: AVAssetTrack
    let isHDR: Bool
    let isMultiview: Bool
    /// Encoded (un-rotated) size.
    let encodedSize: CGSize
    /// Orientation that must be applied to decoded buffers (from `preferredTransform`).
    let orientation: CGImagePropertyOrientation
    let nominalFPS: Double
    let duration: Double
    let formatDescription: CMFormatDescription?
    /// 35 mm-equivalent focal length from QuickTime metadata if present (iPhone / many cameras).
    let focal35mm: Double?

    private var reader: AVAssetReader?
    private var output: AVAssetReaderTrackOutput?
    private let decodeLayers: Bool

    static func videoTracks(_ url: URL) async throws -> [AVAssetTrack] {
        try await AVURLAsset(url: url).loadTracks(withMediaType: .video)
    }

    /// - Parameters:
    ///   - trackIndex: which video track (dual-stream recordings have two).
    ///   - decodeStereoLayers: decode both MV-HEVC views if the track is multiview.
    init(url: URL, trackIndex: Int = 0, decodeStereoLayers: Bool = false, start: Double? = nil, end: Double? = nil) async throws {
        self.url = url
        asset = AVURLAsset(url: url, options: [AVURLAssetPreferPreciseDurationAndTimingKey: true])
        let tracks = try await asset.loadTracks(withMediaType: .video)
        guard trackIndex < tracks.count else { throw PipelineError.noInput("\(url.lastPathComponent) has no video track \(trackIndex)") }
        track = tracks[trackIndex]
        let (size, transform, fps, chars, fds) = try await track.load(.naturalSize, .preferredTransform, .nominalFrameRate, .mediaCharacteristics, .formatDescriptions)
        encodedSize = size
        nominalFPS = Double(fps > 0 ? fps : 30)
        duration = try await asset.load(.duration).seconds
        formatDescription = fds.first
        isMultiview = chars.contains(.containsStereoMultiviewVideo)
        decodeLayers = decodeStereoLayers && isMultiview
        orientation = Self.orientation(from: transform)
        var hdr = chars.contains(.containsHDRVideo)
        if let fd = fds.first, let tf = CMFormatDescriptionGetExtension(fd, extensionKey: kCMFormatDescriptionExtension_TransferFunction) as? String {
            hdr = hdr || tf == (kCMFormatDescriptionTransferFunction_ITU_R_2100_HLG as String) || tf == (kCMFormatDescriptionTransferFunction_SMPTE_ST_2084_PQ as String)
        }
        isHDR = hdr
        var f35: Double? = nil
        if let md = try? await asset.load(.metadata) {
            for item in md {
                let key = (item.identifier?.rawValue ?? "") + " " + ((item.key as? String) ?? "")
                if key.contains("focal_length.35mm_equivalent"), let v = try? await item.load(.numberValue) { f35 = v.doubleValue }
            }
        }
        focal35mm = f35
        try startReading(start: start, end: end)
    }

    /// Oriented (display) size of one decoded buffer.
    var displaySize: CGSize {
        switch orientation {
        case .left, .right, .leftMirrored, .rightMirrored: return CGSize(width: encodedSize.height, height: encodedSize.width)
        default: return encodedSize
        }
    }

    /// MV-HEVC metadata: stereo baseline in metres and horizontal FOV in degrees (if present).
    var stereoMetadata: (baseline: Float?, horizontalFOV: Float?) {
        guard let fd = formatDescription else { return (nil, nil) }
        var b: Float? = nil, fov: Float? = nil
        if let v = CMFormatDescriptionGetExtension(fd, extensionKey: kCMFormatDescriptionExtension_StereoCameraBaseline) as? NSNumber {
            b = v.floatValue / 1_000_000   // micrometres → metres
        }
        if let v = CMFormatDescriptionGetExtension(fd, extensionKey: kCMFormatDescriptionExtension_HorizontalFieldOfView) as? NSNumber {
            fov = v.floatValue / 1000      // thousandths of a degree
        }
        return (b, fov)
    }

    private func layerIDs() -> [Int64] {
        if let tc = formatDescription?.tagCollections {
            let ids = tc.flatMap { $0.filter(matchingCategory: .videoLayerID).map { $0.value } }
            if ids.count >= 2 { return Array(ids.prefix(2)) }
        }
        return [0, 1]
    }

    private func startReading(start: Double?, end: Double?) throws {
        let r = try AVAssetReader(asset: asset)
        let s = CMTime(seconds: max(0, start ?? 0), preferredTimescale: 600)
        let e = end.map { CMTime(seconds: $0, preferredTimescale: 600) } ?? .positiveInfinity
        r.timeRange = CMTimeRange(start: s, end: e.isValid && !e.isPositiveInfinity ? e : CMTime(seconds: max(duration, 0.001), preferredTimescale: 600))
        var settings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: isHDR ? kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange : kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
        ]
        if decodeLayers {
            settings[AVVideoDecompressionPropertiesKey] = [kVTDecompressionPropertyKey_RequestedMVHEVCVideoLayerIDs as String: layerIDs().map { NSNumber(value: $0) }]
        }
        let o = AVAssetReaderTrackOutput(track: track, outputSettings: settings)
        o.alwaysCopiesSampleData = false
        guard r.canAdd(o) else { throw PipelineError.unsupported("cannot decode \(url.lastPathComponent)") }
        r.add(o)
        guard r.startReading() else { throw PipelineError.decodeFailed("\(url.lastPathComponent): \(r.error?.localizedDescription ?? "")") }
        reader = r; output = o
    }

    /// Next decoded sample, or nil at the end. Throws on decode errors.
    func next() throws -> DecodedVideoSample? {
        guard let output, let reader else { return nil }
        while let sb = output.copyNextSampleBuffer() {
            let t = CMSampleBufferGetPresentationTimeStamp(sb).seconds
            if decodeLayers, let tagged = sb.taggedBuffers, !tagged.isEmpty {
                var left: CVPixelBuffer?, right: CVPixelBuffer?, others: [CVPixelBuffer] = []
                for tb in tagged {
                    guard case let .pixelBuffer(pb) = tb.buffer else { continue }
                    let eye = tb.tags.firstValue(matchingCategory: .stereoView)
                    if eye?.contains(.leftEye) == true && left == nil { left = pb }
                    else if eye?.contains(.rightEye) == true && right == nil { right = pb }
                    else { others.append(pb) }
                }
                let views = [left, right].compactMap { $0 } + others
                if !views.isEmpty { return DecodedVideoSample(time: t, buffers: Array(views.prefix(2))) }
            }
            if let pb = CMSampleBufferGetImageBuffer(sb) { return DecodedVideoSample(time: t, buffers: [pb]) }
        }
        if reader.status == .failed { throw PipelineError.decodeFailed("\(url.lastPathComponent): \(reader.error?.localizedDescription ?? "")") }
        return nil
    }

    func cancel() { reader?.cancelReading() }

    static func orientation(from t: CGAffineTransform) -> CGImagePropertyOrientation {
        let angle = atan2(t.b, t.a) * 180 / .pi
        switch Int(angle.rounded()) {
        case 90: return .right
        case -90: return .left
        case 180, -180: return .down
        default: return .up
        }
    }

    // MARK: Pixel helpers

    /// Strided luma thumbnail (8-bit) of a 4:2:0 / BGRA buffer — fast sharpness & cut analysis.
    static func lumaThumbnail(_ pb: CVPixelBuffer, maxWidth: Int = 480, crop: CGRect? = nil) -> (pixels: [UInt8], width: Int, height: Int) {
        CVPixelBufferLockBaseAddress(pb, .readOnly); defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        let fmt = CVPixelBufferGetPixelFormatType(pb)
        let planar = CVPixelBufferIsPlanar(pb)
        let base = planar ? CVPixelBufferGetBaseAddressOfPlane(pb, 0) : CVPixelBufferGetBaseAddress(pb)
        let bpr = planar ? CVPixelBufferGetBytesPerRowOfPlane(pb, 0) : CVPixelBufferGetBytesPerRow(pb)
        let fullW = planar ? CVPixelBufferGetWidthOfPlane(pb, 0) : CVPixelBufferGetWidth(pb)
        let fullH = planar ? CVPixelBufferGetHeightOfPlane(pb, 0) : CVPixelBufferGetHeight(pb)
        guard let base else { return ([], 0, 0) }
        let r = crop ?? CGRect(x: 0, y: 0, width: fullW, height: fullH)
        let x0 = Int(r.minX), y0 = Int(r.minY), cw = Int(r.width), ch = Int(r.height)
        let step = max(1, cw / maxWidth)
        let w = cw / step, h = ch / step
        var out = [UInt8](repeating: 0, count: w * h)
        let is16 = fmt == kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange || fmt == kCVPixelFormatType_420YpCbCr10BiPlanarFullRange
        let isBGRA = fmt == kCVPixelFormatType_32BGRA
        for y in 0..<h {
            let row = base + (y0 + y * step) * bpr
            for x in 0..<w {
                let sx = x0 + x * step
                if is16 { out[y * w + x] = UInt8(row.load(fromByteOffset: sx * 2, as: UInt16.self) >> 8) }
                else if isBGRA { out[y * w + x] = row.load(fromByteOffset: sx * 4 + 1, as: UInt8.self) }
                else { out[y * w + x] = row.load(fromByteOffset: sx, as: UInt8.self) }
            }
        }
        return (out, w, h)
    }

    /// Converts a decoded buffer to an oriented, SDR, optionally cropped and downscaled CGImage.
    /// `crop` is in *oriented* pixel coordinates with top-left origin.
    static func render(_ pb: CVPixelBuffer, orientation: CGImagePropertyOrientation, hdr: Bool, crop: CGRect? = nil,
                       maxSize: Int? = nil) throws -> CGImage {
        var img = CIImage(cvPixelBuffer: pb, options: hdr ? [.toneMapHDRtoSDR: true] : nil).oriented(orientation)
        img = img.transformed(by: CGAffineTransform(translationX: -img.extent.minX, y: -img.extent.minY))
        if let crop {
            // CI origin is bottom-left.
            let H = img.extent.height
            let r = CGRect(x: crop.minX, y: H - crop.maxY, width: crop.width, height: crop.height)
            img = img.cropped(to: r).transformed(by: CGAffineTransform(translationX: -r.minX, y: -r.minY))
        }
        if let maxSize, max(img.extent.width, img.extent.height) > CGFloat(maxSize) {
            let s = CGFloat(maxSize) / max(img.extent.width, img.extent.height)
            let f = CIFilter(name: "CILanczosScaleTransform")!
            f.setValue(img, forKey: kCIInputImageKey); f.setValue(s, forKey: kCIInputScaleKey); f.setValue(1.0, forKey: kCIInputAspectRatioKey)
            if let o = f.outputImage { img = o }
            let target = CGRect(x: 0, y: 0, width: (img.extent.width).rounded(.down), height: (img.extent.height).rounded(.down))
            img = img.cropped(to: target)
        }
        return hdr ? try ImageUtil.toneMapToSDR(img) : try render8(img)
    }

    private static func render8(_ img: CIImage) throws -> CGImage {
        guard let cg = ImageUtil.ciContext.createCGImage(img, from: img.extent.integral, format: .RGBA8, colorSpace: ImageUtil.sRGB) else {
            throw PipelineError.decodeFailed("video frame")
        }
        return cg
    }
}

/// Selects one frame per time window (sharpest), handles scene cuts and static camera, and hands each
/// selected sample to a writer. Shared by mono, 360° and stereo video ingest.
enum VideoFrameSelector {
    /// - Parameters:
    ///   - sharpnessCrop: region (encoded-buffer pixels) used for sharpness, e.g. the left eye of an SBS frame.
    ///   - write: called for every selected sample with its output index; returns the frames it wrote.
    static func run(reader: VideoFrameReader, options: VideoExtractionOptions, stage: String,
                    sharpnessCrop: CGRect? = nil,
                    progress: PipelineProgressHandler?,
                    write: (DecodedVideoSample, Int, Double) throws -> [IngestedFrame]) async throws -> (frames: [IngestedFrame], rejected: [FrameSet.Rejection]) {
        let start = options.startTime ?? 0
        let end = min(options.endTime ?? reader.duration, reader.duration)
        let span = max(end - start, 1e-3)
        let window: Double = {
            if let n = options.targetFrameCount, n > 0 { return span / Double(n) }
            return 1.0 / max(options.targetFPS, 0.01)
        }()
        var frames: [IngestedFrame] = []
        var rejected: [FrameSet.Rejection] = []
        var best: (sample: DecodedVideoSample, sharp: Double, thumb: [UInt8])? = nil
        var currentWindow = -1
        var windowOrigin = start
        var lastThumb: [UInt8]? = nil
        var lastKeptThumb: [UInt8]? = nil
        var decoded = 0
        var outIndex = 0
        var lastReport = Date.distantPast

        func flush() throws {
            guard let b = best else { return }
            best = nil
            if options.skipStaticFrames, let k = lastKeptThumb, ImageMetrics.difference(k, b.thumb) < options.staticThreshold {
                rejected.append(.init(item: String(format: "t=%.2fs", b.sample.time), reason: "static (camera not moving)"))
                return
            }
            let fs = try write(b.sample, outIndex, b.sharp)
            outIndex += 1
            lastKeptThumb = b.thumb
            frames.append(contentsOf: fs)
        }

        while let s = try reader.next() {
            try Task.checkCancellation()
            decoded += 1
            let t = s.time
            if t < start - 1e-6 { continue }
            if t > end + 1e-6 { break }
            let w = Int(floor((t - windowOrigin) / window))
            let analyse = options.pickSharpest ? (decoded % max(1, options.analysisStride) == 0 || w != currentWindow) : (w != currentWindow)
            if !analyse { continue }
            let luma = VideoFrameReader.lumaThumbnail(s.buffers[0], crop: sharpnessCrop)
            let thumb = ImageUtil.gray8FromLuma(luma.pixels, width: luma.width, height: luma.height)
            if options.skipSceneCuts, let lt = lastThumb, ImageMetrics.difference(lt, thumb) > options.sceneCutThreshold {
                // Hard cut: discard the window in progress and restart windows at the cut.
                if let b = best { rejected.append(.init(item: String(format: "t=%.2fs", b.sample.time), reason: "scene cut")) }
                best = nil
                windowOrigin = t
                currentWindow = 0
                lastThumb = thumb
                lastKeptThumb = nil
                continue
            }
            lastThumb = thumb
            if w != currentWindow { try flush(); currentWindow = w }
            if best == nil || options.pickSharpest {
                let sharp = ImageMetrics.sharpness(gray: luma.pixels, width: luma.width, height: luma.height)
                if best == nil || sharp > best!.sharp { best = (s, sharp, thumb) }
            }
            if Date().timeIntervalSince(lastReport) > 0.2 {
                lastReport = Date()
                progress?(PipelineProgress(stage: stage, fraction: min(1, (t - start) / span), message: "\(outIndex) frames"))
            }
        }
        try flush()
        return (frames, rejected)
    }
}

extension ImageUtil {
    /// 32×32 thumbnail from a luma buffer (for cut / static detection).
    static func gray8FromLuma(_ px: [UInt8], width: Int, height: Int) -> [UInt8] {
        guard width > 0, height > 0 else { return [] }
        var out = [UInt8](repeating: 0, count: 32 * 32)
        for y in 0..<32 { for x in 0..<32 { out[y * 32 + x] = px[min(height - 1, y * height / 32) * width + min(width - 1, x * width / 32)] } }
        return out
    }
}

/// Monoscopic video → frames.
enum VideoFrameExtractor {
    static let videoExtensions: Set<String> = ["mov", "mp4", "m4v", "avi", "mkv", "insv", "lrv", "mts", "m2ts", "hevc"]

    static func extract(video url: URL, options: VideoExtractionOptions = .init(), into outputRoot: URL,
                        progress: PipelineProgressHandler? = nil) async throws -> FrameSet {
        let stage = "Extracting frames"
        progress?(PipelineProgress(stage: stage, fraction: 0, message: url.lastPathComponent))
        let reader = try await VideoFrameReader(url: url, start: options.startTime, end: options.endTime)
        defer { reader.cancel() }
        let imagesDir = try FileManager.default.ensureDirectory(outputRoot.appendingPathComponent("images"))
        let ext = options.encoding == .jpeg ? "jpg" : "png"
        let hdr = reader.isHDR && options.toneMapHDR
        let (frames, rejected) = try await VideoFrameSelector.run(reader: reader, options: options, stage: stage, progress: progress) { s, idx, sharp in
            let img = try VideoFrameReader.render(s.buffers[0], orientation: reader.orientation, hdr: hdr, maxSize: options.maxImageSize)
            let name = String(format: "%06d.%@", idx, ext)
            try ImageUtil.write(img, to: imagesDir.appendingPathComponent(name), encoding: options.encoding)
            var f = IngestedFrame(image: "images/\(name)", width: img.width, height: img.height, sharpness: sharp, source: url.path, timestamp: s.time)
            if let f35 = reader.focal35mm { f.focalPx = Float(f35 / 36.0) * Float(max(img.width, img.height)) }
            return [f]
        }
        guard !frames.isEmpty else { throw PipelineError.noInput("no frames extracted from \(url.lastPathComponent)") }
        var set = FrameSet(root: outputRoot, frames: frames, rejected: rejected)
        set.frames.sort { ($0.timestamp ?? 0) < ($1.timestamp ?? 0) }
        try set.save()
        progress?(PipelineProgress(stage: stage, fraction: 1, message: "\(frames.count) frames"))
        return set
    }
}
