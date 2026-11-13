import Foundation
import CoreGraphics
import CoreImage
import ImageIO
import Accelerate
import UniformTypeIdentifiers
import simd
import os

// MARK: - Progress

/// Progress report shared by every pipeline stage (ingest, pose solving, import, upload, …).
///
/// `fraction` is the progress of *this stage* in 0…1 (or `-1` when indeterminate). The
/// `DatasetBuilder` façade maps per-stage fractions into an overall fraction for the UI.
struct PipelineProgress: Sendable {
    var stage: String
    /// 0…1, or < 0 when unknown (the UI then keeps the last known value and shows elapsed time).
    var fraction: Double
    var message: String
    /// Seconds remaining for the current stage when the underlying tool estimates it (Object Capture does).
    var stageETA: Double? = nil

    init(stage: String, fraction: Double, message: String = "", stageETA: Double? = nil) {
        self.stage = stage; self.fraction = fraction; self.message = message; self.stageETA = stageETA
    }
}

/// Callback receiving progress updates. Called from arbitrary threads — hop to the main actor in UI code.
typealias PipelineProgressHandler = @Sendable (PipelineProgress) -> Void

extension PipelineProgress {
    /// Returns a handler that rescales the fraction of `inner` reports into `range` of `outer`.
    static func scaled(_ outer: PipelineProgressHandler?, _ range: ClosedRange<Double>) -> PipelineProgressHandler {
        return { p in
            guard let outer else { return }
            let f = p.fraction < 0 ? p.fraction : range.lowerBound + (range.upperBound - range.lowerBound) * min(max(p.fraction, 0), 1)
            outer(PipelineProgress(stage: p.stage, fraction: f, message: p.message, stageETA: p.stageETA))
        }
    }
}

// MARK: - Errors

/// System memory pressure, for long pipeline loops: they stop with an error at "critical" instead of pushing the
/// Mac into swap until it hangs (a leak in video decoding once froze a 24 GB Mac until the watchdog restarted it).
enum MemoryGuard {
    private static let critical = OSAllocatedUnfairLock(initialState: false)
    private static let source: DispatchSourceMemoryPressure = {
        let s = DispatchSource.makeMemoryPressureSource(eventMask: [.normal, .warning, .critical], queue: .global(qos: .utility))
        s.setEventHandler { let c = s.data.contains(.critical); critical.withLock { $0 = c } }
        s.resume()
        return s
    }()

    static var isCritical: Bool { _ = source; return critical.withLock { $0 } }

    /// Throws when memory is critically low.
    static func check(_ stage: String) throws {
        if isCritical {
            throw PipelineError.processFailed("\(stage) stopped: this Mac is almost out of memory. Close other apps and try again.")
        }
    }
}

enum PipelineError: LocalizedError {
    case noInput(String)
    case unsupported(String)
    case decodeFailed(String)
    case encodeFailed(String)
    case poseSolveFailed(String)
    case processFailed(String)
    case invalidData(String)

    var errorDescription: String? {
        switch self {
        case .noInput(let s): return "No usable input: \(s)"
        case .unsupported(let s): return "Unsupported: \(s)"
        case .decodeFailed(let s): return "Could not decode \(s)"
        case .encodeFailed(let s): return "Could not write \(s)"
        case .poseSolveFailed(let s): return "Camera pose estimation failed: \(s)"
        case .processFailed(let s): return "Process failed: \(s)"
        case .invalidData(let s): return "Invalid data: \(s)"
        }
    }
}

// MARK: - Ingest result types

/// One image produced by an ingest stage (photo folder, video, 360°, stereo).
struct IngestedFrame: Codable, Hashable, Sendable {
    /// Path of the written image relative to the frame-set root (e.g. `images/000012.jpg`).
    var image: String
    var width: Int
    var height: Int
    /// Variance of Laplacian on a normalised gray thumbnail (higher = sharper).
    var sharpness: Double = 0
    /// Original file or video this frame came from.
    var source: String? = nil
    /// Presentation time in seconds (video sources).
    var timestamp: Double? = nil
    /// Focal length in pixels of the written image if known (EXIF, 360 view FOV, rig calibration).
    var focalPx: Float? = nil
    /// Principal point if known (defaults to image centre).
    var cx: Float? = nil
    var cy: Float? = nil
    /// Group of frames that were captured at the same instant (360 views of one equirect frame, stereo pair).
    var group: Int? = nil
    /// For 360 views: camera-to-rig rotation (OpenCV camera convention), row-major 3x3. Known exactly.
    var rotationToRig: [[Float]]? = nil
    /// For stereo: "left" or "right".
    var eye: String? = nil
    /// Optional depth map (float16 LE, metres, Z-depth) relative to the frame-set root.
    var depth: String? = nil
    var depthWidth: Int? = nil
    var depthHeight: Int? = nil
    /// Optional mask (8-bit PNG, 255 = keep) relative to the frame-set root.
    var mask: String? = nil
    /// Optional sky mask (8-bit PNG, 255 = sky) relative to the frame-set root.
    var sky: String? = nil
}

/// Description of a stereo camera rig (two horizontally separated, rectified pinhole cameras).
struct StereoRig: Codable, Hashable, Sendable {
    /// Distance between the optical centres in metres (nil = unknown; then scale stays arbitrary).
    var baseline: Float?
    /// Per-eye intrinsics in pixels of the written eye images (nil = unknown, estimate from FOV/EXIF).
    var leftIntrinsics: [Float]? = nil     // fx, fy, cx, cy
    var rightIntrinsics: [Float]? = nil
    /// Right camera pose in the left camera frame (OpenCV), row-major 4x4. Default: pure +X translation by `baseline`.
    var rightToLeft: [[Float]]? = nil
    /// Horizontal field of view in degrees (from MV-HEVC metadata or user), used when intrinsics are unknown.
    var horizontalFOV: Float? = nil
    /// Where the rig description came from ("mvhevc-metadata", "calibration-file", "user", "default").
    var origin: String = "default"

    /// Loads a calibration file. Accepts either this struct as JSON, or a simple OpenCV-style JSON
    /// `{"K1":[9], "K2":[9], "R":[9], "T":[3], "baseline":0.064}` (T in metres).
    static func loadCalibration(_ url: URL) throws -> StereoRig {
        let data = try Data(contentsOf: url)
        if let rig = try? JSONDecoder().decode(StereoRig.self, from: data), rig.baseline != nil || rig.leftIntrinsics != nil {
            var r = rig; r.origin = "calibration-file"; return r
        }
        guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw PipelineError.invalidData("calibration file \(url.lastPathComponent)")
        }
        func arr(_ k: String) -> [Float]? { (obj[k] as? [NSNumber])?.map { $0.floatValue } }
        var rig = StereoRig(baseline: (obj["baseline"] as? NSNumber)?.floatValue, origin: "calibration-file")
        if let k = arr("K1") ?? arr("K_left"), k.count == 9 { rig.leftIntrinsics = [k[0], k[4], k[2], k[5]] }
        if let k = arr("K2") ?? arr("K_right"), k.count == 9 { rig.rightIntrinsics = [k[0], k[4], k[2], k[5]] }
        if let t = arr("T"), t.count == 3 {
            if rig.baseline == nil { rig.baseline = simd_length(SIMD3(t[0], t[1], t[2])) }
            let r = arr("R") ?? [1, 0, 0, 0, 1, 0, 0, 0, 1]
            // OpenCV stereoCalibrate: x_right = R x_left + T  →  right-to-left = [Rᵀ | -RᵀT]
            let R = simd_float3x3(rows: [SIMD3(r[0], r[1], r[2]), SIMD3(r[3], r[4], r[5]), SIMD3(r[6], r[7], r[8])])
            let Rt = R.transpose
            let c = -(Rt * SIMD3(t[0], t[1], t[2]))
            rig.rightToLeft = [[Rt[0][0], Rt[1][0], Rt[2][0], c.x], [Rt[0][1], Rt[1][1], Rt[2][1], c.y],
                               [Rt[0][2], Rt[1][2], Rt[2][2], c.z], [0, 0, 0, 1]]
        }
        if rig.baseline == nil && rig.leftIntrinsics == nil { throw PipelineError.invalidData("calibration file has no baseline or intrinsics") }
        return rig
    }
}

/// Output of an ingest stage: a directory with `images/` (and optional `depth/`, `masks/`) plus frame metadata.
struct FrameSet: Codable, Sendable {
    var root: URL
    var frames: [IngestedFrame] = []
    /// Files/frames that were skipped and why (blurry, duplicate, unreadable, …).
    var rejected: [Rejection] = []
    /// Present for stereo sources.
    var rig: StereoRig? = nil
    /// True when the frames come from a 360° source (views share an optical centre per `group`).
    var isMultiViewRig: Bool = false

    struct Rejection: Codable, Hashable, Sendable { var item: String; var reason: String }

    /// Circular fisheye footage (the image circle inside a black frame, e.g. Arcturus / dual-fisheye stereo
    /// cameras): its corners are black in every sampled frame while the centre isn't. Such frames need a fisheye
    /// camera model — a pinhole solve "succeeds" but bakes the lens distortion into the scene.
    var looksCircularFisheye: Bool {
        let picks = stride(from: 0, to: frames.count, by: max(1, frames.count / 5)).prefix(5)
        var votes = 0, seen = 0
        for i in picks {
            guard let img = try? ImageUtil.load(root.appendingPathComponent(frames[i].image), maxSize: 64) else { continue }
            seen += 1
            if Self.circularFisheye(img) { votes += 1 }
        }
        return seen > 0 && votes * 2 > seen
    }

    static func circularFisheye(_ img: CGImage) -> Bool {
        let w = img.width, h = img.height
        guard w >= 16, h >= 16, abs(Double(w) / Double(h) - 1) < 0.35 else { return false }
        var px = [UInt8](repeating: 0, count: w * h * 4)
        guard let ctx = CGContext(data: &px, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: ImageUtil.sRGB,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return false }
        ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
        /// (mean, max) brightness of a region.
        func stats(_ x0: Int, _ y0: Int, _ x1: Int, _ y1: Int) -> (Double, Int) {
            var s = 0, n = 0, mx = 0
            for y in y0..<y1 { for x in x0..<x1 {
                let k = (y * w + x) * 4, v = max(Int(px[k]), Int(px[k + 1]), Int(px[k + 2]))
                s += v; n += 1; mx = max(mx, v)
            } }
            return (n > 0 ? Double(s) / Double(n) : 0, mx)
        }
        let c = max(2, w / 10), r = max(2, h / 10)
        let corners = [stats(0, 0, c, r), stats(w - c, 0, w, r), stats(0, h - r, c, h), stats(w - c, h - r, w, h)]
        // The image circle may be cut off at one side (lens circle larger than the frame), so two black corners
        // are enough — but they must be uniformly black (masked lens border), not just dark scenery.
        let black = corners.filter { $0.0 < 10 && $0.1 < 40 }.count
        return black >= 2 && stats(w / 3, h / 3, 2 * w / 3, 2 * h / 3).0 > 30
    }

    static let fileName = "frames.json"
    func save() throws {
        let enc = JSONEncoder(); enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        try enc.encode(self).write(to: root.appendingPathComponent(Self.fileName), options: .atomic)
    }
    static func load(_ root: URL) throws -> FrameSet {
        var fs = try JSONDecoder().decode(FrameSet.self, from: Data(contentsOf: root.appendingPathComponent(fileName)))
        fs.root = root
        return fs
    }
    func url(_ rel: String) -> URL { root.appendingPathComponent(rel) }
}

// MARK: - simd helpers

extension simd_float3x3 {
    init(rows r: [SIMD3<Float>]) {
        self.init(SIMD3(r[0].x, r[1].x, r[2].x), SIMD3(r[0].y, r[1].y, r[2].y), SIMD3(r[0].z, r[1].z, r[2].z))
    }
    var rowMajor: [[Float]] { (0..<3).map { r in [self[0][r], self[1][r], self[2][r]] } }
    init(rowMajor m: [[Float]]) {
        self.init(rows: [SIMD3(m[0][0], m[0][1], m[0][2]), SIMD3(m[1][0], m[1][1], m[1][2]), SIMD3(m[2][0], m[2][1], m[2][2])])
    }
}

extension simd_float4x4 {
    init(rotation r: simd_float3x3, translation t: SIMD3<Float>) {
        self.init(SIMD4(r[0], 0), SIMD4(r[1], 0), SIMD4(r[2], 0), SIMD4(t, 1))
    }
    /// Row-major 4x4 nested array (dataset.json convention).
    var rowMajor: [[Float]] { (0..<4).map { r in [self[0][r], self[1][r], self[2][r], self[3][r]] } }
    /// diag(1,-1,-1,1): converts between camera conventions looking down -Z/+Y-up (ARKit, OpenGL, RealityKit) and OpenCV.
    static let flipYZ = simd_float4x4(diagonal: SIMD4(1, -1, -1, 1))
}

/// Normalises a rotation matrix (removes drift / scale) via quaternion round trip.
func orthonormalized(_ m: simd_float3x3) -> simd_float3x3 {
    let s0 = simd_length(m.columns.0), s1 = simd_length(m.columns.1), s2 = simd_length(m.columns.2)
    let n = simd_float3x3(m.columns.0 / max(s0, 1e-9), m.columns.1 / max(s1, 1e-9), m.columns.2 / max(s2, 1e-9))
    return simd_float3x3(simd_quatf(n).normalized)
}

// MARK: - Image I/O

/// Image decoding/encoding helpers (ImageIO + Core Image), applying EXIF orientation and HDR→SDR.
enum ImageUtil {
    static let photoExtensions: Set<String> = ["jpg", "jpeg", "heic", "heif", "png", "tif", "tiff", "webp", "avif", "jxl", "bmp"]
    static let rawExtensions: Set<String> = ["dng", "cr2", "cr3", "nef", "nrw", "arw", "srf", "sr2", "raf", "orf", "rw2", "pef", "srw", "3fr", "iiq", "erf", "x3f"]
    static var allExtensions: Set<String> { photoExtensions.union(rawExtensions) }

    static let ciContext: CIContext = CIContext(options: [.cacheIntermediates: false, .workingColorSpace: CGColorSpace(name: CGColorSpace.extendedLinearSRGB)!])
    static let sRGB = CGColorSpace(name: CGColorSpace.sRGB)!

    static func isImageFile(_ url: URL) -> Bool { allExtensions.contains(url.pathExtension.lowercased()) }

    /// Decodes any supported image with EXIF orientation applied, HDR gain maps / 10-bit content tone-mapped
    /// to SDR, and optional downscale so that the longest side is ≤ `maxSize`.
    /// - Parameter applyOrientation: set false for posed datasets (COLMAP/Nerfstudio/captures) whose intrinsics
    ///   refer to the stored pixel layout.
    static func load(_ url: URL, maxSize: Int? = nil, applyOrientation: Bool = true) throws -> CGImage {
        if rawExtensions.contains(url.pathExtension.lowercased()) {
            return try loadRAW(url, maxSize: maxSize)
        }
        guard let src = CGImageSourceCreateWithURL(url as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary) else {
            throw PipelineError.decodeFailed(url.lastPathComponent)
        }
        let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any] ?? [:]
        let w = (props[kCGImagePropertyPixelWidth] as? Int) ?? 0, h = (props[kCGImagePropertyPixelHeight] as? Int) ?? 0
        let longest = max(w, h)
        if !applyOrientation, maxSize == nil || longest <= maxSize!,
           let full = CGImageSourceCreateImageAtIndex(src, 0, [kCGImageSourceDecodeRequest: kCGImageSourceDecodeToSDR] as CFDictionary) {
            if full.bitsPerComponent > 8 || isHDRColorSpace(full.colorSpace) {
                return try toneMapToSDR(CIImage(cgImage: full, options: [.toneMapHDRtoSDR: true]))
            }
            return full
        }
        let target = min(maxSize ?? longest, longest > 0 ? longest : (maxSize ?? 100_000))
        var opts: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: applyOrientation,   // applies EXIF orientation
            kCGImageSourceShouldCacheImmediately: true,
            kCGImageSourceThumbnailMaxPixelSize: max(target, 1),
        ]
        opts[kCGImageSourceDecodeRequest] = kCGImageSourceDecodeToSDR
        guard let img = CGImageSourceCreateThumbnailAtIndex(src, 0, opts as CFDictionary) else {
            throw PipelineError.decodeFailed(url.lastPathComponent)
        }
        // 10/16-bit or HDR (PQ/HLG) content → SDR 8-bit sRGB.
        if img.bitsPerComponent > 8 || isHDRColorSpace(img.colorSpace) {
            return try toneMapToSDR(CIImage(cgImage: img, options: [.toneMapHDRtoSDR: true]))
        }
        return img
    }

    static func isHDRColorSpace(_ cs: CGColorSpace?) -> Bool {
        guard let cs else { return false }
        if #available(macOS 11.0, *) { return CGColorSpaceUsesITUR_2100TF(cs) }
        return false
    }

    /// Renders a (possibly HDR) CIImage into an 8-bit sRGB CGImage, tone-mapping values above 1.0.
    static func toneMapToSDR(_ image: CIImage) throws -> CGImage {
        var img = image
        if #available(macOS 15.0, *) {
            let f = CIFilter(name: "CIToneMapHeadroom")
            f?.setValue(img, forKey: kCIInputImageKey)
            f?.setValue(1.0, forKey: "inputTargetHeadroom")
            if let out = f?.outputImage { img = out }
        }
        guard let cg = ciContext.createCGImage(img, from: img.extent.integral, format: .RGBA8, colorSpace: sRGB) else {
            throw PipelineError.decodeFailed("tone mapping")
        }
        return cg
    }

    static func loadRAW(_ url: URL, maxSize: Int?) throws -> CGImage {
        guard let raw = CIRAWFilter(imageURL: url) else { throw PipelineError.decodeFailed(url.lastPathComponent) }
        let native = raw.nativeSize
        if let maxSize, max(native.width, native.height) > CGFloat(maxSize) {
            raw.scaleFactor = Float(CGFloat(maxSize) / max(native.width, native.height))
        }
        guard let out = raw.outputImage else { throw PipelineError.decodeFailed(url.lastPathComponent) }
        return try toneMapToSDR(out)
    }

    /// EXIF-derived focal length in pixels for an image of `width`×`height` (after orientation). Nil if unknown.
    static func exifFocalPixels(_ url: URL, width: Int, height: Int) -> Float? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
              let exif = props[kCGImagePropertyExifDictionary] as? [CFString: Any] else { return nil }
        let longest = Float(max(width, height))
        if let f35 = (exif[kCGImagePropertyExifFocalLenIn35mmFilm] as? NSNumber)?.floatValue, f35 > 0 {
            return f35 / 36.0 * longest
        }
        if let f = (exif[kCGImagePropertyExifFocalLength] as? NSNumber)?.floatValue, f > 0,
           let xres = (exif[kCGImagePropertyExifFocalPlaneXResolution] as? NSNumber)?.floatValue, xres > 0 {
            let unit = (exif[kCGImagePropertyExifFocalPlaneResolutionUnit] as? NSNumber)?.intValue ?? 2
            let mmPerUnit: Float = unit == 3 ? 10 : unit == 4 ? 1 : 25.4
            let pw = (props[kCGImagePropertyPixelWidth] as? NSNumber)?.floatValue ?? Float(width)
            let ph = (props[kCGImagePropertyPixelHeight] as? NSNumber)?.floatValue ?? Float(height)
            let pixelsPerMM = xres / mmPerUnit          // relative to the original (full-resolution) sensor width
            return f * pixelsPerMM * (longest / max(max(pw, ph), 1))
        }
        return nil
    }

    /// Default focal length guess when nothing is known: ~53° horizontal FOV on the longest side.
    static func defaultFocal(width: Int, height: Int) -> Float { 1.0 * Float(max(width, height)) }

    // MARK: Writing

    enum Encoding: String, Codable, Sendable { case jpeg, png }

    /// - Parameter location: copied into the file (GPS + DJI relative altitude only); nil writes no metadata.
    static func write(_ image: CGImage, to url: URL, encoding: Encoding = .jpeg, quality: Double = 0.95,
                      location: PhotoLocation? = nil) throws {
        let type: UTType = encoding == .jpeg ? .jpeg : .png
        guard let dst = CGImageDestinationCreateWithURL(url as CFURL, type.identifier as CFString, 1, nil) else {
            throw PipelineError.encodeFailed(url.lastPathComponent)
        }
        let opts = [kCGImageDestinationLossyCompressionQuality: quality] as CFDictionary
        if let location { CGImageDestinationAddImageAndMetadata(dst, image, location.metadata(), opts) }
        else { CGImageDestinationAddImage(dst, image, opts) }
        guard CGImageDestinationFinalize(dst) else { throw PipelineError.encodeFailed(url.lastPathComponent) }
    }

    /// Writes an 8-bit single-channel image (e.g. a mask) as PNG.
    static func writeGray8(_ pixels: [UInt8], width: Int, height: Int, to url: URL) throws {
        let cs = CGColorSpaceCreateDeviceGray()
        let data = Data(pixels) as CFData
        guard let provider = CGDataProvider(data: data),
              let img = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: width,
                                space: cs, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
        else { throw PipelineError.encodeFailed(url.lastPathComponent) }
        try write(img, to: url, encoding: .png)
    }

    /// Reads an 8-bit gray image (masks), resampled to `width`×`height` if given.
    static func readGray8(_ url: URL, width: Int? = nil, height: Int? = nil) throws -> (pixels: [UInt8], width: Int, height: Int) {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil), let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            throw PipelineError.decodeFailed(url.lastPathComponent)
        }
        let w = width ?? img.width, h = height ?? img.height
        return (gray8(img, width: w, height: h), w, h)
    }

    /// Renders `image` to an 8-bit gray buffer of the given size.
    static func gray8(_ image: CGImage, width: Int, height: Int) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: width * height)
        out.withUnsafeMutableBytes { raw in
            let ctx = CGContext(data: raw.baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width,
                                space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue)
            ctx?.interpolationQuality = .medium
            ctx?.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        return out
    }

    /// Renders `image` into a tightly packed RGBA8 (premultiplied-last, sRGB) buffer.
    static func rgba8(_ image: CGImage, width: Int? = nil, height: Int? = nil) -> (pixels: [UInt8], width: Int, height: Int) {
        let w = width ?? image.width, h = height ?? image.height
        var out = [UInt8](repeating: 0, count: w * h * 4)
        out.withUnsafeMutableBytes { raw in
            let ctx = CGContext(data: raw.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                space: sRGB, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            ctx?.interpolationQuality = .high
            ctx?.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        }
        return (out, w, h)
    }

    static func cgImage(rgba pixels: [UInt8], width: Int, height: Int) -> CGImage? {
        guard let provider = CGDataProvider(data: Data(pixels) as CFData) else { return nil }
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4, space: sRGB,
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
    }

    /// Downscales so the longest side is ≤ maxSize (no-op when already smaller).
    static func downscaled(_ image: CGImage, maxSize: Int?) -> CGImage {
        guard let maxSize, max(image.width, image.height) > maxSize else { return image }
        let s = Double(maxSize) / Double(max(image.width, image.height))
        let w = max(1, Int((Double(image.width) * s).rounded())), h = max(1, Int((Double(image.height) * s).rounded()))
        let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0, space: sRGB,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        ctx?.interpolationQuality = .high
        ctx?.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        return ctx?.makeImage() ?? image
    }

    /// Image size without decoding pixels (orientation applied).
    static func pixelSize(_ url: URL) -> (width: Int, height: Int)? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let p = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
              let w = p[kCGImagePropertyPixelWidth] as? Int, let h = p[kCGImagePropertyPixelHeight] as? Int else { return nil }
        let o = (p[kCGImagePropertyOrientation] as? Int) ?? 1
        return o >= 5 ? (h, w) : (w, h)
    }
}

// MARK: - Image quality metrics

/// Sharpness and near-duplicate metrics computed with vImage/vDSP on a small gray thumbnail.
enum ImageMetrics {
    /// Variance of the Laplacian of a gray image resized to ≤ `analysisSize` on the longest side.
    /// Values are in (0…255)² units; typical sharp photos are > 100, motion-blurred frames < 30.
    static func sharpness(_ image: CGImage, analysisSize: Int = 640) -> Double {
        let s = Double(analysisSize) / Double(max(image.width, image.height))
        let w = max(8, Int(Double(image.width) * min(1, s))), h = max(8, Int(Double(image.height) * min(1, s)))
        return sharpness(gray: ImageUtil.gray8(image, width: w, height: h), width: w, height: h)
    }

    static func sharpness(gray: [UInt8], width w: Int, height h: Int) -> Double {
        var src = [Float](repeating: 0, count: w * h)
        vDSP.convertElements(of: gray, to: &src)
        var dst = [Float](repeating: 0, count: w * h)
        let kernel: [Float] = [0, 1, 0, 1, -4, 1, 0, 1, 0]
        src.withUnsafeMutableBufferPointer { sp in
            dst.withUnsafeMutableBufferPointer { dp in
                var sb = vImage_Buffer(data: sp.baseAddress, height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: w * 4)
                var db = vImage_Buffer(data: dp.baseAddress, height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: w * 4)
                _ = vImageConvolve_PlanarF(&sb, &db, nil, 0, 0, kernel, 3, 3, 0, vImage_Flags(kvImageEdgeExtend))
            }
        }
        // Ignore a 1-px border.
        var mean: Float = 0, meanSq: Float = 0
        var inner: [Float] = []
        inner.reserveCapacity((w - 2) * (h - 2))
        for y in 1..<(h - 1) { inner.append(contentsOf: dst[(y * w + 1)..<(y * w + w - 1)]) }
        vDSP_meanv(inner, 1, &mean, vDSP_Length(inner.count))
        vDSP_measqv(inner, 1, &meanSq, vDSP_Length(inner.count))
        return Double(meanSq - mean * mean)
    }

    /// 64-bit difference hash (dHash) for near-duplicate detection.
    static func dHash(_ image: CGImage) -> UInt64 {
        let g = ImageUtil.gray8(image, width: 9, height: 8)
        var h: UInt64 = 0
        for y in 0..<8 { for x in 0..<8 { h = (h << 1) | (g[y * 9 + x] > g[y * 9 + x + 1] ? 1 : 0) } }
        return h
    }

    static func hamming(_ a: UInt64, _ b: UInt64) -> Int { (a ^ b).nonzeroBitCount }

    /// Mean absolute difference of two 32×32 gray thumbnails in 0…1 (scene-change detection).
    static func thumbnail(_ image: CGImage) -> [UInt8] { ImageUtil.gray8(image, width: 32, height: 32) }
    static func difference(_ a: [UInt8], _ b: [UInt8]) -> Double {
        guard a.count == b.count, !a.isEmpty else { return 1 }
        var s = 0
        for i in 0..<a.count { s += abs(Int(a[i]) - Int(b[i])) }
        return Double(s) / Double(a.count) / 255
    }
}

// MARK: - Process helpers

enum ProcessUtil {
    /// Runs a command to completion, throwing with captured stderr on non-zero exit. Cancels (terminates) with the task.
    @discardableResult
    static func run(_ exe: String, _ args: [String], cwd: URL? = nil, environment: [String: String]? = nil) async throws -> String {
        #if os(macOS)
        let p = Process()
        p.executableURL = URL(fileURLWithPath: exe)
        p.arguments = args
        if let cwd { p.currentDirectoryURL = cwd }
        if let environment { p.environment = environment }
        let out = Pipe(), err = Pipe()
        p.standardOutput = out; p.standardError = err
        let outBox = DataBox(), errBox = DataBox()
        out.fileHandleForReading.readabilityHandler = { h in outBox.append(h.availableData) }
        err.fileHandleForReading.readabilityHandler = { h in errBox.append(h.availableData) }
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
                p.terminationHandler = { _ in cont.resume() }
                do { try p.run() } catch { p.terminationHandler = nil; cont.resume(throwing: error) }
            }
        } onCancel: { if p.isRunning { p.terminate() } }
        out.fileHandleForReading.readabilityHandler = nil
        err.fileHandleForReading.readabilityHandler = nil
        outBox.append(out.fileHandleForReading.readDataToEndOfFile())
        errBox.append(err.fileHandleForReading.readDataToEndOfFile())
        try Task.checkCancellation()
        guard p.terminationStatus == 0 else {
            throw PipelineError.processFailed("\(URL(fileURLWithPath: exe).lastPathComponent) exited \(p.terminationStatus): \(String(decoding: errBox.data, as: UTF8.self).suffix(2000))")
        }
        return String(decoding: outBox.data, as: UTF8.self)
        #else
        throw PipelineError.unsupported("\(URL(fileURLWithPath: exe).lastPathComponent) can't run on iOS")
        #endif
    }

    /// Thread-safe byte accumulator.
    final class DataBox: @unchecked Sendable {
        private let lock = NSLock()
        private var _data = Data()
        func append(_ d: Data) { lock.lock(); _data.append(d); lock.unlock() }
        var data: Data { lock.lock(); defer { lock.unlock() }; return _data }
    }
}

/// Zip helpers built on `/usr/bin/ditto` (preserves the directory as the single top-level entry). On iOS
/// (no ditto) the pure-Swift ZipWriter (stored entries) / ZipReader (stored + deflate) from the iPhone app.
enum ZipUtil {
    /// `ditto -c -k --keepParent <dir> <zip>`
    static func zip(directory: URL, to zip: URL) async throws {
        try? FileManager.default.removeItem(at: zip)
        #if os(macOS)
        try await ProcessUtil.run("/usr/bin/ditto", ["-c", "-k", "--sequesterRsrc", "--keepParent", directory.path, zip.path])
        #else
        try await Task.detached { try ZipWriter.zipDirectory(directory, to: zip) }.value
        #endif
    }
    /// Zips the *contents* of a directory (no top-level folder).
    static func zipContents(of directory: URL, to zip: URL) async throws {
        try? FileManager.default.removeItem(at: zip)
        #if os(macOS)
        try await ProcessUtil.run("/usr/bin/ditto", ["-c", "-k", "--sequesterRsrc", directory.path, zip.path])
        #else
        try await Task.detached { try ZipWriter.zipDirectory(directory, to: zip, keepParent: false) }.value
        #endif
    }
    /// `ditto -x -k <zip> <dir>`
    static func unzip(_ zip: URL, to directory: URL) async throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        #if os(macOS)
        try await ProcessUtil.run("/usr/bin/ditto", ["-x", "-k", zip.path, directory.path])
        #else
        try await Task.detached { _ = try ZipReader.extract(zip, to: directory, maxBytes: UInt64.max) }.value
        #endif
    }
}

/// This device's display name ("Ben's MacBook Pro"; on iOS "This iPhone" — the real name needs an entitlement).
enum LocalDeviceName {
    static var current: String {
        #if os(macOS)
        return Host.current().localizedName ?? "This Mac"
        #else
        return "This iPhone"
        #endif
    }
}

extension FileManager {
    /// Creates `url` (and parents) if missing; returns it.
    @discardableResult
    func ensureDirectory(_ url: URL) throws -> URL {
        try createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
    /// Recreates `src` at `dst`, hard-linking files (falls back to copying across volumes).
    func linkOrCopyTree(_ src: URL, to dst: URL) throws {
        var isDir: ObjCBool = false
        guard fileExists(atPath: src.path, isDirectory: &isDir) else { return }
        if !isDir.boolValue {
            try createDirectory(at: dst.deletingLastPathComponent(), withIntermediateDirectories: true)
            if fileExists(atPath: dst.path) { try removeItem(at: dst) }
            do { try linkItem(at: src, to: dst) } catch { try copyItem(at: src, to: dst) }
            return
        }
        try createDirectory(at: dst, withIntermediateDirectories: true)
        for c in try contentsOfDirectory(at: src, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]) {
            try linkOrCopyTree(c, to: dst.appendingPathComponent(c.lastPathComponent))
        }
    }

    /// Copies, replacing an existing destination.
    func copyReplacing(_ src: URL, to dst: URL) throws {
        if fileExists(atPath: dst.path) { try removeItem(at: dst) }
        try createDirectory(at: dst.deletingLastPathComponent(), withIntermediateDirectories: true)
        try copyItem(at: src, to: dst)
    }
}

/// Parallel photo decodes for dataset writing: every core on a Mac; on iOS a few (HEIC goes through the
/// shared hardware decoder, and each full-size decode is ~50 MB of the app's memory budget).
var imageDecodeConcurrency: Int {
    #if os(macOS)
    return ProcessInfo.processInfo.activeProcessorCount
    #else
    return 2
    #endif
}

/// Runs `body` for indices 0..<count with bounded parallelism, preserving result order. Cancellable.
func parallelMap<T: Sendable>(_ count: Int, maxConcurrent: Int = ProcessInfo.processInfo.activeProcessorCount,
                              _ body: @escaping @Sendable (Int) throws -> T) async throws -> [T] {
    var results = [T?](repeating: nil, count: count)
    try await withThrowingTaskGroup(of: (Int, T).self) { group in
        var next = 0
        let limit = max(1, min(maxConcurrent, count))
        while next < limit { let i = next; group.addTask { try Task.checkCancellation(); return (i, try body(i)) }; next += 1 }
        while let (i, r) = try await group.next() {
            results[i] = r
            if next < count { let j = next; group.addTask { try Task.checkCancellation(); return (j, try body(j)) }; next += 1 }
        }
    }
    return results.map { $0! }
}

// MARK: - Tolerant decoding

extension KeyedDecodingContainer {
    /// `c(.key, default)` — decodes the key if present and valid, otherwise returns `def`.
    /// Used by all option structs so that settings persisted by older/newer app versions still load.
    func callAsFunction<T: Decodable>(_ key: Key, _ def: T) -> T {
        ((try? decodeIfPresent(T.self, forKey: key)) ?? nil) ?? def
    }
}

extension ImageUtil {
    /// Stored pixel size (EXIF orientation NOT applied).
    static func pixelSizeRaw(_ url: URL) -> (Int, Int)? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let p = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
              let w = p[kCGImagePropertyPixelWidth] as? Int, let h = p[kCGImagePropertyPixelHeight] as? Int else { return nil }
        return (w, h)
    }
}

// MARK: - Display formatting

func formatCount(_ n: Int) -> String {
    if n >= 1_000_000 { return String(format: "%.2f M", Double(n) / 1e6) }
    if n >= 10_000 { return String(format: "%.0f k", Double(n) / 1e3) }
    return "\(n)"
}

func formatDuration(_ s: Double) -> String {
    guard s.isFinite, s >= 0 else { return "–" }
    let t = Int(s)
    if t >= 3600 { return String(format: "%d h %02d min", t / 3600, (t % 3600) / 60) }
    if t >= 60 { return String(format: "%d min %02d s", t / 60, t % 60) }
    return "\(t) s"
}

func formatBytes(_ b: Int) -> String { ByteCountFormatter.string(fromByteCount: Int64(b), countStyle: .file) }
