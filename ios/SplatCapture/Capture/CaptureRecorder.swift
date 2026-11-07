import CoreImage
import CoreVideo
import Foundation
import ImageIO
import os
import simd

/// Everything needed to persist one keyframe. The pixel buffer is our own copy.
struct KeyframeJob: @unchecked Sendable {
    var timestamp: TimeInterval
    var pixelBuffer: CVPixelBuffer
    /// fx, fy, cx, cy scaled to the pixel buffer's resolution.
    var intrinsics: SIMD4<Float>
    var cameraToWorld: simd_float4x4
    var trackingState: String
    var exposureDuration: Double?
    var iso: Double?
    var sharpness: Float?
    var depth: DepthSample?
    /// Pose / intrinsics (at `depthImageSize`) of the frame the depth came from — differs
    /// from the colour frame only for high-resolution stills that carry no depth.
    var depthCameraToWorld: simd_float4x4
    var depthIntrinsics: SIMD4<Float>
    var depthImageSize: SIMD2<Float>
}

struct CaptureSummary: Sendable {
    let url: URL
    let name: String
    let frameCount: Int
    let bytes: Int64
    let pointCount: Int
    let duration: TimeInterval
}

enum CaptureError: LocalizedError {
    case noFrames
    case cannotCreate(String)

    var errorDescription: String? {
        switch self {
        case .noFrames: "No keyframes were captured, so nothing was saved. Move the phone slowly around the subject while recording."
        case .cannotCreate(let s): "Could not create the capture folder: \(s)"
        }
    }
}

/// Owns one capture bundle on disk. `submit` is called from the AR delegate queue;
/// all encoding and file IO happens on a private serial queue.
final class CaptureRecorder: @unchecked Sendable {
    let directory: URL
    let name: String
    let settings: CaptureSettings
    let createdAt = Date()
    let hasLiDAR: Bool

    private let queue = DispatchQueue(label: "splatcapture.writer", qos: .userInitiated)
    private let pendingCount = OSAllocatedUnfairLock(initialState: 0)
    private let byteCount = OSAllocatedUnfairLock<Int64>(initialState: 0)
    private let savedCount = OSAllocatedUnfairLock(initialState: 0)
    private let maxPending: Int

    // Writer-queue state
    private let ciContext = CIContext(options: [.cacheIntermediates: false])
    private let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
    private var frames: [ManifestFrame] = []
    private var imageSize: (Int, Int)?
    private var depthSize: (Int, Int)?
    private let fusion: PointFusion?
    private var lastSnapshot: TimeInterval = 0
    private var wroteThumbnail = false
    private var firstTimestamp: TimeInterval?
    private var lastTimestamp: TimeInterval = 0

    /// Delegate-queue only.
    private(set) var submittedCount = 0

    var onPoints: (@Sendable (PointSnapshot) -> Void)?

    init(directory: URL, name: String, settings: CaptureSettings, hasLiDAR: Bool, largeFrames: Bool) throws {
        self.directory = directory
        self.name = name
        self.settings = settings
        self.hasLiDAR = hasLiDAR
        maxPending = largeFrames ? 2 : 4
        fusion = (hasLiDAR && settings.buildPointCloud)
            ? PointFusion(voxelSize: settings.voxelSizeCm / 100,
                          minConfidence: UInt8(settings.minConfidence.rawValue),
                          maxDepth: settings.maxDepth)
            : nil
        let fm = FileManager.default
        do {
            try fm.createDirectory(at: directory.appendingPathComponent("images"), withIntermediateDirectories: true)
            if hasLiDAR {
                try fm.createDirectory(at: directory.appendingPathComponent("depth"), withIntermediateDirectories: true)
                if settings.saveConfidence {
                    try fm.createDirectory(at: directory.appendingPathComponent("confidence"), withIntermediateDirectories: true)
                }
            }
        } catch {
            throw CaptureError.cannotCreate(error.localizedDescription)
        }
    }

    var canAccept: Bool {
        let limit = maxPending
        return pendingCount.withLock { $0 < limit }
    }

    var bytesWritten: Int64 { byteCount.withLock { $0 } }
    var framesSaved: Int { savedCount.withLock { $0 } }

    /// Call from the delegate queue.
    func submit(_ job: KeyframeJob) {
        submittedCount += 1
        pendingCount.withLock { $0 += 1 }
        queue.async { [self] in
            autoreleasepool { process(job) }
            pendingCount.withLock { $0 -= 1 }
        }
    }

    /// Writes points.ply + manifest.json once all queued frames are saved.
    func finish(completion: @escaping @Sendable (Result<CaptureSummary, any Error>) -> Void) {
        queue.async { [self] in
            if frames.isEmpty {
                try? FileManager.default.removeItem(at: directory)
                completion(.failure(CaptureError.noFrames))
                return
            }
            do {
                var points = 0
                if let fusion, fusion.count > 0 {
                    let url = directory.appendingPathComponent("points.ply")
                    try fusion.writePLY(to: url)
                    points = fusion.count
                    addBytes(fileSize(url))
                }
                try writeManifest()
                let summary = CaptureSummary(
                    url: directory, name: name, frameCount: frames.count,
                    bytes: bytesWritten, pointCount: points,
                    duration: lastTimestamp - (firstTimestamp ?? lastTimestamp))
                completion(.success(summary))
            } catch {
                completion(.failure(error))
            }
        }
    }

    func discard() {
        queue.async { [directory] in try? FileManager.default.removeItem(at: directory) }
    }

    // MARK: - Writer queue

    private func process(_ job: KeyframeJob) {
        // Index = position in the manifest, so file numbers stay contiguous even if a frame fails.
        let index = frames.count
        let w = CVPixelBufferGetWidth(job.pixelBuffer), h = CVPixelBufferGetHeight(job.pixelBuffer)
        if let s = imageSize, s != (w, h) { return } // all frames must share a resolution
        let stem = String(format: "%06d", index)
        let imageRel = "images/\(stem).\(settings.imageFormat.fileExtension)"
        let ci = CIImage(cvPixelBuffer: job.pixelBuffer)
        guard let encoded = encode(ci) else { return }
        do {
            try encoded.write(to: directory.appendingPathComponent(imageRel))
        } catch { return }
        addBytes(Int64(encoded.count))
        imageSize = (w, h)

        var depthRel: String?
        var confRel: String?
        if let depth = job.depth, hasLiDAR {
            if depthSize == nil { depthSize = (depth.width, depth.height) }
            if let ds = depthSize, ds == (depth.width, depth.height) {
                let rel = "depth/\(stem).bin"
                let data = depth.float16Data()
                if (try? data.write(to: directory.appendingPathComponent(rel))) != nil {
                    depthRel = rel
                    addBytes(Int64(data.count))
                }
                if settings.saveConfidence, let c = depth.confidenceData() {
                    let crel = "confidence/\(stem).bin"
                    if (try? c.write(to: directory.appendingPathComponent(crel))) != nil {
                        confRel = crel
                        addBytes(Int64(c.count))
                    }
                }
            }
            if let fusion {
                fusion.integrate(depth: depth, intrinsics: job.depthIntrinsics, imageSize: job.depthImageSize,
                                 cameraToWorld: job.depthCameraToWorld, color: job.pixelBuffer)
                let now = ProcessInfo.processInfo.systemUptime
                if now - lastSnapshot > 0.6, let onPoints {
                    lastSnapshot = now
                    onPoints(fusion.snapshot(maxPoints: 160_000))
                }
            }
        }

        if !wroteThumbnail {
            writeThumbnail(ci)
            wroteThumbnail = true
        }

        let i = job.intrinsics
        frames.append(ManifestFrame(
            index: index, timestamp: job.timestamp, image: imageRel,
            depth: depthRel, confidence: confRel,
            intrinsics: [i.x, i.y, i.z, i.w],
            cameraToWorld: job.cameraToWorld.columnMajorArray,
            trackingState: job.trackingState,
            exposureDuration: job.exposureDuration, iso: job.iso,
            sharpness: job.sharpness.map { ($0 * 1000).rounded() / 1000 }))
        if firstTimestamp == nil { firstTimestamp = job.timestamp }
        lastTimestamp = job.timestamp
        savedCount.withLock { $0 += 1 }

        // Periodic manifest so a crash / kill mid-capture still leaves a usable bundle.
        if frames.count % 25 == 0 { try? writeManifest() }
    }

    private func encode(_ image: CIImage) -> Data? {
        let q = CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String)
        let opts: [CIImageRepresentationOption: Any] = [q: settings.imageQuality]
        switch settings.imageFormat {
        case .jpeg:
            return ciContext.jpegRepresentation(of: image, colorSpace: colorSpace, options: opts)
        case .heic:
            return ciContext.heifRepresentation(of: image, format: .RGBA8, colorSpace: colorSpace, options: opts)
        }
    }

    private func writeThumbnail(_ image: CIImage) {
        let scale = 480 / max(image.extent.width, 1)
        // Frames are stored in sensor (landscape) orientation; the UI is portrait.
        let thumb = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale)).oriented(.right)
        let q = CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String)
        if let data = ciContext.jpegRepresentation(of: thumb, colorSpace: colorSpace, options: [q: 0.8]) {
            try? data.write(to: directory.appendingPathComponent("thumbnail.jpg"))
            addBytes(Int64(data.count))
        }
    }

    private func writeManifest() throws {
        let (w, h) = imageSize ?? (0, 0)
        let manifest = CaptureManifest(
            name: name, createdAt: createdAt,
            device: DeviceInfo.modelName, appVersion: DeviceInfo.appVersion,
            hasLiDAR: hasLiDAR && depthSize != nil,
            imageWidth: w, imageHeight: h,
            depthWidth: depthSize?.0, depthHeight: depthSize?.1,
            frames: frames)
        let data = try ManifestCoding.encoder().encode(manifest)
        try data.write(to: directory.appendingPathComponent("manifest.json"), options: .atomic)
    }

    private func addBytes(_ n: Int64) { byteCount.withLock { $0 += n } }

    private func fileSize(_ url: URL) -> Int64 {
        ((try? FileManager.default.attributesOfItem(atPath: url.path)[.size]) as? NSNumber)?.int64Value ?? 0
    }
}
