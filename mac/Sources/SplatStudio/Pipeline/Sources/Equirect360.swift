import Foundation
import CoreGraphics
import simd

/// Options for splitting 360° equirectangular footage into pinhole views.
struct Equirect360Options: Codable, Hashable, Sendable {
    /// One ring of views at a fixed pitch.
    struct Ring: Codable, Hashable, Sendable {
        /// Elevation of the view axis in degrees (+ = up, − = down).
        var pitch: Float
        /// Number of views evenly spread over 360° of yaw.
        var count: Int
        /// Yaw of the first view in degrees (stagger rings to improve overlap).
        var yawOffset: Float = 0
    }
    /// Field of view (both axes, square views) in degrees. 90° with 8 horizon views gives ~45° overlap.
    var fovDegrees: Float = 90
    /// Output view size in pixels (square). Should be ≈ equirect width × fov / 360 or less.
    var outputSize: Int = 1200
    /// Rings of views. Default: 8 at the horizon, 4 at ±35° (staggered by 45°).
    var rings: [Ring] = [Ring(pitch: 0, count: 8), Ring(pitch: 35, count: 4, yawOffset: 45), Ring(pitch: -35, count: 4, yawOffset: 45)]
    /// Add a straight-up view (ceiling / sky).
    var includeZenith: Bool = false
    /// Add a straight-down view (usually shows the tripod / operator — off by default).
    var includeNadir: Bool = false
    /// Write masks that drop pixels looking further down than this latitude in degrees
    /// (e.g. −55 removes the tripod / selfie stick). nil = no nadir mask.
    var maskBelowLatitude: Float? = -60
    /// Write masks that drop pixels looking further up than this latitude (sky / ceiling rig). nil = none.
    var maskAboveLatitude: Float? = nil
    /// Global yaw rotation applied to all views (degrees).
    var yawOffset: Float = 0

    init() {}
    init(from decoder: Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        fovDegrees = c(.fovDegrees, fovDegrees); outputSize = c(.outputSize, outputSize); rings = c(.rings, rings)
        includeZenith = c(.includeZenith, includeZenith); includeNadir = c(.includeNadir, includeNadir)
        maskBelowLatitude = c(.maskBelowLatitude, maskBelowLatitude); maskAboveLatitude = c(.maskAboveLatitude, maskAboveLatitude)
        yawOffset = c(.yawOffset, yawOffset)
    }
}

/// Equirectangular → pinhole reprojection with exactly known per-view rotations.
///
/// Rig frame convention (OpenCV-like): +X right, +Y down, +Z = centre of the equirect image.
/// Each view's `rotation` is camera-to-rig (columns = camera axes in rig coordinates), so for a
/// 360 frame with rig pose `T_rig→world`, the view's c2w is `T_rig→world · [R | 0]`.
enum Equirect360 {
    struct View: Hashable, Sendable {
        var index: Int
        var yaw: Float      // degrees
        var pitch: Float    // degrees, + = up
        var rotation: simd_float3x3   // camera → rig
        var size: Int
        var focal: Float

        static func == (a: View, b: View) -> Bool { a.index == b.index && a.yaw == b.yaw && a.pitch == b.pitch && a.size == b.size && a.focal == b.focal }
        func hash(into h: inout Hasher) { h.combine(index); h.combine(yaw); h.combine(pitch); h.combine(size) }
    }

    /// Rotation for a view with given yaw (about rig +Y-down axis, positive = turn right) and pitch (+ = up).
    static func rotation(yawDegrees: Float, pitchDegrees: Float) -> simd_float3x3 {
        let y = yawDegrees * .pi / 180, p = pitchDegrees * .pi / 180
        let ry = simd_float3x3(rows: [SIMD3(cos(y), 0, sin(y)), SIMD3(0, 1, 0), SIMD3(-sin(y), 0, cos(y))])
        let rx = simd_float3x3(rows: [SIMD3(1, 0, 0), SIMD3(0, cos(p), -sin(p)), SIMD3(0, sin(p), cos(p))])
        return ry * rx
    }

    static func views(_ o: Equirect360Options) -> [View] {
        let f = Float(o.outputSize) / 2 / tan(o.fovDegrees * .pi / 360)
        var out: [View] = []
        for ring in o.rings where ring.count > 0 {
            for k in 0..<ring.count {
                let yaw = o.yawOffset + ring.yawOffset + Float(k) * 360 / Float(ring.count)
                out.append(View(index: out.count, yaw: yaw, pitch: ring.pitch, rotation: rotation(yawDegrees: yaw, pitchDegrees: ring.pitch), size: o.outputSize, focal: f))
            }
        }
        if o.includeZenith { out.append(View(index: out.count, yaw: o.yawOffset, pitch: 90, rotation: rotation(yawDegrees: o.yawOffset, pitchDegrees: 90), size: o.outputSize, focal: f)) }
        if o.includeNadir { out.append(View(index: out.count, yaw: o.yawOffset, pitch: -90, rotation: rotation(yawDegrees: o.yawOffset, pitchDegrees: -90), size: o.outputSize, focal: f)) }
        return out
    }

    /// Lookup table from view pixels to equirect pixels (+ latitude mask).
    static func table(for v: View, sourceWidth W: Int, sourceHeight H: Int, options o: Equirect360Options) -> RemapTable {
        let n = v.size
        var coords = [SIMD2<Float>](repeating: .zero, count: n * n)
        let wantMask = o.maskBelowLatitude != nil || o.maskAboveLatitude != nil
        var mask = [UInt8](repeating: 255, count: wantMask ? n * n : 0)
        let c = Float(n) / 2, R = v.rotation
        let lo = (o.maskBelowLatitude ?? -91) * .pi / 180, hi = (o.maskAboveLatitude ?? 91) * .pi / 180
        coords.withUnsafeMutableBufferPointer { cp in
            mask.withUnsafeMutableBufferPointer { mp in
                DispatchQueue.concurrentPerform(iterations: n) { y in
                    for x in 0..<n {
                        let d = simd_normalize(R * SIMD3((Float(x) + 0.5 - c) / v.focal, (Float(y) + 0.5 - c) / v.focal, 1))
                        let lon = atan2(d.x, d.z)
                        let lat = asin(max(-1, min(1, -d.y)))
                        cp[y * n + x] = SIMD2((lon / (2 * .pi) + 0.5) * Float(W), (0.5 - lat / .pi) * Float(H))
                        if wantMask && (lat < lo || lat > hi) { mp[y * n + x] = 0 }
                    }
                }
            }
        }
        return RemapTable(width: n, height: n, coords: coords, wrapX: true, mask: wantMask ? mask : nil)
    }

    /// Caches tables per (view, source size).
    final class TableCache: @unchecked Sendable {
        private var cache: [String: RemapTable] = [:]
        private let lock = NSLock()
        let options: Equirect360Options
        init(options: Equirect360Options) { self.options = options }
        func table(_ v: View, _ w: Int, _ h: Int) -> RemapTable {
            let key = "\(v.index)-\(w)x\(h)"
            lock.lock(); if let t = cache[key] { lock.unlock(); return t }; lock.unlock()
            let t = Equirect360.table(for: v, sourceWidth: w, sourceHeight: h, options: options)
            lock.lock(); cache[key] = t; lock.unlock()
            return t
        }
    }

    /// Writes all views of one equirect image. Returns the frames (group = `frameIndex`).
    static func writeViews(of image: CGImage, frameIndex: Int, views: [View], cache: TableCache, root: URL,
                           encoding: ImageUtil.Encoding, source: String?, timestamp: Double?, sharpness: Double) throws -> [IngestedFrame] {
        let (px, w, h) = ImageUtil.rgba8(image)
        let ext = encoding == .jpeg ? "jpg" : "png"
        let fm = FileManager.default
        try fm.ensureDirectory(root.appendingPathComponent("images"))
        var out: [IngestedFrame] = []
        for v in views {
            let t = cache.table(v, w, h)
            let rgba = ImageRemapper.shared.remap(rgba: px, width: w, height: h, table: t)
            guard let img = ImageUtil.cgImage(rgba: rgba, width: t.width, height: t.height) else { throw PipelineError.encodeFailed("360 view") }
            let name = String(format: "%06d_v%02d", frameIndex, v.index)
            try ImageUtil.write(img, to: root.appendingPathComponent("images/\(name).\(ext)"), encoding: encoding)
            var f = IngestedFrame(image: "images/\(name).\(ext)", width: t.width, height: t.height, sharpness: sharpness, source: source, timestamp: timestamp)
            f.focalPx = v.focal; f.cx = Float(v.size) / 2; f.cy = Float(v.size) / 2
            f.group = frameIndex
            f.rotationToRig = v.rotation.rowMajor
            if let m = t.mask {
                try fm.ensureDirectory(root.appendingPathComponent("masks"))
                try ImageUtil.writeGray8(m, width: t.width, height: t.height, to: root.appendingPathComponent("masks/\(name).png"))
                f.mask = "masks/\(name).png"
            }
            out.append(f)
        }
        return out
    }

    /// 360° video → views. Frames are selected with `VideoFrameSelector` (sharpest per window).
    static func extract(video url: URL, video vo: VideoExtractionOptions, options o: Equirect360Options, into root: URL,
                        progress: PipelineProgressHandler? = nil) async throws -> FrameSet {
        let stage = "Splitting 360° video"
        let reader = try await VideoFrameReader(url: url, start: vo.startTime, end: vo.endTime)
        defer { reader.cancel() }
        let views = views(o), cache = TableCache(options: o)
        let hdr = reader.isHDR && vo.toneMapHDR
        progress?(PipelineProgress(stage: stage, fraction: 0, message: "\(views.count) views per frame, GPU: \(ImageRemapper.shared.usesGPU)"))
        let (frames, rejected) = try await VideoFrameSelector.run(reader: reader, options: vo, stage: stage, progress: progress) { s, idx, sharp in
            let img = try VideoFrameReader.render(s.buffers[0], orientation: reader.orientation, hdr: hdr, maxSize: nil)
            return try writeViews(of: img, frameIndex: idx, views: views, cache: cache, root: root, encoding: vo.encoding,
                                  source: url.path, timestamp: s.time, sharpness: sharp)
        }
        guard !frames.isEmpty else { throw PipelineError.noInput("no frames extracted") }
        let set = FrameSet(root: root, frames: frames, rejected: rejected, rig: nil, isMultiViewRig: true)
        try set.save()
        progress?(PipelineProgress(stage: stage, fraction: 1, message: "\(frames.count) views"))
        return set
    }

    /// 360° photos (equirect JPEG/HEIC/…) → views.
    static func extract(photos: [URL], options o: Equirect360Options, encoding: ImageUtil.Encoding = .jpeg, into root: URL,
                        progress: PipelineProgressHandler? = nil) async throws -> FrameSet {
        guard !photos.isEmpty else { throw PipelineError.noInput("no 360° photos") }
        let stage = "Splitting 360° photos"
        let views = views(o), cache = TableCache(options: o)
        var set = FrameSet(root: root, isMultiViewRig: true)
        for (i, url) in photos.enumerated() {
            try Task.checkCancellation()
            do {
                let img = try ImageUtil.load(url, maxSize: 16384)
                guard abs(Double(img.width) / Double(img.height) - 2) < 0.1 else {
                    set.rejected.append(.init(item: url.lastPathComponent, reason: "not 2:1 equirectangular")); continue
                }
                let sharp = ImageMetrics.sharpness(img)
                set.frames += try writeViews(of: img, frameIndex: i, views: views, cache: cache, root: root, encoding: encoding,
                                             source: url.path, timestamp: nil, sharpness: sharp)
            } catch let e as CancellationError { throw e } catch {
                set.rejected.append(.init(item: url.lastPathComponent, reason: error.localizedDescription))
            }
            progress?(PipelineProgress(stage: stage, fraction: Double(i + 1) / Double(photos.count), message: url.lastPathComponent))
        }
        guard !set.frames.isEmpty else { throw PipelineError.noInput("no usable 360° photos") }
        try set.save()
        return set
    }
}
