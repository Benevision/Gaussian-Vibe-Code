import Foundation
import AVFoundation
import CoreGraphics
import CoreVideo

/// How the two eyes are stored.
enum StereoLayout: String, Codable, CaseIterable, Sendable {
    /// Detect from MV-HEVC flags, track count, aspect ratio and metadata.
    case auto
    /// Left | right in one frame (full-width or half-width/anamorphic).
    case sideBySide
    /// Left above right in one frame.
    case topBottom
    /// Two separate video files (left, right).
    case separateFiles
    /// Apple MV-HEVC spatial video (two layers in one track).
    case mvhevc
    /// Two video tracks in one file (e.g. some Arcturus Vision recordings).
    case dualStream
}

/// Camera / device family. Only used for detection heuristics and defaults.
enum StereoDevice: String, Codable, CaseIterable, Sendable {
    case auto
    case generic
    /// Apple spatial video (iPhone 15 Pro and later, Vision Pro).
    case appleSpatial
    /// Arcturus Vision stereo recordings — assumed SBS or dual-stream (detected; user-overridable).
    case arcturus
}

/// Options for stereo footage.
struct StereoOptions: Codable, Hashable, Sendable {
    var layout: StereoLayout = .auto
    var device: StereoDevice = .auto
    /// Swap left and right (some cameras store right first).
    var swapEyes: Bool = false
    /// Half-width SBS / half-height TB (each eye squeezed ×2); nil = auto (from aspect ratio).
    var anamorphic: Bool? = nil
    /// Rig baseline in metres; overrides metadata/calibration. Enables metric scale.
    var baseline: Float? = nil
    /// Horizontal field of view of one eye in degrees (used when no calibration/metadata intrinsics).
    var horizontalFOV: Float? = nil
    /// Optional calibration file (see `StereoRig.loadCalibration`).
    var calibrationFile: URL? = nil
    /// Time offset of the right video relative to the left (separate files), seconds.
    var rightTimeOffset: Double = 0
    /// Also use right-eye images as training cameras (doubles coverage; recommended).
    var trainOnBothEyes: Bool = true
    /// Compute stereo depth for the left eye (requires known baseline and focal length).
    var computeDepth: Bool = false
    var depth: StereoDepthOptions = .init()

    init() {}
    init(from decoder: Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        layout = c(.layout, layout); device = c(.device, device); swapEyes = c(.swapEyes, swapEyes); anamorphic = c(.anamorphic, anamorphic)
        baseline = c(.baseline, baseline); horizontalFOV = c(.horizontalFOV, horizontalFOV); calibrationFile = c(.calibrationFile, calibrationFile)
        rightTimeOffset = c(.rightTimeOffset, rightTimeOffset); trainOnBothEyes = c(.trainOnBothEyes, trainOnBothEyes)
        computeDepth = c(.computeDepth, computeDepth); depth = c(.depth, depth)
    }
}

/// Result of stereo format detection.
struct StereoDetection: Sendable, Hashable {
    var layout: StereoLayout
    var device: StereoDevice
    var anamorphic: Bool
    var reason: String
}

/// Stereo video → left/right image pairs (+ rig, + optional left-eye depth).
enum StereoVideoIngest {

    /// Inspects a file and guesses layout/device. User options that are not `.auto` always win.
    static func detect(_ url: URL, options: StereoOptions) async throws -> StereoDetection {
        let asset = AVURLAsset(url: url)
        let tracks = try await asset.loadTracks(withMediaType: .video)
        guard let t0 = tracks.first else { throw PipelineError.noInput("no video track") }
        let (size, chars, transform) = try await t0.load(.naturalSize, .mediaCharacteristics, .preferredTransform)
        let o = VideoFrameReader.orientation(from: transform)
        let disp = (o == .left || o == .right) ? CGSize(width: size.height, height: size.width) : size
        let aspect = disp.width / max(disp.height, 1)
        var meta = url.lastPathComponent.lowercased()
        if let md = try? await asset.load(.metadata) {
            for item in md { if let s = try? await item.load(.stringValue) { meta += " " + s.lowercased() } }
        }
        var device = options.device
        if device == .auto {
            if chars.contains(.containsStereoMultiviewVideo) { device = .appleSpatial }
            else if meta.contains("arcturus") { device = .arcturus }
            else { device = .generic }
        }
        if options.layout != .auto {
            return StereoDetection(layout: options.layout, device: device, anamorphic: options.anamorphic ?? false, reason: "user override")
        }
        if chars.contains(.containsStereoMultiviewVideo) {
            return StereoDetection(layout: .mvhevc, device: device, anamorphic: false, reason: "MV-HEVC stereo multiview track")
        }
        if tracks.count >= 2 {
            let s1 = try await tracks[1].load(.naturalSize)
            if abs(s1.width - size.width) < 2 && abs(s1.height - size.height) < 2 {
                return StereoDetection(layout: .dualStream, device: device, anamorphic: false, reason: "two video tracks of equal size")
            }
        }
        if meta.contains("_tb") || meta.contains("top-bottom") || meta.contains("topbottom") || meta.contains("over-under") || aspect < 1.05 {
            // e.g. 1920×2160 = two 16:9 eyes stacked, or a square frame of two 2:1 eyes.
            return StereoDetection(layout: .topBottom, device: device, anamorphic: options.anamorphic ?? (aspect >= 1.5),
                                   reason: String(format: "aspect %.2f → top-bottom", aspect))
        }
        if aspect >= 2.6 || meta.contains("_sbs") || meta.contains("side-by-side") || device == .arcturus {
            // Full-width SBS of two 16:9 / 4:3 eyes (aspect 3.56 / 2.67), or Arcturus (assumed SBS).
            let ana = options.anamorphic ?? (aspect < 2.6)   // a 16:9 frame flagged as SBS is half-width
            return StereoDetection(layout: .sideBySide, device: device, anamorphic: ana,
                                   reason: device == .arcturus ? "Arcturus Vision metadata, assuming SBS" : String(format: "aspect %.2f → side-by-side", aspect))
        }
        if abs(aspect - 2) < 0.05 {
            return StereoDetection(layout: .sideBySide, device: device, anamorphic: false, reason: "2:1 frame of two square eyes (assumed SBS)")
        }
        throw PipelineError.unsupported(String(format: "cannot detect stereo layout (aspect %.2f) — choose it manually", aspect))
    }

    /// Ingests stereo footage. `rightURL` is used for `.separateFiles`.
    static func ingest(left url: URL, right rightURL: URL? = nil, video vo: VideoExtractionOptions, stereo so: StereoOptions,
                       into root: URL, progress: PipelineProgressHandler? = nil) async throws -> FrameSet {
        let stage = "Extracting stereo frames"
        var so = so
        if rightURL != nil && so.layout == .auto { so.layout = .separateFiles }
        let det = try await detect(url, options: so)
        progress?(PipelineProgress(stage: stage, fraction: 0, message: "\(det.layout.rawValue) (\(det.reason))"))
        let fm = FileManager.default
        // Flat naming (images/000012_L.jpg / _R.jpg) so that tools matching by basename never collide.
        try fm.ensureDirectory(root.appendingPathComponent("images"))
        let ext = vo.encoding == .jpeg ? "jpg" : "png"

        let reader = try await VideoFrameReader(url: url, trackIndex: 0, decodeStereoLayers: det.layout == .mvhevc, start: vo.startTime, end: vo.endTime)
        defer { reader.cancel() }
        var secondary: VideoFrameReader? = nil
        switch det.layout {
        case .separateFiles:
            guard let rightURL else { throw PipelineError.noInput("right-eye video missing") }
            secondary = try await VideoFrameReader(url: rightURL, start: vo.startTime.map { $0 + so.rightTimeOffset }, end: vo.endTime.map { $0 + so.rightTimeOffset })
        case .dualStream:
            secondary = try await VideoFrameReader(url: url, trackIndex: 1, start: vo.startTime, end: vo.endTime)
        default: break
        }
        defer { secondary?.cancel() }

        // Rig: user > calibration file > MV-HEVC metadata > none.
        var rig = StereoRig(baseline: nil)
        if let cal = so.calibrationFile { rig = try StereoRig.loadCalibration(cal) }
        if det.layout == .mvhevc {
            let md = reader.stereoMetadata
            if rig.baseline == nil, let b = md.baseline { rig.baseline = b; rig.origin = "mvhevc-metadata" }
            if rig.horizontalFOV == nil { rig.horizontalFOV = md.horizontalFOV }
        }
        if let b = so.baseline { rig.baseline = b; rig.origin = "user" }
        if let f = so.horizontalFOV { rig.horizontalFOV = f }

        // Eye crop rects (oriented pixels) for packed layouts.
        let ds = reader.displaySize
        var leftRect: CGRect? = nil, rightRect: CGRect? = nil
        if det.layout == .sideBySide {
            leftRect = CGRect(x: 0, y: 0, width: ds.width / 2, height: ds.height)
            rightRect = CGRect(x: ds.width / 2, y: 0, width: ds.width / 2, height: ds.height)
        } else if det.layout == .topBottom {
            leftRect = CGRect(x: 0, y: 0, width: ds.width, height: ds.height / 2)
            rightRect = CGRect(x: 0, y: ds.height / 2, width: ds.width, height: ds.height / 2)
        }
        if so.swapEyes { swap(&leftRect, &rightRect) }
        let sharpCrop: CGRect? = (reader.orientation == .up) ? leftRect : nil
        let hdr = reader.isHDR && vo.toneMapHDR
        let anamorphic = det.anamorphic
        let maxSize = vo.maxImageSize
        var secondaryPending: DecodedVideoSample? = nil

        func renderEye(_ pb: CVPixelBuffer, crop: CGRect?, orientation: CGImagePropertyOrientation, hdr: Bool) throws -> CGImage {
            var img = try VideoFrameReader.render(pb, orientation: orientation, hdr: hdr, crop: crop, maxSize: nil)
            if anamorphic, let crop {
                // Un-squeeze half-width SBS / half-height TB eyes.
                let w = det.layout == .sideBySide ? Int(crop.width * 2) : Int(crop.width)
                let h = det.layout == .topBottom ? Int(crop.height * 2) : Int(crop.height)
                img = resize(img, w, h) ?? img
            }
            return ImageUtil.downscaled(img, maxSize: maxSize)
        }

        /// Closest secondary-stream frame to time t.
        func secondaryFrame(at t: Double) throws -> DecodedVideoSample? {
            guard let sec = secondary else { return nil }
            let target = t + (det.layout == .separateFiles ? so.rightTimeOffset : 0)
            var prev: DecodedVideoSample? = nil
            while secondaryPending == nil || secondaryPending!.time < target {
                prev = secondaryPending
                guard let s = try sec.next() else { secondaryPending = nil; return prev }
                secondaryPending = s
            }
            if let p = prev, abs(p.time - target) < abs(secondaryPending!.time - target) { return p }
            return secondaryPending
        }

        var depthCount = 0
        let (frames, rejected) = try await VideoFrameSelector.run(reader: reader, options: vo, stage: stage, sharpnessCrop: sharpCrop, progress: progress) { s, idx, sharp in
            var limg: CGImage, rimg: CGImage
            switch det.layout {
            case .mvhevc:
                guard s.buffers.count >= 2 else { throw PipelineError.decodeFailed("MV-HEVC frame without two layers") }
                let a = so.swapEyes ? 1 : 0
                limg = try renderEye(s.buffers[a], crop: nil, orientation: reader.orientation, hdr: hdr)
                rimg = try renderEye(s.buffers[1 - a], crop: nil, orientation: reader.orientation, hdr: hdr)
            case .separateFiles, .dualStream:
                guard let rs = try secondaryFrame(at: s.time) else { return [] }
                limg = try renderEye(s.buffers[0], crop: nil, orientation: reader.orientation, hdr: hdr)
                rimg = try renderEye(rs.buffers[0], crop: nil, orientation: secondary!.orientation, hdr: secondary!.isHDR && vo.toneMapHDR)
                if so.swapEyes { swap(&limg, &rimg) }
            default:
                limg = try renderEye(s.buffers[0], crop: leftRect, orientation: reader.orientation, hdr: hdr)
                rimg = try renderEye(s.buffers[0], crop: rightRect, orientation: reader.orientation, hdr: hdr)
            }
            let lname = String(format: "images/%06d_L.%@", idx, ext), rname = String(format: "images/%06d_R.%@", idx, ext)
            try ImageUtil.write(limg, to: root.appendingPathComponent(lname), encoding: vo.encoding)
            try ImageUtil.write(rimg, to: root.appendingPathComponent(rname), encoding: vo.encoding)
            let focalL = focal(rig: rig, eye: 0, width: limg.width, height: limg.height, f35: reader.focal35mm)
            let focalR = focal(rig: rig, eye: 1, width: rimg.width, height: rimg.height, f35: reader.focal35mm)
            var lf = IngestedFrame(image: lname, width: limg.width, height: limg.height, sharpness: sharp, source: url.path, timestamp: s.time)
            lf.focalPx = focalL?.0; lf.cx = focalL?.1; lf.cy = focalL?.2; lf.group = idx; lf.eye = "left"
            var rf = IngestedFrame(image: rname, width: rimg.width, height: rimg.height, sharpness: sharp, source: (rightURL ?? url).path, timestamp: s.time)
            rf.focalPx = focalR?.0; rf.cx = focalR?.1; rf.cy = focalR?.2; rf.group = idx; rf.eye = "right"
            if so.computeDepth, let b = rig.baseline, let f = focalL?.0 {
                let d = StereoDepth.depth(left: limg, right: rimg, focal: f, baseline: b, options: so.depth)
                try fm.ensureDirectory(root.appendingPathComponent("depth"))
                let dn = String(format: "depth/%06d.bin", idx)
                try PreparedDataset.writeDepth(d.depth, to: root.appendingPathComponent(dn))
                lf.depth = dn; lf.depthWidth = d.width; lf.depthHeight = d.height
                depthCount += 1
            }
            return [lf, rf]
        }
        guard !frames.isEmpty else { throw PipelineError.noInput("no stereo frames extracted") }
        var set = FrameSet(root: root, frames: frames, rejected: rejected, rig: rig)
        if so.computeDepth && depthCount == 0 {
            set.rejected.append(.init(item: "depth", reason: "stereo depth skipped: baseline or focal length unknown"))
        }
        try set.save()
        progress?(PipelineProgress(stage: stage, fraction: 1, message: "\(frames.count / 2) pairs"))
        return set
    }

    /// Focal length in pixels (+ principal point) of an eye image.
    static func focal(rig: StereoRig, eye: Int, width: Int, height: Int, f35: Double?) -> (Float, Float, Float)? {
        if let k = eye == 0 ? rig.leftIntrinsics : (rig.rightIntrinsics ?? rig.leftIntrinsics), k.count >= 4 {
            // Intrinsics are given for the calibration resolution ≈ 2·cx; rescale to the written size.
            let calibW = max(k[2] * 2, 1)
            let s = Float(width) / calibW
            return (k[0] * s, k[2] * s, k[3] * s)
        }
        if let fov = rig.horizontalFOV, fov > 1 {
            return (Float(width) / 2 / tan(fov * .pi / 360), Float(width) / 2, Float(height) / 2)
        }
        if let f35 { return (Float(f35 / 36) * Float(max(width, height)), Float(width) / 2, Float(height) / 2) }
        return nil
    }

    static func resize(_ img: CGImage, _ w: Int, _ h: Int) -> CGImage? {
        let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0, space: ImageUtil.sRGB,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        ctx?.interpolationQuality = .high
        ctx?.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
        return ctx?.makeImage()
    }
}
