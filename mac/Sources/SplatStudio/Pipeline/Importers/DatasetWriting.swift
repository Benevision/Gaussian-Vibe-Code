import Foundation
import CoreGraphics
import simd

/// Options shared by all dataset importers (COLMAP, Nerfstudio, capture bundles).
struct ImportOptions: Codable, Hashable, Sendable {
    /// Longest image side written into the dataset (nil = keep). Intrinsics are rescaled accordingly.
    var maxImageSize: Int? = nil
    /// Mark every Nth camera as holdout (evaluation only). nil / ≤1 = no holdout.
    var holdoutInterval: Int? = nil
    /// Undistort images to pinhole when the camera has lens distortion (required for training).
    var undistort: Bool = true
    /// Write a validity mask for undistorted images (black borders are excluded from the loss).
    var maskUndistortionBorders: Bool = true
    /// Re-encode format for images that must be rewritten.
    var encoding: ImageUtil.Encoding = .jpeg

    init() {}
    init(from decoder: Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        maxImageSize = c(.maxImageSize, maxImageSize); holdoutInterval = c(.holdoutInterval, holdoutInterval)
        undistort = c(.undistort, undistort); maskUndistortionBorders = c(.maskUndistortionBorders, maskUndistortionBorders)
        encoding = c(.encoding, encoding)
    }
}

/// Lens distortion models (COLMAP / OpenCV / Nerfstudio / capture-bundle naming).
struct LensDistortion: Codable, Hashable, Sendable {
    enum Kind: String, Codable, Sendable {
        /// No distortion.
        case pinhole
        /// Brown–Conrady: k1, k2, p1, p2, k3 (covers SIMPLE_RADIAL, RADIAL, OPENCV).
        case opencv
        /// Rational: k1…k6, p1, p2 (COLMAP FULL_OPENCV).
        case fullOpencv
        /// Equidistant fisheye: k1…k4 (OPENCV_FISHEYE, RADIAL_FISHEYE, "fisheye"/"equidistant").
        case fisheye
    }
    var kind: Kind = .pinhole
    var k1: Double = 0, k2: Double = 0, k3: Double = 0, k4: Double = 0, k5: Double = 0, k6: Double = 0
    var p1: Double = 0, p2: Double = 0

    static let none = LensDistortion()

    var isIdentity: Bool {
        kind == .pinhole || (kind != .fisheye && k1 == 0 && k2 == 0 && k3 == 0 && k4 == 0 && k5 == 0 && k6 == 0 && p1 == 0 && p2 == 0)
    }

    /// Maps undistorted normalised coordinates to distorted normalised coordinates.
    @inline(__always)
    func distort(_ x: Double, _ y: Double) -> (Double, Double) {
        switch kind {
        case .pinhole: return (x, y)
        case .opencv, .fullOpencv:
            let r2 = x * x + y * y
            var radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
            if kind == .fullOpencv { radial /= (1 + r2 * (k4 + r2 * (k5 + r2 * k6))) }
            let xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
            let yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
            return (xd, yd)
        case .fisheye:
            let r = (x * x + y * y).squareRoot()
            if r < 1e-12 { return (x, y) }
            let th = atan(r), t2 = th * th
            let thd = th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
            let s = thd / r
            return (x * s, y * s)
        }
    }

    /// From an OpenCV-style parameter array `[k1,k2,p1,p2,k3]` (fisheye: `[k1,k2,k3,k4]`).
    init(kind: Kind = .pinhole, params: [Double] = []) {
        self.kind = kind
        func p(_ i: Int) -> Double { i < params.count ? params[i] : 0 }
        switch kind {
        case .pinhole: break
        case .opencv: k1 = p(0); k2 = p(1); p1 = p(2); p2 = p(3); k3 = p(4)
        case .fullOpencv: k1 = p(0); k2 = p(1); p1 = p(2); p2 = p(3); k3 = p(4); k4 = p(5); k5 = p(6); k6 = p(7)
        case .fisheye: k1 = p(0); k2 = p(1); k3 = p(2); k4 = p(3)
        }
    }
}

/// Pinhole intrinsics + distortion of a source image (pixel-centre-at-0.5 convention, like COLMAP).
struct SourceIntrinsics: Hashable, Sendable {
    var width: Int, height: Int
    var fx: Double, fy: Double, cx: Double, cy: Double
    var distortion: LensDistortion = .none
}

/// Writes images into a dataset: copies untouched files when possible, otherwise resizes and/or
/// undistorts (GPU remap) and rescales intrinsics.
enum DatasetImageWriter {
    struct Written: Sendable {
        var image: String
        var width: Int, height: Int
        var fx: Float, fy: Float, cx: Float, cy: Float
        var mask: String?
    }

    /// - Parameters:
    ///   - rel: destination path inside the dataset without extension (e.g. `images/000012`).
    ///   - forceReencode: decode + re-encode even if a plain copy would do (HDR sources → SDR).
    static func write(source: URL, intrinsics k: SourceIntrinsics, datasetRoot: URL, rel: String, options: ImportOptions,
                      forceReencode: Bool = false) throws -> Written {
        let fm = FileManager.default
        let ext = source.pathExtension.lowercased()
        let needsUndistort = options.undistort && !k.distortion.isIdentity
        var scale = 1.0
        if let m = options.maxImageSize, max(k.width, k.height) > m { scale = Double(m) / Double(max(k.width, k.height)) }
        let ow = max(1, Int((Double(k.width) * scale).rounded())), oh = max(1, Int((Double(k.height) * scale).rounded()))
        let sx = Double(ow) / Double(k.width), sy = Double(oh) / Double(k.height)
        var fx = k.fx * sx, fy = k.fy * sy
        let cx = k.cx * sx, cy = k.cy * sy
        try fm.ensureDirectory(datasetRoot.appendingPathComponent(rel).deletingLastPathComponent())

        if !needsUndistort && scale == 1 && !forceReencode && ["jpg", "jpeg", "png"].contains(ext) {
            let dst = "\(rel).\(ext == "jpeg" ? "jpg" : ext)"
            try fm.copyReplacing(source, to: datasetRoot.appendingPathComponent(dst))
            return Written(image: dst, width: k.width, height: k.height, fx: Float(fx), fy: Float(fy), cx: Float(cx), cy: Float(cy), mask: nil)
        }
        let dstExt = options.encoding == .jpeg ? "jpg" : "png"
        let dst = "\(rel).\(dstExt)"
        if !needsUndistort {
            var img = try ImageUtil.load(source, maxSize: nil, applyOrientation: false)
            if img.width != ow || img.height != oh { img = StereoVideoIngest.resize(img, ow, oh) ?? img }
            try ImageUtil.write(img, to: datasetRoot.appendingPathComponent(dst), encoding: options.encoding)
            return Written(image: dst, width: ow, height: oh, fx: Float(fx), fy: Float(fy), cx: Float(cx), cy: Float(cy), mask: nil)
        }
        // Undistort: output pinhole with the same (scaled) K; fisheye gets a wider output FOV.
        if k.distortion.kind == .fisheye { fx *= 0.75; fy *= 0.75 }
        let img = try ImageUtil.load(source, maxSize: nil, applyOrientation: false)
        let table = undistortTable(source: k, srcPixelW: img.width, srcPixelH: img.height, outW: ow, outH: oh,
                                   fx: fx, fy: fy, cx: cx, cy: cy)
        let out = try ImageRemapper.shared.remap(img, table: table)
        try ImageUtil.write(out, to: datasetRoot.appendingPathComponent(dst), encoding: options.encoding)
        var maskRel: String? = nil
        if options.maskUndistortionBorders, let m = table.mask, m.contains(0) {
            maskRel = "masks/" + (rel as NSString).lastPathComponent + ".png"
            try fm.ensureDirectory(datasetRoot.appendingPathComponent("masks"))
            try ImageUtil.writeGray8(m, width: ow, height: oh, to: datasetRoot.appendingPathComponent(maskRel!))
        }
        return Written(image: dst, width: ow, height: oh, fx: Float(fx), fy: Float(fy), cx: Float(cx), cy: Float(cy), mask: maskRel)
    }

    /// Lookup table from undistorted output pixels to distorted source pixels.
    static func undistortTable(source k: SourceIntrinsics, srcPixelW: Int, srcPixelH: Int, outW: Int, outH: Int,
                               fx: Double, fy: Double, cx: Double, cy: Double) -> RemapTable {
        var coords = [SIMD2<Float>](repeating: .zero, count: outW * outH)
        var mask = [UInt8](repeating: 255, count: outW * outH)
        // Source image may have been stored at a different size than the calibration.
        let ssx = Double(srcPixelW) / Double(k.width), ssy = Double(srcPixelH) / Double(k.height)
        coords.withUnsafeMutableBufferPointer { cp in mask.withUnsafeMutableBufferPointer { mp in
            DispatchQueue.concurrentPerform(iterations: outH) { y in
                for x in 0..<outW {
                    let nx = (Double(x) + 0.5 - cx) / fx, ny = (Double(y) + 0.5 - cy) / fy
                    let (dx, dy) = k.distortion.distort(nx, ny)
                    let u = (k.fx * dx + k.cx) * ssx, v = (k.fy * dy + k.cy) * ssy
                    cp[y * outW + x] = SIMD2(Float(u), Float(v))
                    if u < 0 || v < 0 || u > Double(srcPixelW) || v > Double(srcPixelH) { mp[y * outW + x] = 0 }
                }
            }
        } }
        return RemapTable(width: outW, height: outH, coords: coords, wrapX: false, mask: mask)
    }
}

/// Helpers to assemble and finalise a `PreparedDataset`.
enum DatasetAssembly {
    /// Applies holdout, recomputes bounds, saves. Every importer finishes with this.
    static func finalize(root: URL, cameras: [DatasetCamera], points: String?, holdoutInterval: Int?) throws -> PreparedDataset {
        var cams = cameras.sorted { $0.image.localizedStandardCompare($1.image) == .orderedAscending }
        for i in cams.indices {
            cams[i].id = i
            if let n = holdoutInterval, n > 1 { cams[i].holdout = (i % n == 0) ? true : nil }
        }
        var ds = PreparedDataset(root: root, manifest: DatasetManifest(points: points, cameras: cams))
        ds.recomputeSceneBounds()
        try ds.save()
        return ds
    }

    /// Voxel-grid downsampling of a coloured point cloud (average position and colour per voxel).
    static func voxelDownsample(_ pts: [SIMD3<Float>], _ cols: [SIMD3<UInt8>], voxel: Float, maxPoints: Int? = nil) -> PointCloud {
        guard !pts.isEmpty else { return PointCloud() }
        var v = max(voxel, 1e-5)
        while true {
            struct Acc { var p: SIMD3<Float>; var c: SIMD3<Float>; var n: Float }
            var grid: [SIMD3<Int32>: Acc] = [:]
            grid.reserveCapacity(min(pts.count, 2_000_000))
            for i in 0..<pts.count {
                let p = pts[i]
                let key = SIMD3<Int32>(Int32(clamping: Int((p.x / v).rounded(.down))), Int32(clamping: Int((p.y / v).rounded(.down))), Int32(clamping: Int((p.z / v).rounded(.down))))
                let c = i < cols.count ? SIMD3<Float>(Float(cols[i].x), Float(cols[i].y), Float(cols[i].z)) : SIMD3(128, 128, 128)
                if var a = grid[key] { a.p += p; a.c += c; a.n += 1; grid[key] = a } else { grid[key] = Acc(p: p, c: c, n: 1) }
            }
            if let m = maxPoints, grid.count > m { v *= Float(sqrt(Double(grid.count) / Double(m))) * 1.05; continue }
            var pc = PointCloud()
            pc.points.reserveCapacity(grid.count); pc.colors.reserveCapacity(grid.count)
            for a in grid.values {
                pc.points.append(a.p / a.n)
                let c = a.c / a.n
                pc.colors.append(SIMD3(UInt8(min(255, c.x)), UInt8(min(255, c.y)), UInt8(min(255, c.z))))
            }
            return pc
        }
    }
}
