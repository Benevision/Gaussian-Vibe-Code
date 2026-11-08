import Foundation
import simd

/// `manifest.json` of a `.splatcapture` bundle, v1 and v2 (PROTOCOL.md §1, §8). All fields optional/tolerant.
struct CaptureManifest: Codable, Sendable {
    struct Distortion: Codable, Sendable { var model: String?; var params: [Double]? }
    struct Sensor: Codable, Sendable {
        var id: String
        var kind: String?
        var width: Int?
        var height: Int?
        var intrinsics: [Float]?
        var distortion: Distortion?
        /// 16 floats, column-major, in the manifest's camera convention.
        var sensorToRig: [Float]?
    }
    struct Rig: Codable, Sendable { var sensors: [Sensor]?; var baseline: Float? }
    struct Frame: Codable, Sendable {
        var index: Int?
        var timestamp: Double?
        var image: String?
        var images: [String: String]?
        var depth: String?
        var confidence: String?
        var depthWidth: Int?
        var depthHeight: Int?
        var depthSource: String?
        var intrinsics: [Float]?
        var cameraToWorld: [Float]?
        var rigToWorld: [Float]?
        var trackingState: String?
        var exposureDuration: Double?
        var iso: Double?
        var sharpness: Double?
    }
    var format: String?
    var version: Int?
    var name: String?
    var createdAt: String?
    var device: String?
    var appVersion: String?
    var source: String?
    var hasLiDAR: Bool?
    var coordinateSystem: String?
    var imageWidth: Int?
    var imageHeight: Int?
    var depthWidth: Int?
    var depthHeight: Int?
    var colorEncoding: String?
    var rig: Rig?
    var frames: [Frame]

    static func load(_ bundle: URL) throws -> CaptureManifest {
        try JSONDecoder().decode(CaptureManifest.self, from: Data(contentsOf: bundle.appendingPathComponent("manifest.json")))
    }
}

/// Options for importing a capture bundle.
struct CaptureImportOptions: Codable, Hashable, Sendable {
    /// Image size / holdout / undistortion settings.
    var images: ImportOptions = .init()
    /// Use every Nth frame (1 = all).
    var frameStride: Int = 1
    /// Skip frames whose ARKit tracking state is not "normal".
    var dropLimitedTracking: Bool = true
    /// Skip frames whose capture-time sharpness (0…1) is below this value.
    var minSharpness: Double? = nil
    /// Include right-eye images of stereo (v2) captures as extra training cameras.
    var includeRightEye: Bool = true
    /// Copy depth maps into the dataset for depth supervision.
    var copyDepth: Bool = true
    /// Depth pixels with confidence below this (0 low, 1 medium, 2 high) are zeroed in copied depth maps.
    var depthMinConfidence: Int = 1
    /// Build `points3d.ply` from depth when the bundle has no `points.ply`.
    var buildPointsFromDepth: Bool = true
    /// Minimum confidence of depth pixels used for the point cloud.
    var pointsMinConfidence: Int = 2
    /// Voxel size for point-cloud downsampling, metres.
    var pointsVoxelSize: Float = 0.01
    /// Upper bound on the number of initial points (voxel grows to fit).
    var pointsMaxCount: Int = 500_000
    /// Unproject every Nth kept frame.
    var pointsFrameStride: Int = 2
    /// Ignore depth beyond this distance (LiDAR is unreliable past ~5 m), metres.
    var maxDepth: Float = 6

    init() {}
    init(from decoder: Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        images = c(.images, images); frameStride = c(.frameStride, frameStride); dropLimitedTracking = c(.dropLimitedTracking, dropLimitedTracking)
        minSharpness = c(.minSharpness, minSharpness); includeRightEye = c(.includeRightEye, includeRightEye); copyDepth = c(.copyDepth, copyDepth)
        depthMinConfidence = c(.depthMinConfidence, depthMinConfidence); buildPointsFromDepth = c(.buildPointsFromDepth, buildPointsFromDepth)
        pointsMinConfidence = c(.pointsMinConfidence, pointsMinConfidence); pointsVoxelSize = c(.pointsVoxelSize, pointsVoxelSize)
        pointsMaxCount = c(.pointsMaxCount, pointsMaxCount); pointsFrameStride = c(.pointsFrameStride, pointsFrameStride); maxDepth = c(.maxDepth, maxDepth)
    }
}

/// Imports `.splatcapture` bundles (directory or zip) into a `PreparedDataset`.
enum CaptureImporter {
    /// Resolves a bundle directory from a directory or a zip (unzipped next to `scratch`).
    static func resolveBundle(_ url: URL, scratch: URL) async throws -> URL {
        let fm = FileManager.default
        var dir = url
        if url.pathExtension.lowercased() == "zip" {
            dir = scratch.appendingPathComponent("unzipped-\(UUID().uuidString.prefix(8))")
            try await ZipUtil.unzip(url, to: dir)
        }
        if fm.fileExists(atPath: dir.appendingPathComponent("manifest.json").path) { return dir }
        let subs = (try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
        if let b = subs.first(where: { fm.fileExists(atPath: $0.appendingPathComponent("manifest.json").path) }) { return b }
        throw PipelineError.noInput("no manifest.json in \(url.lastPathComponent)")
    }

    /// Converts a pose from the manifest's coordinate system to OpenCV camera-to-world.
    static func toOpenCV(_ m: simd_float4x4, system: String?) -> simd_float4x4 {
        switch (system ?? "arkit").lowercased() {
        case "opencv": return m
        default: return m * .flipYZ   // arkit / openxr / opengl: camera looks −Z, +Y up
        }
    }

    static func distortion(_ d: CaptureManifest.Distortion?) -> LensDistortion {
        guard let d, let model = d.model?.lowercased() else { return .none }
        switch model {
        case "opencv": return LensDistortion(kind: .opencv, params: d.params ?? [])
        case "fisheye", "equidistant": return LensDistortion(kind: .fisheye, params: d.params ?? [])
        default: return .none
        }
    }

    static func importBundle(_ url: URL, options: CaptureImportOptions = .init(), into out: URL,
                             progress: PipelineProgressHandler? = nil) async throws -> PreparedDataset {
        let stage = "Importing capture"
        let fm = FileManager.default
        try fm.ensureDirectory(out)
        let scratch = try fm.ensureDirectory(out.appendingPathComponent(".import-tmp"))
        defer { try? fm.removeItem(at: scratch) }
        let bundle = try await resolveBundle(url, scratch: scratch)
        let m = try CaptureManifest.load(bundle)
        guard !m.frames.isEmpty else { throw PipelineError.noInput("capture has no frames") }
        let system = m.coordinateSystem
        let sensors = Dictionary((m.rig?.sensors ?? []).map { ($0.id, $0) }, uniquingKeysWith: { a, _ in a })
        let leftSensor = sensors["left"] ?? m.rig?.sensors?.first(where: { ($0.kind ?? "rgb") == "rgb" })
        let rightSensor = sensors["right"]
        let forceReencode = (m.colorEncoding ?? "srgb").lowercased() != "srgb"
        progress?(PipelineProgress(stage: stage, fraction: 0, message: "\(m.name ?? bundle.lastPathComponent): \(m.frames.count) frames"))

        // Frame selection.
        var kept: [CaptureManifest.Frame] = []
        for (i, f) in m.frames.enumerated() {
            if i % max(1, options.frameStride) != 0 { continue }
            if options.dropLimitedTracking, let t = f.trackingState, t != "normal" { continue }
            if let ms = options.minSharpness, let s = f.sharpness, s < ms { continue }
            guard f.image != nil || f.images?["left"] != nil, f.cameraToWorld != nil || f.rigToWorld != nil else { continue }
            kept.append(f)
        }
        guard !kept.isEmpty else { throw PipelineError.noInput("no usable frames (tracking/sharpness filters)") }

        struct Job: Sendable {
            var src: URL; var k: SourceIntrinsics; var c2w: simd_float4x4; var rel: String
            var depth: URL?; var conf: URL?; var dw: Int; var dh: Int
            var exposure: Double? = nil
        }
        var jobs: [Job] = []
        for (n, f) in kept.enumerated() {
            let idx = f.index ?? n
            let leftRel = f.images?["left"] ?? f.image!
            let leftSrc = bundle.appendingPathComponent(leftRel)
            let size = ImageUtil.pixelSizeRaw(leftSrc)
            let iw = leftSensor?.width ?? m.imageWidth ?? size?.0 ?? 0, ih = leftSensor?.height ?? m.imageHeight ?? size?.1 ?? 0
            guard iw > 0, ih > 0 else { continue }
            let intr = f.intrinsics ?? leftSensor?.intrinsics ?? [ImageUtil.defaultFocal(width: iw, height: ih), ImageUtil.defaultFocal(width: iw, height: ih), Float(iw) / 2, Float(ih) / 2]
            let kL = SourceIntrinsics(width: iw, height: ih, fx: Double(intr[0]), fy: Double(intr[1]), cx: Double(intr[2]), cy: Double(intr[3]),
                                      distortion: distortion(leftSensor?.distortion))
            let rigToWorld: simd_float4x4? = f.rigToWorld.map { simd_float4x4(columnMajor: $0) }
            let leftC2W: simd_float4x4 = {
                if let c = f.cameraToWorld { return simd_float4x4(columnMajor: c) }
                let s2r = leftSensor?.sensorToRig.map { simd_float4x4(columnMajor: $0) } ?? matrix_identity_float4x4
                return rigToWorld! * s2r
            }()
            let dw = f.depthWidth ?? m.depthWidth ?? 0, dh = f.depthHeight ?? m.depthHeight ?? 0
            jobs.append(Job(src: leftSrc, k: kL, c2w: toOpenCV(leftC2W, system: system), rel: String(format: "images/%06d", idx),
                            depth: f.depth.map { bundle.appendingPathComponent($0) }, conf: f.confidence.map { bundle.appendingPathComponent($0) },
                            dw: dw, dh: dh,
                            exposure: f.exposureDuration.flatMap { d in d > 0 ? d * (f.iso ?? 100) / 100 : nil }))
            if options.includeRightEye, let rr = f.images?["right"], let rs = rightSensor {
                let rSrc = bundle.appendingPathComponent(rr)
                let rsize = ImageUtil.pixelSizeRaw(rSrc)
                let rw = rs.width ?? rsize?.0 ?? iw, rh = rs.height ?? rsize?.1 ?? ih
                let ri = rs.intrinsics ?? intr
                let kR = SourceIntrinsics(width: rw, height: rh, fx: Double(ri[0]), fy: Double(ri[1]), cx: Double(ri[2]), cy: Double(ri[3]),
                                          distortion: distortion(rs.distortion))
                let r2w: simd_float4x4
                if let rig = rigToWorld, let s2r = rs.sensorToRig {
                    r2w = rig * simd_float4x4(columnMajor: s2r)
                } else if let l2r = leftSensor?.sensorToRig, let s2r = rs.sensorToRig {
                    r2w = leftC2W * simd_inverse(simd_float4x4(columnMajor: l2r)) * simd_float4x4(columnMajor: s2r)
                } else if let b = m.rig?.baseline {
                    var t = matrix_identity_float4x4; t.columns.3 = SIMD4(b, 0, 0, 1)   // +X of the left camera
                    r2w = leftC2W * t
                } else { continue }
                jobs.append(Job(src: rSrc, k: kR, c2w: toOpenCV(r2w, system: system), rel: String(format: "images/%06d_right", idx),
                                depth: nil, conf: nil, dw: 0, dh: 0))
            }
        }
        guard !jobs.isEmpty else { throw PipelineError.noInput("no frames with valid images") }

        try fm.ensureDirectory(out.appendingPathComponent("images"))
        if options.copyDepth { try fm.ensureDirectory(out.appendingPathComponent("depth")) }
        let counter = ProgressCounter(total: jobs.count)
        let imageOpts = options.images
        let allJobs = jobs
        let cams: [DatasetCamera?] = try await parallelMap(allJobs.count, maxConcurrent: imageDecodeConcurrency) { i in
            let j = allJobs[i]
            defer { let n = counter.increment(); progress?(PipelineProgress(stage: stage, fraction: 0.8 * Double(n) / Double(allJobs.count), message: j.src.lastPathComponent)) }
            guard FileManager.default.fileExists(atPath: j.src.path) else { return nil }
            let w = try DatasetImageWriter.write(source: j.src, intrinsics: j.k, datasetRoot: out, rel: j.rel, options: imageOpts, forceReencode: forceReencode)
            var c = DatasetCamera.make(id: i, image: w.image, width: w.width, height: w.height, fx: w.fx, fy: w.fy, cx: w.cx, cy: w.cy, c2w: j.c2w)
            c.mask = w.mask
            c.exposure = j.exposure
            // Depth is only valid for undistorted (pinhole) sensors such as the iPhone wide camera.
            if options.copyDepth, j.k.distortion.isIdentity, let d = j.depth, j.dw > 0, j.dh > 0, FileManager.default.fileExists(atPath: d.path) {
                var depth = try PreparedDataset.readDepth(d, width: j.dw, height: j.dh)
                if let cu = j.conf, options.depthMinConfidence > 0, let conf = try? Data(contentsOf: cu), conf.count >= j.dw * j.dh {
                    for p in 0..<depth.count where Int(conf[p]) < options.depthMinConfidence { depth[p] = 0 }
                }
                for p in 0..<depth.count where !(depth[p].isFinite && depth[p] > 0) { depth[p] = 0 }
                let rel = "depth/" + (j.rel as NSString).lastPathComponent + ".bin"
                try PreparedDataset.writeDepth(depth, to: out.appendingPathComponent(rel))
                c.depth = rel; c.depthWidth = j.dw; c.depthHeight = j.dh
            }
            return c
        }
        let valid = cams.compactMap { $0 }
        guard !valid.isEmpty else { throw PipelineError.noInput("capture images missing") }

        // Initial point cloud.
        var pointsRel: String? = nil
        let fusedPLY = bundle.appendingPathComponent("points.ply")
        if fm.fileExists(atPath: fusedPLY.path) {
            var pc = try PLYCodec.readPointCloud(url: fusedPLY)
            // The world frame is unchanged by the camera-convention flip, so points are kept as-is (only thinned).
            pc = DatasetAssembly.voxelDownsample(pc.points, pc.colors, voxel: options.pointsVoxelSize * 0.5, maxPoints: options.pointsMaxCount)
            try PLYCodec.writePointCloud(pc, to: out.appendingPathComponent("points3d.ply"))
            pointsRel = "points3d.ply"
        } else if options.buildPointsFromDepth {
            progress?(PipelineProgress(stage: stage, fraction: 0.85, message: "Building point cloud from depth"))
            let pc = try unprojectDepth(jobs: jobs.filter { $0.depth != nil }.map { ($0.src, $0.k, $0.c2w, $0.depth!, $0.conf, $0.dw, $0.dh) },
                                        options: options)
            if pc.count > 0 {
                try PLYCodec.writePointCloud(pc, to: out.appendingPathComponent("points3d.ply"))
                pointsRel = "points3d.ply"
            }
        }
        var ds = try DatasetAssembly.finalize(root: out, cameras: valid, points: pointsRel, holdoutInterval: options.images.holdoutInterval)
        // ARKit / OpenXR worlds are gravity-aligned with +Y up: record it so levelling never has to
        // guess from camera poses (portrait captures have sideways image axes).
        if (system ?? "arkit").lowercased() != "opencv" {
            ds.manifest.gravityUp = [0, 1, 0]
            ds.recomputeSceneBounds()
            try ds.save()
        }
        progress?(PipelineProgress(stage: stage, fraction: 1, message: "\(valid.count) cameras\(pointsRel != nil ? ", point cloud" : "")"))
        return ds
    }

    /// Unprojects confident depth pixels of (every Nth) frame into world space, coloured from the image,
    /// and voxel-downsamples the result.
    static func unprojectDepth(jobs: [(URL, SourceIntrinsics, simd_float4x4, URL, URL?, Int, Int)], options: CaptureImportOptions) throws -> PointCloud {
        var pts: [SIMD3<Float>] = [], cols: [SIMD3<UInt8>] = []
        for (n, j) in jobs.enumerated() where n % max(1, options.pointsFrameStride) == 0 {
            try Task.checkCancellation()
            let (src, k, c2w, depthURL, confURL, dw, dh) = j
            guard dw > 0, dh > 0, k.distortion.isIdentity, let depth = try? PreparedDataset.readDepth(depthURL, width: dw, height: dh) else { continue }
            let conf = confURL.flatMap { try? Data(contentsOf: $0) }
            guard let img = try? ImageUtil.load(src, maxSize: nil, applyOrientation: false) else { continue }
            let (rgba, _, _) = ImageUtil.rgba8(img, width: dw, height: dh)
            let sx = Double(k.width) / Double(dw), sy = Double(k.height) / Double(dh)
            for v in 0..<dh {
                for u in 0..<dw {
                    let p = v * dw + u
                    let z = depth[p]
                    guard z.isFinite, z > 0.05, z < options.maxDepth else { continue }
                    if let conf, conf.count > p, Int(conf[conf.startIndex + p]) < options.pointsMinConfidence { continue }
                    let x = ((Double(u) + 0.5) * sx - k.cx) / k.fx, y = ((Double(v) + 0.5) * sy - k.cy) / k.fy
                    let pc = SIMD4<Float>(Float(x) * z, Float(y) * z, z, 1)
                    let w = c2w * pc
                    pts.append(SIMD3(w.x, w.y, w.z))
                    cols.append(SIMD3(rgba[p * 4], rgba[p * 4 + 1], rgba[p * 4 + 2]))
                }
            }
        }
        return DatasetAssembly.voxelDownsample(pts, cols, voxel: options.pointsVoxelSize, maxPoints: options.pointsMaxCount)
    }
}
