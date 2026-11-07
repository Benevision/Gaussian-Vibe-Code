import Foundation
import simd

/// Parsed COLMAP sparse model (text or binary).
struct COLMAPModel: Sendable {
    struct Camera: Sendable { var id: Int; var model: String; var width: Int; var height: Int; var params: [Double] }
    struct Image: Sendable {
        var id: Int
        /// World-to-camera rotation quaternion (w, x, y, z) and translation, OpenCV camera convention.
        var qvec: SIMD4<Double>
        var tvec: SIMD3<Double>
        var cameraId: Int
        var name: String
    }
    var cameras: [Int: Camera] = [:]
    var images: [Image] = []
    var points: PointCloud = PointCloud()

    static let modelNames: [Int: (String, Int)] = [
        0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5), 4: ("OPENCV", 8),
        5: ("OPENCV_FISHEYE", 8), 6: ("FULL_OPENCV", 12), 7: ("FOV", 5), 8: ("SIMPLE_RADIAL_FISHEYE", 4),
        9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
    ]

    /// Intrinsics + distortion of a camera.
    static func intrinsics(_ c: Camera) throws -> SourceIntrinsics {
        let p = c.params
        func need(_ n: Int) throws { if p.count < n { throw PipelineError.invalidData("camera \(c.id) \(c.model) has \(p.count) params") } }
        var k = SourceIntrinsics(width: c.width, height: c.height, fx: 0, fy: 0, cx: 0, cy: 0)
        switch c.model {
        case "SIMPLE_PINHOLE": try need(3); k.fx = p[0]; k.fy = p[0]; k.cx = p[1]; k.cy = p[2]
        case "PINHOLE": try need(4); k.fx = p[0]; k.fy = p[1]; k.cx = p[2]; k.cy = p[3]
        case "SIMPLE_RADIAL": try need(4); k.fx = p[0]; k.fy = p[0]; k.cx = p[1]; k.cy = p[2]
            k.distortion = LensDistortion(kind: .opencv, params: [p[3]])
        case "RADIAL": try need(5); k.fx = p[0]; k.fy = p[0]; k.cx = p[1]; k.cy = p[2]
            k.distortion = LensDistortion(kind: .opencv, params: [p[3], p[4]])
        case "OPENCV": try need(8); k.fx = p[0]; k.fy = p[1]; k.cx = p[2]; k.cy = p[3]
            k.distortion = LensDistortion(kind: .opencv, params: Array(p[4..<8]))
        case "FULL_OPENCV": try need(12); k.fx = p[0]; k.fy = p[1]; k.cx = p[2]; k.cy = p[3]
            k.distortion = LensDistortion(kind: .fullOpencv, params: Array(p[4..<12]))
        case "OPENCV_FISHEYE": try need(8); k.fx = p[0]; k.fy = p[1]; k.cx = p[2]; k.cy = p[3]
            k.distortion = LensDistortion(kind: .fisheye, params: Array(p[4..<8]))
        case "SIMPLE_RADIAL_FISHEYE": try need(4); k.fx = p[0]; k.fy = p[0]; k.cx = p[1]; k.cy = p[2]
            k.distortion = LensDistortion(kind: .fisheye, params: [p[3]])
        case "RADIAL_FISHEYE": try need(5); k.fx = p[0]; k.fy = p[0]; k.cx = p[1]; k.cy = p[2]
            k.distortion = LensDistortion(kind: .fisheye, params: [p[3], p[4]])
        default: throw PipelineError.unsupported("COLMAP camera model \(c.model)")
        }
        return k
    }

    /// Camera-to-world (OpenCV) of an image.
    static func c2w(_ im: Image) -> simd_float4x4 {
        let q = simd_quatd(ix: im.qvec.y, iy: im.qvec.z, iz: im.qvec.w, r: im.qvec.x).normalized
        let R = simd_double3x3(q)
        let Rt = R.transpose
        let c = -(Rt * im.tvec)
        func f(_ v: SIMD3<Double>, _ w: Float) -> SIMD4<Float> { SIMD4(Float(v.x), Float(v.y), Float(v.z), w) }
        return simd_float4x4(f(Rt[0], 0), f(Rt[1], 0), f(Rt[2], 0), f(c, 1))
    }

    // MARK: Loading

    /// Loads `cameras/images/points3D` (.bin preferred, falls back to .txt) from a sparse model directory.
    static func load(sparseDir dir: URL) throws -> COLMAPModel {
        let fm = FileManager.default
        func has(_ n: String) -> Bool { fm.fileExists(atPath: dir.appendingPathComponent(n).path) }
        var m = COLMAPModel()
        if has("cameras.bin") && has("images.bin") {
            m.cameras = try readCamerasBin(dir.appendingPathComponent("cameras.bin"))
            m.images = try readImagesBin(dir.appendingPathComponent("images.bin"))
            if has("points3D.bin") { m.points = try readPointsBin(dir.appendingPathComponent("points3D.bin")) }
        } else if has("cameras.txt") && has("images.txt") {
            m.cameras = try readCamerasTxt(dir.appendingPathComponent("cameras.txt"))
            m.images = try readImagesTxt(dir.appendingPathComponent("images.txt"))
            if has("points3D.txt") { m.points = try readPointsTxt(dir.appendingPathComponent("points3D.txt")) }
        } else {
            throw PipelineError.noInput("no COLMAP model (cameras/images .bin or .txt) in \(dir.path)")
        }
        return m
    }

    private static func dataLines(_ url: URL) throws -> [Substring] {
        let s = try String(contentsOf: url, encoding: .utf8)
        return s.split(separator: "\n", omittingEmptySubsequences: false).map { $0.hasSuffix("\r") ? $0.dropLast() : $0 }
    }

    static func readCamerasTxt(_ url: URL) throws -> [Int: Camera] {
        var out: [Int: Camera] = [:]
        for l in try dataLines(url) {
            let t = l.split(separator: " ", omittingEmptySubsequences: true)
            guard !t.isEmpty, !t[0].hasPrefix("#"), t.count >= 4, let id = Int(t[0]), let w = Int(t[2]), let h = Int(t[3]) else { continue }
            out[id] = Camera(id: id, model: String(t[1]), width: w, height: h, params: t.dropFirst(4).compactMap { Double($0) })
        }
        return out
    }

    static func readImagesTxt(_ url: URL) throws -> [Image] {
        var out: [Image] = []
        let lines = try dataLines(url).filter { !$0.hasPrefix("#") }
        var i = 0
        while i < lines.count {
            let t = lines[i].split(separator: " ", omittingEmptySubsequences: true)
            if t.count >= 10, let id = Int(t[0]), let cam = Int(t[8]) {
                let d = t[1...7].map { Double($0) ?? 0 }
                out.append(Image(id: id, qvec: SIMD4(d[0], d[1], d[2], d[3]), tvec: SIMD3(d[4], d[5], d[6]), cameraId: cam,
                                 name: t[9...].joined(separator: " ")))
                i += 2   // skip the POINTS2D line (may be empty)
            } else { i += 1 }
        }
        return out
    }

    static func readPointsTxt(_ url: URL) throws -> PointCloud {
        var pc = PointCloud()
        for l in try dataLines(url) {
            guard !l.hasPrefix("#") else { continue }
            let t = l.split(separator: " ", omittingEmptySubsequences: true)
            guard t.count >= 7, let x = Float(t[1]), let y = Float(t[2]), let z = Float(t[3]) else { continue }
            pc.points.append(SIMD3(x, y, z))
            pc.colors.append(SIMD3(UInt8(t[4]) ?? 128, UInt8(t[5]) ?? 128, UInt8(t[6]) ?? 128))
        }
        return pc
    }

    /// Little-endian binary reader.
    struct BinReader {
        let data: Data
        var o = 0
        init(_ d: Data) { data = d }
        mutating func read<T: FixedWidthInteger>(_: T.Type) throws -> T {
            guard o + MemoryLayout<T>.size <= data.count else { throw PipelineError.invalidData("truncated COLMAP binary") }
            let v = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: o, as: T.self) }
            o += MemoryLayout<T>.size
            return T(littleEndian: v)
        }
        mutating func double() throws -> Double { Double(bitPattern: try read(UInt64.self)) }
        mutating func cString() throws -> String {
            guard let end = data[(data.startIndex + o)...].firstIndex(of: 0) else { throw PipelineError.invalidData("unterminated name") }
            let s = String(decoding: data[(data.startIndex + o)..<end], as: UTF8.self)
            o = end - data.startIndex + 1
            return s
        }
        mutating func skip(_ n: Int) throws {
            guard o + n <= data.count else { throw PipelineError.invalidData("truncated COLMAP binary") }
            o += n
        }
    }

    static func readCamerasBin(_ url: URL) throws -> [Int: Camera] {
        var r = BinReader(try Data(contentsOf: url))
        let n = try r.read(UInt64.self)
        var out: [Int: Camera] = [:]
        for _ in 0..<n {
            let id = Int(try r.read(Int32.self)), model = Int(try r.read(Int32.self))
            let w = Int(try r.read(UInt64.self)), h = Int(try r.read(UInt64.self))
            guard let (name, np) = modelNames[model] else { throw PipelineError.unsupported("COLMAP camera model id \(model)") }
            var params: [Double] = []
            for _ in 0..<np { params.append(try r.double()) }
            out[id] = Camera(id: id, model: name, width: w, height: h, params: params)
        }
        return out
    }

    static func readImagesBin(_ url: URL) throws -> [Image] {
        var r = BinReader(try Data(contentsOf: url, options: .mappedIfSafe))
        let n = try r.read(UInt64.self)
        var out: [Image] = []
        out.reserveCapacity(Int(n))
        for _ in 0..<n {
            let id = Int(try r.read(Int32.self))
            let q = SIMD4(try r.double(), try r.double(), try r.double(), try r.double())
            let t = SIMD3(try r.double(), try r.double(), try r.double())
            let cam = Int(try r.read(Int32.self))
            let name = try r.cString()
            let np = Int(try r.read(UInt64.self))
            try r.skip(np * 24)
            out.append(Image(id: id, qvec: q, tvec: t, cameraId: cam, name: name))
        }
        return out
    }

    static func readPointsBin(_ url: URL) throws -> PointCloud {
        var r = BinReader(try Data(contentsOf: url, options: .mappedIfSafe))
        let n = try r.read(UInt64.self)
        var pc = PointCloud()
        pc.points.reserveCapacity(Int(n)); pc.colors.reserveCapacity(Int(n))
        for _ in 0..<n {
            _ = try r.read(UInt64.self)
            let p = SIMD3(Float(try r.double()), Float(try r.double()), Float(try r.double()))
            let c = SIMD3(try r.read(UInt8.self), try r.read(UInt8.self), try r.read(UInt8.self))
            _ = try r.double()
            let tl = Int(try r.read(UInt64.self))
            try r.skip(tl * 8)
            pc.points.append(p); pc.colors.append(c)
        }
        return pc
    }
}

/// Imports a COLMAP project (sparse model + images) into a `PreparedDataset`.
enum COLMAPImporter {
    /// Finds the sparse model directory and the images directory for a COLMAP project folder.
    /// Accepts the project root (with `sparse/0` or `sparse/`), or the model directory itself.
    static func locate(_ root: URL, imagesDir: URL? = nil) throws -> (sparse: URL, images: URL) {
        let fm = FileManager.default
        func isModel(_ u: URL) -> Bool {
            fm.fileExists(atPath: u.appendingPathComponent("cameras.bin").path) || fm.fileExists(atPath: u.appendingPathComponent("cameras.txt").path)
        }
        let candidates = [root.appendingPathComponent("sparse/0"), root.appendingPathComponent("sparse"),
                          root.appendingPathComponent("colmap/sparse/0"), root]
        guard let sparse = candidates.first(where: isModel) else { throw PipelineError.noInput("no COLMAP sparse model under \(root.path)") }
        if let imagesDir { return (sparse, imagesDir) }
        var dir = sparse
        for _ in 0..<3 {
            let im = dir.appendingPathComponent("images")
            if fm.fileExists(atPath: im.path) { return (sparse, im) }
            dir = dir.deletingLastPathComponent()
        }
        return (sparse, root)
    }

    static func importProject(at root: URL, imagesDir: URL? = nil, options: ImportOptions = .init(), into out: URL,
                              progress: PipelineProgressHandler? = nil) async throws -> PreparedDataset {
        let stage = "Importing COLMAP"
        let (sparse, images) = try locate(root, imagesDir: imagesDir)
        let model = try COLMAPModel.load(sparseDir: sparse)
        guard !model.images.isEmpty else { throw PipelineError.noInput("COLMAP model has no registered images") }
        progress?(PipelineProgress(stage: stage, fraction: 0, message: "\(model.images.count) images, \(model.points.count) points"))
        let fm = FileManager.default
        try fm.ensureDirectory(out.appendingPathComponent("images"))
        let counter = ProgressCounter(total: model.images.count)
        let sorted = model.images.sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
        let cams: [DatasetCamera?] = try await parallelMap(sorted.count) { i in
            let im = sorted[i]
            defer { let n = counter.increment(); progress?(PipelineProgress(stage: stage, fraction: Double(n) / Double(sorted.count), message: im.name)) }
            guard let cam = model.cameras[im.cameraId] else { return nil }
            let src = images.appendingPathComponent(im.name)
            guard FileManager.default.fileExists(atPath: src.path) else { return nil }
            let k = try COLMAPModel.intrinsics(cam)
            let stem = (im.name as NSString).deletingPathExtension.replacingOccurrences(of: "/", with: "_")
            let w = try DatasetImageWriter.write(source: src, intrinsics: k, datasetRoot: out, rel: "images/\(stem)", options: options)
            var c = DatasetCamera.make(id: i, image: w.image, width: w.width, height: w.height, fx: w.fx, fy: w.fy, cx: w.cx, cy: w.cy, c2w: COLMAPModel.c2w(im))
            c.mask = w.mask
            return c
        }
        let valid = cams.compactMap { $0 }
        guard !valid.isEmpty else { throw PipelineError.noInput("none of the COLMAP images were found in \(images.path)") }
        var pointsRel: String? = nil
        if model.points.count > 0 {
            try PLYCodec.writePointCloud(model.points, to: out.appendingPathComponent("points3d.ply"))
            pointsRel = "points3d.ply"
        }
        let ds = try DatasetAssembly.finalize(root: out, cameras: valid, points: pointsRel, holdoutInterval: options.holdoutInterval)
        progress?(PipelineProgress(stage: stage, fraction: 1, message: "\(valid.count) cameras"))
        return ds
    }
}
