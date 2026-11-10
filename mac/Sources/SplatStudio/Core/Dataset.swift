import Foundation
import simd

/// One posed, undistorted pinhole camera of a prepared dataset (PROTOCOL.md §2).
struct DatasetCamera: Codable, Identifiable, Hashable {
    var id: Int
    var image: String
    var width: Int
    var height: Int
    var fx: Float
    var fy: Float
    var cx: Float
    var cy: Float
    /// Row-major 4x4, OpenCV camera convention (x right, y down, z forward).
    var cameraToWorld: [[Float]]
    var depth: String? = nil
    var depthWidth: Int? = nil
    var depthHeight: Int? = nil
    /// Optional 8-bit PNG mask (255 = keep). Masked-out pixels are ignored by the loss.
    var mask: String? = nil
    /// Optional 8-bit PNG sky mask (255 = sky): drawn by the learned skybox at infinity, not by splats.
    var sky: String? = nil
    var holdout: Bool? = nil
    /// Capture time in seconds from the start of the sequence (reserved for temporal / 4D splatting).
    var time: Double? = nil
    /// Source group, e.g. "iphone", "drone", "360", "frame-left" (reserved for multi-source datasets:
    /// per-group appearance/exposure and intrinsics).
    var group: String? = nil
    /// Relative exposure (shutter seconds × ISO / 100) when the source records it (iPhone captures).
    /// Seeds per-image exposure correction during training.
    var exposure: Double? = nil

    var c2w: simd_float4x4 {
        get {
            let m = cameraToWorld
            return simd_float4x4(rows: (0..<4).map { r in SIMD4(m[r][0], m[r][1], m[r][2], m[r][3]) })
        }
        set {
            cameraToWorld = (0..<4).map { r in [newValue[0][r], newValue[1][r], newValue[2][r], newValue[3][r]] }
        }
    }
    var center: SIMD3<Float> { let m = c2w; return SIMD3(m[3][0], m[3][1], m[3][2]) }
    var forward: SIMD3<Float> { let m = c2w; return simd_normalize(SIMD3(m[2][0], m[2][1], m[2][2])) }

    static func make(id: Int, image: String, width: Int, height: Int, fx: Float, fy: Float, cx: Float, cy: Float, c2w: simd_float4x4) -> DatasetCamera {
        var c = DatasetCamera(id: id, image: image, width: width, height: height, fx: fx, fy: fy, cx: cx, cy: cy, cameraToWorld: [])
        c.c2w = c2w
        return c
    }
}

struct DatasetManifest: Codable {
    var format: String = "splatdataset"
    var version: Int = 1
    var convention: String = "opencv"
    var sceneCenter: [Float] = [0, 0, 0]
    var sceneExtent: Float = 1
    var upVector: [Float] = [0, -1, 0]
    /// World up measured by the capture device (ARKit / OpenXR gravity), when known.
    /// Takes precedence over the camera-based estimate when levelling.
    var gravityUp: [Float]? = nil
    var points: String? = nil
    var cameras: [DatasetCamera] = []
}

/// A dataset directory on disk: `dataset.json`, `images/`, optional `depth/`, `masks/`, `points3d.ply`.
struct PreparedDataset {
    var root: URL
    var manifest: DatasetManifest

    static let fileName = "dataset.json"

    static func load(_ root: URL) throws -> PreparedDataset {
        let data = try Data(contentsOf: root.appendingPathComponent(fileName))
        let m = try JSONDecoder().decode(DatasetManifest.self, from: data)
        // Manifest paths are joined onto `root`; one with a ".." component could reach files
        // outside the dataset folder (the manifest may come from an upload in worker mode).
        var paths: [String] = m.points.map { [$0] } ?? []
        for c in m.cameras { paths += [c.image] + [c.mask, c.sky, c.depth].compactMap { $0 } }
        for p in paths where !isInsideRoot(p) {
            throw FormatError.invalid("path escapes the dataset folder: \(p)")
        }
        return PreparedDataset(root: root, manifest: m)
    }

    /// True when `rel` has no ".." component. Splits on UTF-8 bytes, as the filesystem does,
    /// so a "/" followed by a combining mark still counts as a separator.
    static func isInsideRoot(_ rel: String) -> Bool {
        !rel.utf8.split(separator: UInt8(ascii: "/")).contains { $0.elementsEqual("..".utf8) }
    }

    func save() throws {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try enc.encode(manifest).write(to: root.appendingPathComponent(Self.fileName), options: .atomic)
    }

    func url(_ rel: String) -> URL { root.appendingPathComponent(rel) }

    var pointCloudURL: URL? { manifest.points.map { url($0) } }

    var hasDepth: Bool { manifest.cameras.contains { $0.depth != nil } }
    var hasMasks: Bool { manifest.cameras.contains { $0.mask != nil } }

    /// Recomputes sceneCenter / sceneExtent / upVector from the cameras (same rule as 3DGS:
    /// extent = 1.1 × max distance of a camera centre from the mean centre).
    mutating func recomputeSceneBounds() {
        let cams = manifest.cameras
        guard !cams.isEmpty else { return }
        var mean = SIMD3<Float>.zero
        for c in cams { mean += c.center }
        mean /= Float(cams.count)
        var r: Float = 0
        for c in cams { r = max(r, simd_distance(c.center, mean)) }
        manifest.sceneCenter = [mean.x, mean.y, mean.z]
        manifest.sceneExtent = max(r * 1.1, 1e-3)
        if let g = manifest.gravityUp, g.count == 3 { manifest.upVector = g }
        else if let u = Orientation.estimateUp(cameras: cams) { manifest.upVector = [u.x, u.y, u.z] }
    }

    /// Reads a float16 depth map written per PROTOCOL.md (§1/§2).
    static func readDepth(_ url: URL, width: Int, height: Int) throws -> [Float] {
        let d = try Data(contentsOf: url)
        guard d.count >= width * height * 2 else { throw FormatError.invalid("depth map size") }
        var out = [Float](repeating: 0, count: width * height)
        d.withUnsafeBytes { raw in
            let p = raw.bindMemory(to: UInt16.self)
            for i in 0..<(width * height) { out[i] = Float(Float16(bitPattern: UInt16(littleEndian: p[i]))) }
        }
        return out
    }

    static func writeDepth(_ depth: [Float], to url: URL) throws {
        var d = Data(count: depth.count * 2)
        d.withUnsafeMutableBytes { raw in
            let p = raw.bindMemory(to: UInt16.self)
            for i in 0..<depth.count { p[i] = Float16(depth[i]).bitPattern.littleEndian }
        }
        try d.write(to: url, options: .atomic)
    }
}

extension simd_float4x4 {
    init(rows: [SIMD4<Float>]) {
        self.init(SIMD4(rows[0].x, rows[1].x, rows[2].x, rows[3].x),
                  SIMD4(rows[0].y, rows[1].y, rows[2].y, rows[3].y),
                  SIMD4(rows[0].z, rows[1].z, rows[2].z, rows[3].z),
                  SIMD4(rows[0].w, rows[1].w, rows[2].w, rows[3].w))
    }
    /// From 16 floats in column-major (simd) order, as used by capture manifests.
    init(columnMajor v: [Float]) {
        self.init(SIMD4(v[0], v[1], v[2], v[3]), SIMD4(v[4], v[5], v[6], v[7]),
                  SIMD4(v[8], v[9], v[10], v[11]), SIMD4(v[12], v[13], v[14], v[15]))
    }
    var translation: SIMD3<Float> { SIMD3(columns.3.x, columns.3.y, columns.3.z) }
    var upperLeft3x3: simd_float3x3 {
        simd_float3x3(SIMD3(columns.0.x, columns.0.y, columns.0.z), SIMD3(columns.1.x, columns.1.y, columns.1.z), SIMD3(columns.2.x, columns.2.y, columns.2.z))
    }
}
