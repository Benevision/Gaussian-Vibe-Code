import Foundation
import simd

/// One solved camera: OpenCV camera-to-world and (optionally) refined intrinsics in pixels of the frame image.
struct SolvedPose: Sendable, Hashable {
    var c2w: simd_float4x4
    var fx: Float? = nil
    var fy: Float? = nil
    var cx: Float? = nil
    var cy: Float? = nil
    /// Replacement image path relative to the dataset root (when the solver produced undistorted copies).
    var image: String? = nil
    var width: Int? = nil
    var height: Int? = nil

    static func == (a: SolvedPose, b: SolvedPose) -> Bool { a.c2w == b.c2w && a.fx == b.fx && a.image == b.image }
    func hash(into h: inout Hasher) { h.combine(image); h.combine(fx) }
}

/// Output of a pose solver: poses keyed by the frame's `image` path (as in `FrameSet`), plus sparse points.
struct PoseSolution: Sendable {
    var poses: [String: SolvedPose] = [:]
    var points: PointCloud? = nil
    /// Human-readable notes (unregistered images, solver stats).
    var notes: [String] = []
    /// Solver statistics (e.g. splatprep `stats`).
    var stats: [String: JSONValue] = [:]
}

/// Common interface of all camera-pose solvers.
protocol PoseSolver: Sendable {
    /// Solves camera poses for the frames of `frames` (images live under `frames.root`).
    func solve(_ frames: FrameSet, progress: PipelineProgressHandler?) async throws -> PoseSolution
}

/// Turns a `FrameSet` + `PoseSolution` into a `PreparedDataset` written at `frames.root`.
enum PoseAssembly {
    static func makeDataset(frames: FrameSet, solution: PoseSolution, holdoutInterval: Int?) throws -> PreparedDataset {
        var cams: [DatasetCamera] = []
        for (i, f) in frames.frames.enumerated() {
            guard let p = solution.poses[f.image] else { continue }
            let w = p.width ?? f.width, h = p.height ?? f.height
            let fx = p.fx ?? f.focalPx ?? ImageUtil.defaultFocal(width: w, height: h)
            var c = DatasetCamera.make(id: i, image: p.image ?? f.image, width: w, height: h, fx: fx, fy: p.fy ?? p.fx ?? fx,
                                       cx: p.cx ?? f.cx ?? Float(w) / 2, cy: p.cy ?? f.cy ?? Float(h) / 2, c2w: p.c2w)
            if p.image == nil || (p.width == f.width && p.height == f.height) {
                c.mask = f.mask
                c.depth = f.depth; c.depthWidth = f.depthWidth; c.depthHeight = f.depthHeight
            }
            cams.append(c)
        }
        guard !cams.isEmpty else { throw PipelineError.poseSolveFailed("no camera could be registered") }
        var pointsRel: String? = nil
        if let pc = solution.points, pc.count > 0 {
            try PLYCodec.writePointCloud(pc, to: frames.root.appendingPathComponent("points3d.ply"))
            pointsRel = "points3d.ply"
        }
        return try DatasetAssembly.finalize(root: frames.root, cameras: cams, points: pointsRel, holdoutInterval: holdoutInterval)
    }
}

// MARK: - Rig constraints (stereo + 360)

/// Post-solve rig handling: metric rescale from a known stereo baseline, stereo-rig enforcement, and
/// 360° multi-view rig completion (views of one equirect frame share a centre and known relative rotations).
enum RigConstraints {
    /// Estimates the scale factor metric/solved = baseline / median(|c_left − c_right|). Nil if < 3 pairs.
    static func stereoScale(_ s: PoseSolution, frames: FrameSet, baseline: Float) -> Float? {
        var ratios: [Float] = []
        for (l, r) in stereoPairs(frames) {
            guard let pl = s.poses[l.image], let pr = s.poses[r.image] else { continue }
            let d = simd_distance(pl.c2w.translation, pr.c2w.translation)
            if d > 1e-6 { ratios.append(baseline / d) }
        }
        guard ratios.count >= 3 else { return nil }
        ratios.sort()
        // Robust: median of the inner 50 %.
        let lo = ratios.count / 4, hi = max(lo + 1, ratios.count - ratios.count / 4)
        let inner = Array(ratios[lo..<hi])
        return inner[inner.count / 2]
    }

    static func stereoPairs(_ frames: FrameSet) -> [(IngestedFrame, IngestedFrame)] {
        var byGroup: [Int: (IngestedFrame?, IngestedFrame?)] = [:]
        for f in frames.frames {
            guard let g = f.group, let eye = f.eye else { continue }
            var e = byGroup[g] ?? (nil, nil)
            if eye == "left" { e.0 = f } else if eye == "right" { e.1 = f }
            byGroup[g] = e
        }
        return byGroup.keys.sorted().compactMap { k in
            let e = byGroup[k]!
            if let l = e.0, let r = e.1 { return (l, r) }
            return nil
        }
    }

    /// Uniformly scales the solved scene (camera centres and points) about the origin.
    static func applyScale(_ s: Float, to sol: inout PoseSolution) {
        for k in sol.poses.keys {
            var p = sol.poses[k]!
            p.c2w.columns.3 = SIMD4(p.c2w.columns.3.x * s, p.c2w.columns.3.y * s, p.c2w.columns.3.z * s, 1)
            sol.poses[k] = p
        }
        if var pc = sol.points { pc.points = pc.points.map { $0 * s }; sol.points = pc }
    }

    /// Scales an already written dataset (cameras, points3d.ply, depth is metric and untouched).
    static func applyScale(_ s: Float, to ds: inout PreparedDataset) throws {
        for i in ds.manifest.cameras.indices {
            var m = ds.manifest.cameras[i].c2w
            m.columns.3 = SIMD4(m.columns.3.x * s, m.columns.3.y * s, m.columns.3.z * s, 1)
            ds.manifest.cameras[i].c2w = m
        }
        if let u = ds.pointCloudURL, FileManager.default.fileExists(atPath: u.path) {
            var pc = try PLYCodec.readPointCloud(url: u)
            pc.points = pc.points.map { $0 * s }
            try PLYCodec.writePointCloud(pc, to: u)
        }
        ds.recomputeSceneBounds()
        try ds.save()
    }

    /// Replaces right-eye poses by `left · rightToLeft` (call after metric rescale), filling unregistered right eyes.
    static func enforceStereoRig(_ sol: inout PoseSolution, frames: FrameSet, rig: StereoRig) {
        guard let b = rig.baseline else { return }
        var r2l = matrix_identity_float4x4
        if let m = rig.rightToLeft, m.count == 4 { r2l = simd_float4x4(rows: m.map { SIMD4($0[0], $0[1], $0[2], $0[3]) }) }
        else { r2l.columns.3 = SIMD4(b, 0, 0, 1) }   // right eye is +X of the left (OpenCV)
        for (l, r) in stereoPairs(frames) {
            guard let pl = sol.poses[l.image] else { continue }
            var pr = sol.poses[r.image] ?? SolvedPose(c2w: matrix_identity_float4x4)
            pr.c2w = pl.c2w * r2l
            if pr.fx == nil { pr.fx = pl.fx; pr.fy = pl.fy; pr.cx = pl.cx; pr.cy = pl.cy }
            sol.poses[r.image] = pr
        }
    }

    /// For 360 sources: per equirect frame, estimates the rig pose from all registered views
    /// (rotation = quaternion mean of R_view·R_toRigᵀ, centre = mean centre) and re-derives every view of the
    /// group from it — registering views SfM missed and removing intra-group drift.
    static func enforceMultiViewRig(_ sol: inout PoseSolution, frames: FrameSet, minRegisteredViews: Int = 2) {
        var groups: [Int: [IngestedFrame]] = [:]
        for f in frames.frames where f.group != nil && f.rotationToRig != nil { groups[f.group!, default: []].append(f) }
        for (_, views) in groups {
            var qs: [simd_quatf] = [], centres: [SIMD3<Float>] = []
            for v in views {
                guard let p = sol.poses[v.image] else { continue }
                let Rv = p.c2w.upperLeft3x3
                let Rc = simd_float3x3(rowMajor: v.rotationToRig!)
                qs.append(simd_quatf(orthonormalized(Rv * Rc.transpose)))
                centres.append(p.c2w.translation)
            }
            guard qs.count >= minRegisteredViews else { continue }
            // Sign-aligned quaternion average.
            var acc = SIMD4<Float>.zero
            for q in qs { let v = q.vector; acc += simd_dot(v, qs[0].vector) < 0 ? -v : v }
            let qRig = simd_quatf(vector: simd_normalize(acc))
            let Rrig = simd_float3x3(qRig)
            let c = centres.reduce(SIMD3<Float>.zero, +) / Float(centres.count)
            for v in views {
                let Rc = simd_float3x3(rowMajor: v.rotationToRig!)
                var p = sol.poses[v.image] ?? SolvedPose(c2w: matrix_identity_float4x4)
                p.c2w = simd_float4x4(rotation: Rrig * Rc, translation: c)
                sol.poses[v.image] = p
            }
        }
    }
}

// MARK: - Priors file (splatprep `priorPoses`)

/// Writes `priors.json` in the splatprep format: `{"format":"splatpriors","convention":"opencv","frames":[...]}`
/// with nested ROW-major `cameraToWorld` and optional `intrinsics` (matched by image basename).
enum PriorsFile {
    static func write(cameras: [(image: String, c2w: simd_float4x4, intrinsics: [Float]?)], to url: URL) throws {
        let frames: [[String: Any]] = cameras.map { c in
            var d: [String: Any] = ["image": (c.image as NSString).lastPathComponent, "cameraToWorld": c.c2w.rowMajor]
            if let k = c.intrinsics { d["intrinsics"] = k }
            return d
        }
        let obj: [String: Any] = ["format": "splatpriors", "convention": "opencv", "frames": frames]
        try JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted, .sortedKeys]).write(to: url, options: .atomic)
    }

    static func write(dataset: PreparedDataset, to url: URL) throws {
        try write(cameras: dataset.manifest.cameras.map { ($0.image, $0.c2w, [$0.fx, $0.fy, $0.cx, $0.cy]) }, to: url)
    }
}

// MARK: - JSON value

/// Tolerant JSON value for schemas that are still evolving (solver stats, model cards, benchmark extras).
enum JSONValue: Codable, Hashable, Sendable, CustomStringConvertible {
    case null, bool(Bool), number(Double), string(String), array([JSONValue]), object([String: JSONValue])

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { self = .null }
        else if let b = try? c.decode(Bool.self) { self = .bool(b) }
        else if let d = try? c.decode(Double.self) { self = .number(d) }
        else if let s = try? c.decode(String.self) { self = .string(s) }
        else if let a = try? c.decode([JSONValue].self) { self = .array(a) }
        else { self = .object(try c.decode([String: JSONValue].self)) }
    }
    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .null: try c.encodeNil()
        case .bool(let b): try c.encode(b)
        case .number(let d): try c.encode(d)
        case .string(let s): try c.encode(s)
        case .array(let a): try c.encode(a)
        case .object(let o): try c.encode(o)
        }
    }
    init(any: Any?) {
        switch any {
        case nil, is NSNull: self = .null
        case let n as NSNumber:
            self = CFGetTypeID(n) == CFBooleanGetTypeID() ? .bool(n.boolValue) : .number(n.doubleValue)
        case let s as String: self = .string(s)
        case let a as [Any]: self = .array(a.map { JSONValue(any: $0) })
        case let o as [String: Any]: self = .object(o.mapValues { JSONValue(any: $0) })
        default: self = .string(String(describing: any!))
        }
    }
    var double: Double? { if case .number(let d) = self { return d }; return nil }
    var string: String? { if case .string(let s) = self { return s }; return nil }
    var bool: Bool? { if case .bool(let b) = self { return b }; return nil }
    var array: [JSONValue]? { if case .array(let a) = self { return a }; return nil }
    var object: [String: JSONValue]? { if case .object(let o) = self { return o }; return nil }
    subscript(_ k: String) -> JSONValue? { object?[k] }
    var description: String {
        switch self {
        case .null: return "null"
        case .bool(let b): return b ? "true" : "false"
        case .number(let d): return d == d.rounded() && abs(d) < 1e15 ? String(Int(d)) : String(format: "%.4g", d)
        case .string(let s): return s
        case .array(let a): return "[" + a.map(\.description).joined(separator: ", ") + "]"
        case .object(let o): return "{" + o.sorted { $0.key < $1.key }.map { "\($0.key): \($0.value)" }.joined(separator: ", ") + "}"
        }
    }
}
