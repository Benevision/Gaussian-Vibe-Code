import Foundation
import simd

/// Scene orientation helpers.
///
/// Splat files carry no "up" direction. SplatWork uses the 3DGS/COLMAP convention (world up = −Y),
/// normalises every dataset it builds to it, writes `comment splatstudio_up x y z` into PLYs it saves,
/// and remembers the viewer's up direction per file when the user changes it.
enum Orientation {
    static let standardUp = SIMD3<Float>(0, -1, 0)
    private static let defaultsKey = "orientationByFile"

    static func savedUp(for url: URL) -> SIMD3<Float>? {
        guard let d = UserDefaults.standard.dictionary(forKey: defaultsKey)?[url.standardizedFileURL.path] as? [Double], d.count == 3 else { return nil }
        return SIMD3(Float(d[0]), Float(d[1]), Float(d[2]))
    }

    static func saveUp(_ up: SIMD3<Float>, for url: URL) {
        var all = UserDefaults.standard.dictionary(forKey: defaultsKey) ?? [:]
        all[url.standardizedFileURL.path] = [Double(up.x), Double(up.y), Double(up.z)]
        UserDefaults.standard.set(all, forKey: defaultsKey)
    }

    /// Reads `comment splatstudio_up x y z` from a PLY header.
    static func upFromPLYComment(_ url: URL) -> SIMD3<Float>? {
        guard url.pathExtension.lowercased() == "ply", let h = try? FileHandle(forReadingFrom: url),
              let d = try? h.read(upToCount: 16 * 1024) else { return nil }
        try? h.close()
        guard let text = String(data: d, encoding: .isoLatin1) else { return nil }
        for line in text.split(separator: "\n") {
            if line.hasPrefix("end_header") { break }
            let t = line.split(separator: " ")
            if t.count == 5, t[0] == "comment", t[1] == "splatstudio_up", let x = Float(t[2]), let y = Float(t[3]), let z = Float(t[4]) {
                let v = SIMD3(x, y, z)
                return simd_length(v) > 0.5 ? simd_normalize(v) : nil
            }
        }
        return nil
    }

    /// For `<project>/outputs/x.ply`, the up direction of `<project>/dataset/dataset.json`.
    static func upFromProject(of url: URL) -> SIMD3<Float>? {
        let project = url.deletingLastPathComponent().deletingLastPathComponent()
        guard url.deletingLastPathComponent().lastPathComponent == "outputs",
              let ds = try? PreparedDataset.load(project.appendingPathComponent("dataset")) else { return nil }
        let u = ds.manifest.gravityUp ?? ds.manifest.upVector
        return u.count == 3 ? simd_normalize(SIMD3(u[0], u[1], u[2])) : nil
    }

    /// Rotation taking `from` onto `to` (both unit vectors).
    static func rotation(from a: SIMD3<Float>, to b: SIMD3<Float>) -> simd_quatf {
        let a = simd_normalize(a), b = simd_normalize(b)
        let d = simd_dot(a, b)
        if d > 0.99999 { return simd_quatf(angle: 0, axis: SIMD3(1, 0, 0)) }
        if d < -0.99999 {
            var axis = simd_cross(a, SIMD3(1, 0, 0))
            if simd_length(axis) < 1e-3 { axis = simd_cross(a, SIMD3(0, 0, 1)) }
            return simd_quatf(angle: .pi, axis: simd_normalize(axis))
        }
        return simd_quatf(from: a, to: b)
    }

    /// Up direction estimated from camera poses (OpenCV convention).
    ///
    /// When the cameras look in many horizontal directions (walking around, orbiting) up is the
    /// direction most perpendicular to all viewing directions — this works even for portrait footage
    /// whose image "up" is sideways. Its sign is taken from cameras that were held upright. When the
    /// cameras mostly look the same way (one-sided or top-down drone captures) the average camera
    /// up axis, or the opposite of the average viewing direction for nadir shots, is used.
    static func estimateUp(cameras: [DatasetCamera]) -> SIMD3<Float>? {
        guard !cameras.isEmpty else { return nil }
        var avgUp = SIMD3<Float>(), avgFwd = SIMD3<Float>()
        var S = simd_float3x3()
        for c in cameras {
            let m = c.c2w
            let f = simd_normalize(SIMD3(m[2][0], m[2][1], m[2][2]))
            avgUp -= SIMD3(m[1][0], m[1][1], m[1][2])
            avgFwd += f
            S += simd_float3x3(f * f.x, f * f.y, f * f.z)
        }
        let n = Float(cameras.count)
        // Upright images (photos with EXIF applied, video frames, landscape captures): the camera
        // up axes agree — trust them directly.
        if simd_length(avgUp) / n > 0.7 { return simd_normalize(avgUp) }
        let (vals, vecs) = symmetricEigen(S * (1 / n))
        // vals ascending; spread when the two largest eigenvalues share the variance.
        let spread = vals[1] > 0.15 && vals[0] < 0.6 * vals[1]
        if spread {
            var u = vecs[0]
            var vote: Float = 0
            for c in cameras {
                let m = c.c2w
                // Image up (−y) for landscape frames; −x for portrait-held phones whose sensor
                // images are stored landscape (iPhone sensor convention).
                let camUp = -SIMD3(m[1][0], m[1][1], m[1][2]), camLeft = -SIMD3(m[0][0], m[0][1], m[0][2])
                let a = simd_dot(camUp, u), b = simd_dot(camLeft, u)
                let d = abs(a) >= abs(b) ? a : b
                if abs(d) > 0.6 { vote += d }
            }
            if vote == 0 { vote = simd_dot(avgUp, u) }
            if vote < 0 { u = -u }
            return simd_normalize(u)
        }
        if simd_length(avgUp) / n > 0.3 { return simd_normalize(avgUp) }
        if simd_length(avgFwd) / n > 0.5 { return -simd_normalize(avgFwd) }
        return simd_length(avgUp) > 1e-6 ? simd_normalize(avgUp) : nil
    }

    /// Jacobi eigen-decomposition of a symmetric 3×3 matrix; eigenvalues ascending.
    static func symmetricEigen(_ m: simd_float3x3) -> ([Float], [SIMD3<Float>]) {
        var a = (0..<3).map { r in (0..<3).map { c in Double(m[c][r]) } }
        var v: [[Double]] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        for _ in 0..<50 {
            var p = 0, q = 1
            for (i, j) in [(0, 1), (0, 2), (1, 2)] where abs(a[i][j]) > abs(a[p][q]) { p = i; q = j }
            if abs(a[p][q]) < 1e-12 { break }
            let th = 0.5 * atan2(2 * a[p][q], a[q][q] - a[p][p])
            let c = cos(th), s = sin(th)
            for k in 0..<3 {
                let akp = a[k][p], akq = a[k][q]
                a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq
            }
            for k in 0..<3 {
                let apk = a[p][k], aqk = a[q][k]
                a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk
            }
            for k in 0..<3 {
                let vkp = v[k][p], vkq = v[k][q]
                v[k][p] = c * vkp - s * vkq; v[k][q] = s * vkp + c * vkq
            }
        }
        let order = [0, 1, 2].sorted { a[$0][$0] < a[$1][$1] }
        return (order.map { Float(a[$0][$0]) }, order.map { i in simd_normalize(SIMD3(Float(v[0][i]), Float(v[1][i]), Float(v[2][i]))) })
    }
}

extension PreparedDataset {
    /// Rotates (and recentres) the dataset so its estimated up direction becomes −Y and the scene
    /// centre the origin. Rewrites cameras, `points3d.ply` and `dataset.json`. Depth maps and images
    /// are camera-relative and unchanged. Splats trained afterwards open upright in any 3DGS viewer.
    mutating func normalizeOrientation() throws {
        let gravity = manifest.gravityUp.flatMap { $0.count == 3 ? SIMD3($0[0], $0[1], $0[2]) : nil }
        guard let up = gravity.map({ simd_normalize($0) }) ?? Orientation.estimateUp(cameras: manifest.cameras) else { return }
        let q = Orientation.rotation(from: up, to: Orientation.standardUp)
        let c = manifest.sceneCenter
        let center = SIMD3(c[0], c[1], c[2])
        let R = simd_float4x4(q)
        var T = matrix_identity_float4x4
        T.columns.3 = SIMD4(-q.act(center), 1)
        let M = T * R  // p' = R p − R c
        for i in manifest.cameras.indices {
            manifest.cameras[i].c2w = M * manifest.cameras[i].c2w
        }
        if let pURL = pointCloudURL, FileManager.default.fileExists(atPath: pURL.path), var pc = try? PLYCodec.readPointCloud(url: pURL) {
            for i in pc.points.indices { pc.points[i] = q.act(pc.points[i] - center) }
            try PLYCodec.writePointCloud(pc, to: pURL)
        }
        recomputeSceneBounds()
        manifest.upVector = [0, -1, 0]
        if manifest.gravityUp != nil { manifest.gravityUp = [0, -1, 0] }
        try save()
    }
}

/// Exact rotation of real spherical-harmonic coefficients (3DGS basis ordering), so that a scene can
/// be re-oriented without changing its view-dependent colour. Each band l transforms by a (2l+1)²
/// matrix found by least squares over sample directions (exact up to float precision).
enum SHRotation {
    static func basis(_ d: SIMD3<Float>) -> [Float] {
        let C0: Float = 0.28209479177387814, C1: Float = 0.4886025119029199
        let C2: [Float] = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
        let C3: [Float] = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658, 1.445305721320277, -0.5900435899266435]
        let x = d.x, y = d.y, z = d.z, xx = x*x, yy = y*y, zz = z*z
        return [C0,
                -C1*y, C1*z, -C1*x,
                C2[0]*x*y, C2[1]*y*z, C2[2]*(2*zz-xx-yy), C2[3]*x*z, C2[4]*(xx-yy),
                C3[0]*y*(3*xx-yy), C3[1]*x*y*z, C3[2]*y*(4*zz-xx-yy), C3[3]*z*(2*zz-3*xx-3*yy),
                C3[4]*x*(4*zz-xx-yy), C3[5]*z*(xx-yy), C3[6]*x*(xx-3*yy)]
    }

    /// Returns per-band matrices `D_l` such that new coefficients c'_l = D_l · c_l when the world is
    /// rotated by `q` (positions p' = q·p).
    static func matrices(for q: simd_quatf) -> [[[Float]]] {
        // Fibonacci sphere samples
        let n = 64
        var dirs: [SIMD3<Float>] = []
        for i in 0..<n {
            let t = (Float(i) + 0.5) / Float(n)
            let phi = acos(1 - 2 * t), th = Float(i) * 2.399963
            dirs.append(SIMD3(sin(phi) * cos(th), sin(phi) * sin(th), cos(phi)))
        }
        let inv = q.inverse
        var out: [[[Float]]] = []
        for (lo, size) in [(1, 3), (4, 5), (9, 7)] {
            // f'(d) = f(q⁻¹ d):  Σ c'_k Y_k(d) = Σ c_j Y_j(q⁻¹ d)  → solve A c' = B c for D = (AᵀA)⁻¹AᵀB
            var AtA = [[Double]](repeating: [Double](repeating: 0, count: size), count: size)
            var AtB = AtA
            for d in dirs {
                let a = basis(d), b = basis(inv.act(d))
                for r in 0..<size { for c in 0..<size {
                    AtA[r][c] += Double(a[lo + r] * a[lo + c])
                    AtB[r][c] += Double(a[lo + r] * b[lo + c])
                } }
            }
            let D = solve(AtA, AtB)
            out.append(D.map { $0.map(Float.init) })
        }
        return out
    }

    /// Solves A X = B (Gauss-Jordan with partial pivoting).
    private static func solve(_ A0: [[Double]], _ B0: [[Double]]) -> [[Double]] {
        var A = A0, B = B0
        let n = A.count
        for col in 0..<n {
            let p = (col..<n).max { abs(A[$0][col]) < abs(A[$1][col]) }!
            A.swapAt(col, p); B.swapAt(col, p)
            let piv = A[col][col]
            guard abs(piv) > 1e-12 else { continue }
            for c in 0..<n { A[col][c] /= piv; B[col][c] /= piv }
            for r in 0..<n where r != col {
                let f = A[r][col]
                if f == 0 { continue }
                for c in 0..<n { A[r][c] -= f * A[col][c]; B[r][c] -= f * B[col][c] }
            }
        }
        return B
    }
}

extension GaussianCloud {
    /// Rotates the whole scene, including view-dependent colour (exact SH rotation).
    mutating func rotate(by q: simd_quatf) {
        apply(rotation: q, translation: .zero, scale: 1)
        guard shDegree > 0 else { return }
        let D = SHRotation.matrices(for: q)
        let bands = [(1, 3), (4, 5), (9, 7)].prefix(shDegree)
        for i in 0..<count {
            let b = i * GaussianCloud.shStride
            for (bi, (lo, size)) in bands.enumerated() {
                for ch in 0..<3 {
                    var v = [Float](repeating: 0, count: size)
                    for r in 0..<size { var s: Float = 0; for c in 0..<size { s += D[bi][r][c] * sh[b + (lo + c) * 3 + ch] }; v[r] = s }
                    for r in 0..<size { sh[b + (lo + r) * 3 + ch] = v[r] }
                }
            }
        }
    }
}

extension PreparedDataset {
    var upVectorSIMD: SIMD3<Float> { let u = manifest.upVector; return SIMD3(u[0], u[1], u[2]) }
}
