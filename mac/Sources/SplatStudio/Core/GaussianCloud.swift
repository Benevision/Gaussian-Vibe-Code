import Foundation
import simd

/// CPU-side, structure-of-arrays representation of a Gaussian splat scene.
///
/// The storage convention matches the original INRIA 3DGS PLY export, so that
/// files round-trip losslessly:
/// * `positions`  – xyz, world units
/// * `scales`     – **log** scale per axis
/// * `rotations`  – quaternion (w, x, y, z), not necessarily normalised
/// * `opacities`  – **logit** of opacity
/// * `sh`         – 16 coefficients × 3 channels per gaussian, coefficient-major
///                  (`sh[g*48 + k*3 + c]`). Coefficient 0 is the DC term.
///
/// The camera convention for a scene is OpenCV (x right, y down, z forward).
struct GaussianCloud {
    static let shStride = 48
    static let C0: Float = 0.28209479177387814

    var positions: [Float] = []
    var scales: [Float] = []
    var rotations: [Float] = []
    var opacities: [Float] = []
    var sh: [Float] = []
    /// Highest SH degree that carries information (0…3).
    var shDegree: Int = 0
    /// Whether the scene was trained with the Mip-Splatting style 2D filter (affects rendering).
    var antialiased: Bool = false

    var count: Int { opacities.count }

    init() {}

    init(count: Int, shDegree: Int) {
        positions = .init(repeating: 0, count: count * 3)
        scales = .init(repeating: 0, count: count * 3)
        rotations = .init(repeating: 0, count: count * 4)
        for i in 0..<count { rotations[i * 4] = 1 }
        opacities = .init(repeating: 0, count: count)
        sh = .init(repeating: 0, count: count * Self.shStride)
        self.shDegree = shDegree
    }

    mutating func reserve(_ n: Int) {
        positions.reserveCapacity(n * 3); scales.reserveCapacity(n * 3)
        rotations.reserveCapacity(n * 4); opacities.reserveCapacity(n)
        sh.reserveCapacity(n * Self.shStride)
    }

    func position(_ i: Int) -> SIMD3<Float> { SIMD3(positions[i*3], positions[i*3+1], positions[i*3+2]) }

    /// Colour of the DC term in [0,1] (no view dependence).
    func baseColor(_ i: Int) -> SIMD3<Float> {
        let b = i * Self.shStride
        return simd_clamp(SIMD3(sh[b], sh[b+1], sh[b+2]) * Self.C0 + 0.5, .zero, .one)
    }

    func opacity(_ i: Int) -> Float { 1 / (1 + exp(-opacities[i])) }

    /// Axis aligned bounds of all centres, ignoring outliers beyond the given percentile.
    func robustBounds(percentile: Float = 0.01) -> (min: SIMD3<Float>, max: SIMD3<Float>) {
        guard count > 0 else { return (.zero, .zero) }
        var lo = SIMD3<Float>(repeating: 0), hi = SIMD3<Float>(repeating: 0)
        let stride = max(1, count / 200_000)
        for axis in 0..<3 {
            var v: [Float] = []
            v.reserveCapacity(count / stride + 1)
            var i = 0
            while i < count { v.append(positions[i*3+axis]); i += stride }
            v.sort()
            let a = Int(Float(v.count - 1) * percentile)
            let b = Int(Float(v.count - 1) * (1 - percentile))
            lo[axis] = v[a]; hi[axis] = v[b]
        }
        return (lo, hi)
    }

    /// Keep only gaussians where `keep[i]` is true.
    func filtered(_ keep: [Bool]) -> GaussianCloud {
        var out = GaussianCloud()
        out.shDegree = shDegree; out.antialiased = antialiased
        let n = keep.reduce(0) { $0 + ($1 ? 1 : 0) }
        out.reserve(n)
        for i in 0..<count where keep[i] {
            out.positions.append(contentsOf: positions[i*3..<i*3+3])
            out.scales.append(contentsOf: scales[i*3..<i*3+3])
            out.rotations.append(contentsOf: rotations[i*4..<i*4+4])
            out.opacities.append(opacities[i])
            out.sh.append(contentsOf: sh[i*Self.shStride..<(i+1)*Self.shStride])
        }
        return out
    }

    /// Applies a rigid/similarity transform to the whole scene (positions, rotations, scales).
    /// SH coefficients of degree ≥1 are not rotated (rotation of SH is approximated by
    /// keeping them, which is exact for rotations about the view axis only); for large
    /// re-orientations prefer exporting with degree 0 or retraining.
    mutating func apply(rotation r: simd_quatf, translation t: SIMD3<Float>, scale s: Float) {
        let logS = log(s)
        for i in 0..<count {
            let p = r.act(position(i)) * s + t
            positions[i*3] = p.x; positions[i*3+1] = p.y; positions[i*3+2] = p.z
            let q = simd_quatf(ix: rotations[i*4+1], iy: rotations[i*4+2], iz: rotations[i*4+3], r: rotations[i*4])
            let nq = r * q
            rotations[i*4] = nq.real; rotations[i*4+1] = nq.imag.x; rotations[i*4+2] = nq.imag.y; rotations[i*4+3] = nq.imag.z
            scales[i*3] += logS; scales[i*3+1] += logS; scales[i*3+2] += logS
        }
    }

    /// Approximate memory footprint in bytes (CPU representation).
    var byteSize: Int { (positions.count + scales.count + rotations.count + opacities.count + sh.count) * 4 }

    static func invSigmoid(_ x: Float) -> Float { let c = min(max(x, 1e-6), 1 - 1e-6); return log(c / (1 - c)) }
}

/// Simple coloured point cloud (used to initialise training).
struct PointCloud {
    var points: [SIMD3<Float>] = []
    var colors: [SIMD3<UInt8>] = []
    var count: Int { points.count }
}
