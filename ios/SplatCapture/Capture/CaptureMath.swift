import Foundation
import simd

extension simd_float4x4 {
    var position: SIMD3<Float> { SIMD3(columns.3.x, columns.3.y, columns.3.z) }
    /// Direction the camera looks at (ARKit cameras look down -Z).
    var forward: SIMD3<Float> { -SIMD3(columns.2.x, columns.2.y, columns.2.z) }
    var orientation: simd_quatf { simd_quatf(self) }

    /// Column-major array of 16 floats (simd memory order).
    var columnMajorArray: [Float] {
        let c = columns
        return [c.0.x, c.0.y, c.0.z, c.0.w,
                c.1.x, c.1.y, c.1.z, c.1.w,
                c.2.x, c.2.y, c.2.z, c.2.w,
                c.3.x, c.3.y, c.3.z, c.3.w]
    }
}

enum CaptureMath {
    /// Smallest rotation angle (radians) between two orientations.
    static func angle(_ a: simd_quatf, _ b: simd_quatf) -> Float {
        let d = min(1, abs(simd_dot(a, b)))
        return 2 * acos(d)
    }

    static func degrees(_ radians: Float) -> Float { radians * 180 / .pi }
}

/// Wraps a non-Sendable value so it can cross a queue hop we know is safe
/// (the value is handed off, never used concurrently).
struct UncheckedSendableBox<T>: @unchecked Sendable {
    let value: T
    init(_ value: T) { self.value = value }
}
