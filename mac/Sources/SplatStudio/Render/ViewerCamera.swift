import Foundation
import simd

/// Interactive camera. Internally an orbit camera (target, yaw, pitch, distance) that can
/// also be driven in fly mode (WASD) — fly mode moves the target with the eye so switching
/// between modes never jumps. World "up" is configurable because splat scenes from SfM have
/// arbitrary orientation (COLMAP scenes are usually -Y up).
struct ViewerCamera: Equatable {
    enum Mode: String, CaseIterable, Identifiable, Codable { case orbit, fly; var id: String { rawValue } }

    var target = SIMD3<Float>(0, 0, 0)
    var yaw: Float = 0
    var pitch: Float = 0.3
    var distance: Float = 4
    var fovY: Float = 50 * .pi / 180
    var up = SIMD3<Float>(0, -1, 0)
    var mode: Mode = .orbit
    var near: Float = 0.02
    var far: Float = 1000

    /// Orthonormal basis where `upAxis` is world up.
    private var basis: (right: SIMD3<Float>, up: SIMD3<Float>, fwd: SIMD3<Float>) {
        let u = simd_normalize(up)
        let ref: SIMD3<Float> = abs(u.z) < 0.9 ? SIMD3(0, 0, 1) : SIMD3(1, 0, 0)
        let r = simd_normalize(simd_cross(ref, u))
        let f = simd_cross(u, r)
        return (r, u, f)
    }

    /// Direction from target to eye.
    var offsetDir: SIMD3<Float> {
        let b = basis
        let cp = cos(pitch), sp = sin(pitch)
        return simd_normalize(b.fwd * (cos(yaw) * cp) + b.right * (sin(yaw) * cp) + b.up * sp)
    }

    var eye: SIMD3<Float> { target + offsetDir * distance }

    /// World→camera transform in OpenCV convention (x right, y down, z forward).
    var worldToCamera: simd_float4x4 {
        let z = simd_normalize(target - eye)            // forward
        let u = simd_normalize(up)
        var x = simd_cross(-u, z)                       // right = down × forward
        if simd_length(x) < 1e-6 { x = SIMD3(1, 0, 0) }
        x = simd_normalize(x)
        let y = simd_cross(z, x)                        // down
        let e = eye
        return simd_float4x4(rows: [SIMD4(x, -simd_dot(x, e)), SIMD4(y, -simd_dot(y, e)), SIMD4(z, -simd_dot(z, e)), SIMD4(0, 0, 0, 1)])
    }

    func intrinsics(width: Float, height: Float) -> SIMD4<Float> {
        let fy = 0.5 * height / tan(0.5 * fovY)
        return SIMD4(fy, fy, width * 0.5, height * 0.5)
    }

    /// Projection to Metal clip space matching the OpenCV intrinsics (for line overlays).
    func viewProjection(width: Float, height: Float) -> simd_float4x4 {
        let k = intrinsics(width: width, height: height)
        let a = far / (far - near), b = -far * near / (far - near)
        let p = simd_float4x4(rows: [
            SIMD4(2 * k.x / width, 0, 2 * k.z / width - 1, 0),
            SIMD4(0, -2 * k.y / height, 1 - 2 * k.w / height, 0),
            SIMD4(0, 0, a, b),
            SIMD4(0, 0, 1, 0),
        ])
        return p * worldToCamera
    }

    // MARK: interaction

    /// "Grab" orbit: dragging right turns the scene right, dragging down tilts its top towards you.
    /// `dx`/`dy` are in points; `speed` 1 = 0.25° per point.
    mutating func orbit(dx: Float, dy: Float, speed: Float = 1) {
        let k = 0.0044 * speed
        yaw += dx * k
        pitch = min(max(pitch + dy * k, -1.55), 1.55)
    }

    /// Look around while keeping the eye fixed (fly mode): drag right looks right.
    mutating func look(dx: Float, dy: Float, speed: Float = 1) {
        let e = eye
        orbit(dx: dx, dy: dy, speed: speed * 0.8)
        target = e - offsetDir * distance
    }

    /// Up vector seen on screen (for "level to current view").
    var screenUp: SIMD3<Float> { let m = worldToCamera; return -SIMD3(m[0][1], m[1][1], m[2][1]) }

    /// Component-wise interpolation used for camera smoothing.
    func interpolated(to g: ViewerCamera, t: Float) -> ViewerCamera {
        var c = g
        c.target = simd_mix(target, g.target, SIMD3(repeating: t))
        c.yaw = yaw + (g.yaw - yaw) * t
        c.pitch = pitch + (g.pitch - pitch) * t
        c.distance = exp(log(distance) + (log(g.distance) - log(distance)) * t)
        c.fovY = fovY + (g.fovY - fovY) * t
        return c
    }

    func isClose(to g: ViewerCamera) -> Bool {
        simd_distance(target, g.target) < 1e-5 * max(distance, 1) && abs(yaw - g.yaw) < 1e-5 && abs(pitch - g.pitch) < 1e-5
            && abs(distance - g.distance) < 1e-5 * distance && abs(fovY - g.fovY) < 1e-6 && up == g.up && mode == g.mode
    }

    /// The distance navigation is measured against: how far away the thing you're looking at is (the orbit
    /// pivot, or in fly mode the surface picked when entering/re-aiming). Clamped to the scene's scale so a
    /// stale or extreme value can't make panning or flying absurdly fast or slow.
    func workingDistance(sceneRadius r: Float?) -> Float {
        guard let r, r > 0 else { return max(distance, 0.005) }
        return min(max(distance, r * 0.005, 0.005), r * 1.0)
    }

    /// Drag the scene with the cursor: a point at the working distance follows the pointer.
    mutating func pan(dx: Float, dy: Float, viewHeight: Float, speed: Float = 1, sceneRadius: Float? = nil) {
        let m = worldToCamera
        let right = SIMD3(m[0][0], m[1][0], m[2][0])
        let down = SIMD3(m[0][1], m[1][1], m[2][1])
        let s = workingDistance(sceneRadius: sceneRadius) * 2 * tan(fovY / 2) / max(viewHeight, 1) * speed
        target += (-right * dx - down * dy) * s
    }

    mutating func zoom(_ factor: Float) {
        distance = min(max(distance * factor, 0.01), 10_000)
    }

    /// Moves eye and target together (fly mode). `v` is in camera axes: x right, y up, z forward.
    mutating func fly(_ v: SIMD3<Float>, speed: Float) {
        let m = worldToCamera
        let right = SIMD3(m[0][0], m[1][0], m[2][0])
        let down = SIMD3(m[0][1], m[1][1], m[2][1])
        let fwd = SIMD3(m[0][2], m[1][2], m[2][2])
        target += (right * v.x - down * v.y + fwd * v.z) * speed
    }

    /// Positions the camera to frame a bounding sphere.
    mutating func frame(center: SIMD3<Float>, radius: Float) {
        target = center
        distance = max(radius / sin(fovY / 2) * 0.9, 0.05)
        near = max(distance * 0.001, 0.001)
        far = max(distance * 50, 100)
    }

    /// Adopts a dataset camera (OpenCV c2w) — used for "jump to training view".
    /// `worldUp` keeps the orbit level with the scene (recommended); nil adopts the camera's own roll.
    mutating func adopt(cameraToWorld c2w: simd_float4x4, focusDistance: Float, fovY newFov: Float? = nil, worldUp: SIMD3<Float>? = nil) {
        let eyeP = SIMD3(c2w[3][0], c2w[3][1], c2w[3][2])
        let fwd = simd_normalize(SIMD3(c2w[2][0], c2w[2][1], c2w[2][2]))
        let down = SIMD3(c2w[1][0], c2w[1][1], c2w[1][2])
        up = worldUp.map { simd_normalize($0) } ?? -simd_normalize(down)
        target = eyeP + fwd * focusDistance
        distance = focusDistance
        // Recover yaw/pitch for offsetDir = -fwd in the up-basis.
        let b = basis
        let d = -fwd
        pitch = asin(min(max(simd_dot(d, b.up), -1), 1))
        yaw = atan2(simd_dot(d, b.right), simd_dot(d, b.fwd))
        if let newFov { fovY = newFov }
    }
}
