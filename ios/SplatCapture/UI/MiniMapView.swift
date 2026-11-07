import simd
import SwiftUI

/// Top-down map of the keyframe path plus an orbit ring showing which directions
/// around the subject have been covered.
struct MiniMapView: View {
    let keyframes: [SIMD3<Float>]
    let camera: SIMD3<Float>
    let forward: SIMD3<Float>
    let center: SIMD3<Float>?
    let sectors: [Bool]

    var body: some View {
        Canvas { ctx, size in
            let pts = keyframes.map { SIMD2($0.x, $0.z) } + [SIMD2(camera.x, camera.z)]
            let c2 = center.map { SIMD2($0.x, $0.z) }
            var lo = pts.reduce(SIMD2<Float>(repeating: .infinity)) { simd_min($0, $1) }
            var hi = pts.reduce(SIMD2<Float>(repeating: -.infinity)) { simd_max($0, $1) }
            if let c2 { lo = simd_min(lo, c2); hi = simd_max(hi, c2) }
            let mid = (lo + hi) / 2
            let span = max(0.5, max(hi.x - lo.x, hi.y - lo.y)) * 1.25
            let inset: CGFloat = 14
            let scale = Float(min(size.width, size.height) - inset * 2) / span
            func map(_ p: SIMD2<Float>) -> CGPoint {
                CGPoint(x: size.width / 2 + CGFloat((p.x - mid.x) * scale),
                        y: size.height / 2 + CGFloat((p.y - mid.y) * scale))
            }

            // Orbit ring around the subject.
            if let c2 {
                let cp = map(c2)
                let radius = min(size.width, size.height) / 2 - 5
                let n = sectors.count
                for i in 0..<n {
                    let a0 = Double(i) / Double(n) * 2 * .pi - .pi
                    let a1 = Double(i + 1) / Double(n) * 2 * .pi - .pi
                    var arc = Path()
                    arc.addArc(center: cp, radius: radius, startAngle: .radians(a0 + 0.04),
                               endAngle: .radians(a1 - 0.04), clockwise: false)
                    ctx.stroke(arc, with: .color(sectors[i] ? .green : .white.opacity(0.18)),
                               style: StrokeStyle(lineWidth: 4, lineCap: .round))
                }
                let dot = CGRect(x: cp.x - 3, y: cp.y - 3, width: 6, height: 6)
                ctx.fill(Path(ellipseIn: dot), with: .color(.yellow))
            }

            // Keyframe path.
            if keyframes.count > 1 {
                var path = Path()
                path.move(to: map(SIMD2(keyframes[0].x, keyframes[0].z)))
                for k in keyframes.dropFirst() { path.addLine(to: map(SIMD2(k.x, k.z))) }
                ctx.stroke(path, with: .color(.white.opacity(0.35)), lineWidth: 1)
            }
            for k in keyframes {
                let p = map(SIMD2(k.x, k.z))
                ctx.fill(Path(ellipseIn: CGRect(x: p.x - 1.5, y: p.y - 1.5, width: 3, height: 3)),
                         with: .color(.cyan))
            }

            // Current camera with heading.
            let cp = map(SIMD2(camera.x, camera.z))
            let f = simd_normalize(SIMD2(forward.x, forward.z) + SIMD2(1e-5, 0))
            let tip = CGPoint(x: cp.x + CGFloat(f.x) * 12, y: cp.y + CGFloat(f.y) * 12)
            var heading = Path()
            heading.move(to: cp)
            heading.addLine(to: tip)
            ctx.stroke(heading, with: .color(.orange), style: StrokeStyle(lineWidth: 2.5, lineCap: .round))
            ctx.fill(Path(ellipseIn: CGRect(x: cp.x - 4, y: cp.y - 4, width: 8, height: 8)), with: .color(.orange))
        }
        .background(.black.opacity(0.45), in: RoundedRectangle(cornerRadius: 14))
        .overlay(RoundedRectangle(cornerRadius: 14).strokeBorder(.white.opacity(0.15)))
        .accessibilityLabel("Mini map showing \(keyframes.count) keyframes and orbit coverage")
    }
}
