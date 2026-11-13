import SwiftUI

/// The SplatWork mark (branding/logo-mark.svg): six linked device nodes around a splat. Drawn in code so it is crisp
/// at any size and needs no image asset.
struct HexMeshLogo: View {
    var body: some View {
        Canvas { ctx, size in
            let s = min(size.width, size.height) / 704          // the SVG's view box is 704 units across
            let c = CGPoint(x: size.width / 2, y: size.height / 2)
            let r = 292 * s
            let nodes = (0..<6).map { k -> CGPoint in
                let a = (-90 + 60 * Double(k)) * .pi / 180
                return CGPoint(x: c.x + r * cos(a), y: c.y + r * sin(a))
            }
            var ring = Path(); ring.addLines(nodes); ring.closeSubpath()
            ctx.stroke(ring, with: .color(Color(red: 0.376, green: 0.647, blue: 0.98).opacity(0.8)),
                       style: StrokeStyle(lineWidth: 24 * s, lineJoin: .round))
            for n in nodes {
                let dx = c.x - n.x, dy = c.y - n.y, l = hypot(dx, dy)
                var spoke = Path(); spoke.move(to: n)
                spoke.addLine(to: CGPoint(x: n.x + dx / l * (l - 165 * s), y: n.y + dy / l * (l - 165 * s)))
                ctx.stroke(spoke, with: .color(Color(red: 0.576, green: 0.773, blue: 0.992).opacity(0.4)),
                           style: StrokeStyle(lineWidth: 12 * s, lineCap: .round))
            }
            func ellipse(_ dx: Double, _ dy: Double, _ rx: Double, _ ry: Double, _ color: Color) {
                var e = ctx
                e.translateBy(x: c.x + dx * s, y: c.y + dy * s)
                e.rotate(by: .degrees(-30))
                e.fill(Path(ellipseIn: CGRect(x: -rx * s, y: -ry * s, width: 2 * rx * s, height: 2 * ry * s)), with: .color(color))
            }
            ellipse(0, 0, 150, 104, Color(red: 0.659, green: 0.333, blue: 0.969).opacity(0.85))
            ellipse(16, 10, 104, 70, Color(red: 0.231, green: 0.51, blue: 0.965))
            ellipse(-30, -20, 44, 24, .white.opacity(0.3))
            for n in nodes {
                ctx.fill(Path(ellipseIn: CGRect(x: n.x - 42 * s, y: n.y - 42 * s, width: 84 * s, height: 84 * s)),
                         with: .color(Color(red: 0.231, green: 0.51, blue: 0.965)))
                ctx.fill(Path(ellipseIn: CGRect(x: n.x - 19 * s, y: n.y - 19 * s, width: 38 * s, height: 38 * s)),
                         with: .color(Color(red: 0.749, green: 0.859, blue: 0.996)))
            }
        }
        .accessibilityLabel("SplatWork")
    }
}
