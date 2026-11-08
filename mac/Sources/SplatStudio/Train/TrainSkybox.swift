import Foundation
import Metal
import ImageIO
import CoreGraphics
import simd

/// Mirrors `SkyParams` in TrainShaders: rows of the camera-direction → sky (right, up, forward) matrix,
/// intrinsics, panorama size, and the per-pixel sky-coverage gradient (−skyAlphaWeight / #sky pixels).
struct SkyParams {
    var r0 = SIMD4<Float>(), r1 = SIMD4<Float>(), r2 = SIMD4<Float>()
    var intr = SIMD4<Float>()
    var width: UInt32 = 0, height: UInt32 = 0
    var penalty: Float = 0
    var hasMask: UInt32 = 0
}

/// Learned skybox (same model as linux/splatworker/skybox.py): an equirectangular RGB panorama at infinity,
/// H × 2H texels stored as logits (sigmoid on use), oriented by the dataset's up vector. Every training pixel's
/// background is the panorama looked up by its view direction: pred = splats + T_final · sky(dir).
/// Longitude atan2(d·right, d·forward) (left edge −180°, wraps), latitude asin(d·up) (top row +90°), bilinear.
final class TrainSkybox {
    let width: Int, height: Int
    let up: SIMD3<Float>, forward: SIMD3<Float>, right: SIMD3<Float>
    /// Logits, interleaved RGB per texel: ((y · W) + x) · 3 + c.
    let tex: MTLBuffer
    let grad: MTLBuffer
    private let m1: MTLBuffer, m2: MTLBuffer
    private(set) var steps = 0
    var lr: Float = 0.01

    /// (up, forward, right): forward ⟂ up, as close to world +Z as possible (else +X); right = up × forward.
    static func basis(up u0: SIMD3<Float>) -> (SIMD3<Float>, SIMD3<Float>, SIMD3<Float>) {
        let u = u0 / (simd_length(u0) + 1e-12)
        let ref: SIMD3<Float> = abs(u.z) < 0.9 ? SIMD3(0, 0, 1) : SIMD3(1, 0, 0)
        var f = ref - u * simd_dot(ref, u)
        f /= simd_length(f) + 1e-12
        return (u, f, simd_cross(u, f))
    }

    init(up upVec: SIMD3<Float>, height h: Int, initRGB: SIMD3<Float> = SIMD3(0.6, 0.65, 0.7), ctx: MetalContext) {
        let b = Self.basis(up: upVec)
        up = b.0; forward = b.1; right = b.2
        height = max(h, 2); width = 2 * height
        let count = width * height * 3
        let c = simd_clamp(initRGB, SIMD3(repeating: 0.02), SIMD3(repeating: 0.98))
        let logit = log(c / (1 - c))
        var v = [Float](repeating: 0, count: count)
        for i in 0..<(width * height) { v[i * 3] = logit.x; v[i * 3 + 1] = logit.y; v[i * 3 + 2] = logit.z }
        tex = ctx.buffer(v, label: "skybox")
        grad = ctx.buffer(count * 4, label: "skybox grad")
        m1 = ctx.buffer(count * 4, label: "skybox m"); m2 = ctx.buffer(count * 4, label: "skybox v")
        memset(grad.contents(), 0, count * 4); memset(m1.contents(), 0, count * 4); memset(m2.contents(), 0, count * 4)
    }

    /// From an exported `<name>.sky.png` + `.sky.json` (e.g. to evaluate a trained splat with its sky).
    init?(_ s: GaussianCloud.Skybox, ctx: MetalContext) {
        guard let meta = try? JSONSerialization.jsonObject(with: s.meta) as? [String: Any],
              let r = meta["right"] as? [Double], let u = meta["up"] as? [Double], let f = meta["forward"] as? [Double],
              r.count == 3, u.count == 3, f.count == 3,
              let src = CGImageSourceCreateWithData(s.png as CFData, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil), img.width >= 2, img.height >= 1 else { return nil }
        func v(_ a: [Double]) -> SIMD3<Float> { SIMD3(Float(a[0]), Float(a[1]), Float(a[2])) }
        right = v(r); up = v(u); forward = v(f)
        let W = img.width, H = img.height
        width = W; height = H
        var px = [UInt8](repeating: 0, count: W * H * 4)
        let ok: Bool = px.withUnsafeMutableBytes { raw in
            guard let cg = CGContext(data: raw.baseAddress, width: W, height: H, bitsPerComponent: 8, bytesPerRow: W * 4,
                                     space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return false }
            cg.draw(img, in: CGRect(x: 0, y: 0, width: W, height: H))
            return true
        }
        guard ok else { return nil }
        let count = W * H * 3
        var t = [Float](repeating: 0, count: count)
        for i in 0..<(W * H) {
            for c in 0..<3 {
                let x = min(max(Float(px[i * 4 + c]) / 255, 0.5 / 255), 1 - 0.5 / 255)
                t[i * 3 + c] = log(x / (1 - x))
            }
        }
        tex = ctx.buffer(t, label: "skybox")
        grad = ctx.buffer(count * 4, label: "skybox grad")
        m1 = ctx.buffer(count * 4, label: "skybox m"); m2 = ctx.buffer(count * 4, label: "skybox v")
        memset(grad.contents(), 0, count * 4); memset(m1.contents(), 0, count * 4); memset(m2.contents(), 0, count * 4)
    }

    var count: Int { width * height * 3 }

    /// Shader parameters for one view: s = B · Rᵀ · d_cam with B rows (right, up, forward), R = world → camera.
    func params(for v: TrainView, penalty: Float) -> SkyParams {
        let w = v.worldToCam
        let R = simd_float3x3(SIMD3(w.columns.0.x, w.columns.0.y, w.columns.0.z),
                              SIMD3(w.columns.1.x, w.columns.1.y, w.columns.1.z),
                              SIMD3(w.columns.2.x, w.columns.2.y, w.columns.2.z))
        let B = simd_float3x3(rows: [right, up, forward])
        let M = (B * R.transpose).transpose   // columns of Mᵀ = rows of M
        let c = v.camera
        var p = SkyParams()
        p.r0 = SIMD4(M.columns.0, 0); p.r1 = SIMD4(M.columns.1, 0); p.r2 = SIMD4(M.columns.2, 0)
        p.intr = SIMD4(c.fx, c.fy, c.cx, c.cy)
        p.width = UInt32(width); p.height = UInt32(height)
        p.penalty = penalty
        p.hasMask = v.sky != nil ? 1 : 0
        return p
    }

    /// Adam (β 0.9 / 0.999, ε 1e-15) on the logits from the accumulated gradient; encoded into `enc`.
    func encodeStep(_ enc: MTLComputeCommandEncoder, ctx: MetalContext) {
        steps += 1
        let t = Float(steps)
        var ap = AdamParams(lr: lr, lr2: lr, period: 1, split: 1, bc1: 1 - pow(Float(0.9), t), bc2: 1 - pow(Float(0.999), t), n: UInt32(count))
        enc.setBuffer(tex, offset: 0, index: 0); enc.setBuffer(grad, offset: 0, index: 1)
        enc.setBuffer(m1, offset: 0, index: 2); enc.setBuffer(m2, offset: 0, index: 3)
        enc.setBytes(&ap, length: MemoryLayout<AdamParams>.stride, index: 4)
        enc.dispatch1D(ctx.pipeline("adam_step"), count: count)
    }

    /// 8-bit sRGB PNG of sigmoid(panorama) plus its orientation JSON (the viewer's and the PLY writer's format).
    func export() -> GaussianCloud.Skybox? {
        let p = tex.contents().bindMemory(to: Float.self, capacity: count)
        var px = [UInt8](repeating: 255, count: width * height * 4)
        for i in 0..<(width * height) {
            for c in 0..<3 {
                let s = 1 / (1 + exp(-p[i * 3 + c]))
                px[i * 4 + c] = UInt8(min(max(s, 0), 1) * 255 + 0.5)
            }
        }
        let data = NSMutableData()
        let ok: Bool = px.withUnsafeMutableBytes { raw in
            guard let cg = CGContext(data: raw.baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                                     space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue),
                  let img = cg.makeImage(),
                  let dst = CGImageDestinationCreateWithData(data as CFMutableData, "public.png" as CFString, 1, nil) else { return false }
            CGImageDestinationAddImage(dst, img, nil)
            return CGImageDestinationFinalize(dst)
        }
        guard ok else { return nil }
        func a(_ v: SIMD3<Float>) -> [Double] { [Double(v.x), Double(v.y), Double(v.z)] }
        let meta: [String: Any] = ["format": "equirectangular", "image": "skybox.sky.png",
                                   "up": a(up), "forward": a(forward), "right": a(right),
                                   "longitude": "atan2(dir·right, dir·forward), left edge = −180°",
                                   "latitude": "asin(dir·up), top = +90°"]
        guard let m = try? JSONSerialization.data(withJSONObject: meta, options: [.sortedKeys]) else { return nil }
        return GaussianCloud.Skybox(png: data as Data, meta: m)
    }
}
