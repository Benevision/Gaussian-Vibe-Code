import Foundation
import Metal
import CoreGraphics
import simd

/// A per-output-pixel lookup table: for every output pixel, the (sub-pixel) source coordinate to sample.
///
/// Coordinates are in source *pixel* units with the pixel-centre-at-0.5 convention (so `(0.5, 0.5)` is the
/// centre of the top-left source pixel). Negative / out-of-range coordinates produce black unless `wrapX`
/// (used for equirectangular longitude wrap-around).
struct RemapTable: Sendable {
    var width: Int
    var height: Int
    var coords: [SIMD2<Float>]
    var wrapX: Bool = false
    /// Optional per-output-pixel validity mask (255 keep, 0 drop) computed alongside the table.
    var mask: [UInt8]? = nil
}

/// Applies a `RemapTable` to images with bilinear filtering — on the GPU (Metal compute) when available,
/// otherwise on all CPU cores. Tables are typically computed once and reused for many frames
/// (e.g. every frame of a 360° video uses the same per-view tables).
final class ImageRemapper: @unchecked Sendable {
    static let shared = ImageRemapper()

    private let device: MTLDevice?
    private let queue: MTLCommandQueue?
    private let pipeline: MTLComputePipelineState?
    private let lock = NSLock()

    private static let source = """
    #include <metal_stdlib>
    using namespace metal;
    struct Params { uint outW; uint outH; float srcW; float srcH; uint wrapX; };
    kernel void remap(texture2d<float, access::sample> src [[texture(0)]],
                      texture2d<float, access::write> dst [[texture(1)]],
                      device const float2* map [[buffer(0)]],
                      constant Params& p [[buffer(1)]],
                      uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= p.outW || gid.y >= p.outH) return;
        float2 c = map[gid.y * p.outW + gid.x];
        float4 col = float4(0, 0, 0, 1);
        bool inside = c.y >= 0 && c.y <= p.srcH && (p.wrapX != 0 || (c.x >= 0 && c.x <= p.srcW));
        if (inside) {
            constexpr sampler sWrap(coord::normalized, address::repeat, filter::linear);
            constexpr sampler sClamp(coord::normalized, address::clamp_to_edge, filter::linear);
            float2 uv = c / float2(p.srcW, p.srcH);
            col = p.wrapX != 0 ? src.sample(sWrap, float2(uv.x, clamp(uv.y, 0.0, 1.0))) : src.sample(sClamp, uv);
            col.a = 1;
        }
        dst.write(col, gid);
    }
    """

    struct Params { var outW: UInt32; var outH: UInt32; var srcW: Float; var srcH: Float; var wrapX: UInt32 }

    init() {
        let dev = MTLCreateSystemDefaultDevice()
        device = dev
        queue = dev?.makeCommandQueue()
        if let dev, let lib = try? dev.makeLibrary(source: Self.source, options: nil), let fn = lib.makeFunction(name: "remap") {
            pipeline = try? dev.makeComputePipelineState(function: fn)
        } else { pipeline = nil }
    }

    var usesGPU: Bool { pipeline != nil }

    /// Remaps an RGBA8 image (see `ImageUtil.rgba8`) and returns RGBA8 pixels of size table.width × table.height.
    func remap(rgba: [UInt8], width: Int, height: Int, table: RemapTable) -> [UInt8] {
        if let out = remapGPU(rgba: rgba, width: width, height: height, table: table) { return out }
        return remapCPU(rgba: rgba, width: width, height: height, table: table)
    }

    func remap(_ image: CGImage, table: RemapTable) throws -> CGImage {
        let (px, w, h) = ImageUtil.rgba8(image)
        let out = remap(rgba: px, width: w, height: h, table: table)
        guard let img = ImageUtil.cgImage(rgba: out, width: table.width, height: table.height) else {
            throw PipelineError.encodeFailed("remapped image")
        }
        return img
    }

    private func remapGPU(rgba: [UInt8], width: Int, height: Int, table: RemapTable) -> [UInt8]? {
        guard let device, let queue, let pipeline else { return nil }
        lock.lock(); defer { lock.unlock() }
        let sd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: width, height: height, mipmapped: false)
        sd.usage = .shaderRead; sd.storageMode = .shared
        let dd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: table.width, height: table.height, mipmapped: false)
        dd.usage = .shaderWrite; dd.storageMode = .shared
        guard let src = device.makeTexture(descriptor: sd), let dst = device.makeTexture(descriptor: dd) else { return nil }
        rgba.withUnsafeBytes { src.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0, withBytes: $0.baseAddress!, bytesPerRow: width * 4) }
        guard let mapBuf = table.coords.withUnsafeBytes({ device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }),
              let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return nil }
        var p = Params(outW: UInt32(table.width), outH: UInt32(table.height), srcW: Float(width), srcH: Float(height), wrapX: table.wrapX ? 1 : 0)
        enc.setComputePipelineState(pipeline)
        enc.setTexture(src, index: 0); enc.setTexture(dst, index: 1)
        enc.setBuffer(mapBuf, offset: 0, index: 0)
        enc.setBytes(&p, length: MemoryLayout<Params>.stride, index: 1)
        let tg = MTLSize(width: 16, height: 16, depth: 1)
        enc.dispatchThreads(MTLSize(width: table.width, height: table.height, depth: 1), threadsPerThreadgroup: tg)
        enc.endEncoding()
        cb.commit(); cb.waitUntilCompleted()
        guard cb.status == .completed else { return nil }
        var out = [UInt8](repeating: 0, count: table.width * table.height * 4)
        out.withUnsafeMutableBytes { dst.getBytes($0.baseAddress!, bytesPerRow: table.width * 4, from: MTLRegionMake2D(0, 0, table.width, table.height), mipmapLevel: 0) }
        return out
    }

    /// Multi-core bilinear fallback.
    func remapCPU(rgba: [UInt8], width: Int, height: Int, table: RemapTable) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: table.width * table.height * 4)
        let ow = table.width, wrap = table.wrapX
        rgba.withUnsafeBufferPointer { sp in
            out.withUnsafeMutableBufferPointer { dp in
                let s = sp.baseAddress!, d = dp.baseAddress!
                table.coords.withUnsafeBufferPointer { mp in
                    let m = mp.baseAddress!
                    DispatchQueue.concurrentPerform(iterations: table.height) { y in
                        for x in 0..<ow {
                            let c = m[y * ow + x]
                            let o = (y * ow + x) * 4
                            d[o + 3] = 255
                            guard c.y >= 0, c.y <= Float(height), wrap || (c.x >= 0 && c.x <= Float(width)) else { continue }
                            let fx = c.x - 0.5, fy = min(max(c.y - 0.5, 0), Float(height - 1))
                            let x0f = floor(fx), y0f = floor(fy)
                            let ax = fx - x0f, ay = fy - y0f
                            var x0 = Int(x0f), x1 = x0 + 1
                            let y0 = Int(y0f), y1 = min(y0 + 1, height - 1)
                            if wrap { x0 = ((x0 % width) + width) % width; x1 = ((x1 % width) + width) % width }
                            else { x0 = min(max(x0, 0), width - 1); x1 = min(max(x1, 0), width - 1) }
                            for ch in 0..<3 {
                                let v00 = Float(s[(y0 * width + x0) * 4 + ch]), v10 = Float(s[(y0 * width + x1) * 4 + ch])
                                let v01 = Float(s[(y1 * width + x0) * 4 + ch]), v11 = Float(s[(y1 * width + x1) * 4 + ch])
                                let v = (v00 * (1 - ax) + v10 * ax) * (1 - ay) + (v01 * (1 - ax) + v11 * ax) * ay
                                d[o + ch] = UInt8(min(max(v + 0.5, 0), 255))
                            }
                        }
                    }
                }
            }
        }
        return out
    }
}
