import ARKit
import CoreVideo
import Foundation

/// Depth + confidence copied out of an `ARDepthData` (so the ARFrame can be released at once).
struct DepthSample: Sendable {
    let width: Int
    let height: Int
    /// Metres, row-major, tightly packed.
    let depth: [Float]
    /// 0/1/2 per pixel, row-major, tightly packed. nil when unavailable.
    let confidence: [UInt8]?

    init?(_ data: ARDepthData) {
        let map = data.depthMap
        guard CVPixelBufferGetPixelFormatType(map) == kCVPixelFormatType_DepthFloat32 else { return nil }
        CVPixelBufferLockBaseAddress(map, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(map, .readOnly) }
        let w = CVPixelBufferGetWidth(map), h = CVPixelBufferGetHeight(map)
        let bpr = CVPixelBufferGetBytesPerRow(map)
        guard let base = CVPixelBufferGetBaseAddress(map) else { return nil }
        var values = [Float](repeating: 0, count: w * h)
        values.withUnsafeMutableBufferPointer { dst in
            for y in 0..<h {
                let row = base.advanced(by: y * bpr).assumingMemoryBound(to: Float32.self)
                for x in 0..<w {
                    let v = row[x]
                    dst[y * w + x] = v.isFinite ? v : 0
                }
            }
        }
        width = w
        height = h
        depth = values

        if let conf = data.confidenceMap {
            CVPixelBufferLockBaseAddress(conf, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(conf, .readOnly) }
            let cw = CVPixelBufferGetWidth(conf), ch = CVPixelBufferGetHeight(conf)
            let cbpr = CVPixelBufferGetBytesPerRow(conf)
            if cw == w, ch == h, let cbase = CVPixelBufferGetBaseAddress(conf) {
                var c = [UInt8](repeating: 0, count: w * h)
                c.withUnsafeMutableBufferPointer { dst in
                    for y in 0..<h {
                        let row = cbase.advanced(by: y * cbpr).assumingMemoryBound(to: UInt8.self)
                        for x in 0..<w { dst[y * w + x] = row[x] }
                    }
                }
                confidence = c
            } else {
                confidence = nil
            }
        } else {
            confidence = nil
        }
    }

    /// float16 little-endian, row-major (PROTOCOL §1).
    func float16Data() -> Data {
        var out = Data(count: depth.count * 2)
        out.withUnsafeMutableBytes { raw in
            let p = raw.bindMemory(to: UInt16.self)
            for i in 0..<depth.count {
                p[i] = Float16(depth[i]).bitPattern.littleEndian
            }
        }
        return out
    }

    func confidenceData() -> Data? {
        confidence.map { Data($0) }
    }

    /// Median depth of the central patch (for "too close" guidance).
    func centerDepth() -> Float? {
        let rx = max(2, width / 16), ry = max(2, height / 16)
        var vals: [Float] = []
        vals.reserveCapacity(4 * rx * ry)
        for y in (height / 2 - ry)..<(height / 2 + ry) {
            for x in (width / 2 - rx)..<(width / 2 + rx) {
                let v = depth[y * width + x]
                if v > 0 { vals.append(v) }
            }
        }
        guard !vals.isEmpty else { return nil }
        vals.sort()
        return vals[vals.count / 2]
    }
}

enum PixelBufferTools {
    /// Deep copy of a (planar or packed) pixel buffer into an IOSurface-backed
    /// buffer we own, so ARKit's capture pool isn't starved while we encode.
    static func copy(_ src: CVPixelBuffer) -> CVPixelBuffer? {
        let w = CVPixelBufferGetWidth(src), h = CVPixelBufferGetHeight(src)
        let format = CVPixelBufferGetPixelFormatType(src)
        let attrs: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [CFString: Any]()]
        var out: CVPixelBuffer?
        guard CVPixelBufferCreate(kCFAllocatorDefault, w, h, format, attrs as CFDictionary, &out) == kCVReturnSuccess,
              let dst = out else { return nil }
        CVPixelBufferLockBaseAddress(src, .readOnly)
        CVPixelBufferLockBaseAddress(dst, [])
        defer {
            CVPixelBufferUnlockBaseAddress(dst, [])
            CVPixelBufferUnlockBaseAddress(src, .readOnly)
        }
        if CVPixelBufferIsPlanar(src) {
            let planes = CVPixelBufferGetPlaneCount(src)
            for p in 0..<planes {
                guard let s = CVPixelBufferGetBaseAddressOfPlane(src, p),
                      let d = CVPixelBufferGetBaseAddressOfPlane(dst, p) else { return nil }
                let rows = CVPixelBufferGetHeightOfPlane(src, p)
                let sb = CVPixelBufferGetBytesPerRowOfPlane(src, p)
                let db = CVPixelBufferGetBytesPerRowOfPlane(dst, p)
                copyRows(s, sb, d, db, rows)
            }
        } else {
            guard let s = CVPixelBufferGetBaseAddress(src), let d = CVPixelBufferGetBaseAddress(dst) else { return nil }
            copyRows(s, CVPixelBufferGetBytesPerRow(src), d, CVPixelBufferGetBytesPerRow(dst), h)
        }
        if let attachments = CVBufferCopyAttachments(src, .shouldPropagate) {
            CVBufferSetAttachments(dst, attachments, .shouldPropagate)
        }
        return dst
    }

    private static func copyRows(_ s: UnsafeMutableRawPointer, _ sb: Int, _ d: UnsafeMutableRawPointer, _ db: Int, _ rows: Int) {
        if sb == db {
            memcpy(d, s, sb * rows)
        } else {
            let n = min(sb, db)
            for r in 0..<rows { memcpy(d + r * db, s + r * sb, n) }
        }
    }

    /// Variance of a 4-neighbour Laplacian over a sparse grid of the luma plane.
    /// Cheap (~30k samples) and a good relative focus / motion-blur measure.
    static func laplacianVariance(_ pb: CVPixelBuffer) -> Float {
        guard CVPixelBufferIsPlanar(pb) else { return 0 }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddressOfPlane(pb, 0) else { return 0 }
        let w = CVPixelBufferGetWidthOfPlane(pb, 0), h = CVPixelBufferGetHeightOfPlane(pb, 0)
        let bpr = CVPixelBufferGetBytesPerRowOfPlane(pb, 0)
        let p = base.assumingMemoryBound(to: UInt8.self)
        // Scale the neighbour offset with resolution so 4K and 1440p behave alike.
        let d = max(1, w / 960)
        let step = max(d * 2, w / 200)
        let x0 = w / 10 + d, x1 = w - w / 10 - d
        let y0 = h / 10 + d, y1 = h - h / 10 - d
        var sum: Double = 0, sumSq: Double = 0, n = 0
        var y = y0
        while y < y1 {
            let row = p + y * bpr
            let up = row - d * bpr, down = row + d * bpr
            var x = x0
            while x < x1 {
                let lap = 4 * Int(row[x]) - Int(row[x - d]) - Int(row[x + d]) - Int(up[x]) - Int(down[x])
                let v = Double(lap)
                sum += v
                sumSq += v * v
                n += 1
                x += step
            }
            y += step
        }
        guard n > 1 else { return 0 }
        let mean = sum / Double(n)
        return Float(sumSq / Double(n) - mean * mean)
    }
}

/// Read-only YCbCr 4:2:0 bi-planar sampler (ARKit capturedImage format), full-range BT.601.
struct YCbCrSampler {
    private let y: UnsafePointer<UInt8>
    private let c: UnsafePointer<UInt8>
    private let yBPR: Int, cBPR: Int
    let width: Int, height: Int

    /// The buffer must stay locked (read-only) for the lifetime of the sampler.
    init?(lockedBuffer pb: CVPixelBuffer) {
        guard CVPixelBufferGetPlaneCount(pb) == 2,
              let yb = CVPixelBufferGetBaseAddressOfPlane(pb, 0),
              let cb = CVPixelBufferGetBaseAddressOfPlane(pb, 1) else { return nil }
        y = UnsafePointer(yb.assumingMemoryBound(to: UInt8.self))
        c = UnsafePointer(cb.assumingMemoryBound(to: UInt8.self))
        yBPR = CVPixelBufferGetBytesPerRowOfPlane(pb, 0)
        cBPR = CVPixelBufferGetBytesPerRowOfPlane(pb, 1)
        width = CVPixelBufferGetWidthOfPlane(pb, 0)
        height = CVPixelBufferGetHeightOfPlane(pb, 0)
    }

    func rgb(x: Int, y py: Int) -> SIMD3<Float> {
        let xx = min(max(x, 0), width - 1), yy = min(max(py, 0), height - 1)
        let Y = Float(y[yy * yBPR + xx])
        let ci = (yy / 2) * cBPR + (xx / 2) * 2
        let cb = Float(c[ci]) - 128, cr = Float(c[ci + 1]) - 128
        let r = Y + 1.402 * cr
        let g = Y - 0.344136 * cb - 0.714136 * cr
        let b = Y + 1.772 * cb
        return simd_clamp(SIMD3(r, g, b), SIMD3(repeating: 0), SIMD3(repeating: 255))
    }
}
