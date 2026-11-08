// PlayCanvas "compressed PLY" (`.compressed.ply`) reader and writer.
//
// Format taken from the official PlayCanvas sources (fetched 2026-11-08):
//   Writer:  https://github.com/playcanvas/splat-transform/blob/main/src/lib/writers/write-compressed-ply.ts
//            https://github.com/playcanvas/splat-transform/blob/main/src/lib/writers/compressed-chunk.ts
//   Order:   https://github.com/playcanvas/splat-transform/blob/main/src/lib/ops/morton-order.ts
//            https://github.com/playcanvas/splat-transform/blob/main/src/lib/data-table/morton-order.ts
//   Reader:  https://github.com/playcanvas/splat-transform/blob/main/src/lib/readers/decompress-ply.ts
//   Engine:  https://github.com/playcanvas/engine/blob/main/src/framework/parsers/ply.js (isCompressedPly, readCompressedPly)
//            https://github.com/playcanvas/engine/blob/main/src/scene/gsplat/gsplat-compressed-data.js
//
// Layout (binary little endian), exactly as splat-transform writes it:
//   element chunk  ceil(N/256): 18 × float  min_x min_y min_z max_x max_y max_z
//                                            min_scale_x.. max_scale_z (log scale, clamped to ±20)
//                                            min_r min_g min_b max_r max_g max_b (colour = f_dc·C0 + 0.5)
//   element vertex N:           4 × uint    packed_position (11-10-11, x in the top bits, chunk-normalised)
//                                            packed_rotation (2-bit largest index, then the other three
//                                                             components ×√2/2+0.5 as 10-bit unorm, in
//                                                             rot_0..rot_3 order; largest made positive)
//                                            packed_scale    (11-10-11, chunk-normalised log scale)
//                                            packed_color    (RGBA8: r<<24 | g<<16 | b<<8 | sigmoid(opacity))
//   element sh N (optional):    9/24/45 × uchar f_rest_i (standard channel-major order), byte = trunc((v/8+0.5)·256)
// Splats are written in Morton (Z-order) of their centres; the final chunk is padded with its last splat
// when computing bounds. Quaternions are stored in the file's rot_0..rot_3 order (our w,x,y,z) unchanged.
// The engine requires the chunk element first with all 18 float properties in this order, then vertex.

import Foundation
import simd

enum CompressedPLYCodec {
    static let chunkSize = 256
    static let chunkProps = ["min_x", "min_y", "min_z", "max_x", "max_y", "max_z",
                             "min_scale_x", "min_scale_y", "min_scale_z", "max_scale_x", "max_scale_y", "max_scale_z",
                             "min_r", "min_g", "min_b", "max_r", "max_g", "max_b"]
    static let vertexProps = ["packed_position", "packed_rotation", "packed_scale", "packed_color"]

    /// True for file names ending in `.compressed.ply` (the conventional PlayCanvas extension).
    static func hasCompressedName(_ url: URL) -> Bool { url.lastPathComponent.lowercased().hasSuffix(".compressed.ply") }

    /// Sniffs the header: a PLY whose vertex element carries `packed_position` and that has a `chunk` element.
    static func isCompressedPLY(url: URL) -> Bool {
        guard let h = try? FileHandle(forReadingFrom: url), let d = try? h.read(upToCount: 16 * 1024) else { return false }
        try? h.close()
        guard let end = d.range(of: Data("end_header".utf8)),
              let s = String(data: d[..<end.lowerBound], encoding: .ascii) else { return false }
        return s.hasPrefix("ply") && s.contains("element chunk") && s.contains("packed_position")
    }

    // MARK: - Morton order (port of splat-transform sortMortonStrided)

    private static func part1By2(_ x: UInt32) -> UInt32 {
        var v = x & 0x0000_03ff
        v = (v ^ (v << 16)) & 0xff00_00ff
        v = (v ^ (v << 8)) & 0x0300_f00f
        v = (v ^ (v << 4)) & 0x030c_30c3
        v = (v ^ (v << 2)) & 0x0924_9249
        return v
    }

    /// Sorts `idx[range]` in place into Morton order; buckets of >256 identical codes are refined recursively.
    private static func mortonSort(_ idx: inout [UInt32], _ range: Range<Int>, _ pos: UnsafeBufferPointer<Float>) {
        let n = range.count
        guard n > 0 else { return }
        var mn = SIMD3<Float>(repeating: .infinity), mx = SIMD3<Float>(repeating: -.infinity)
        for i in range {
            let b = Int(idx[i]) * 3
            let p = SIMD3(pos[b], pos[b+1], pos[b+2])
            mn = simd_min(mn, p); mx = simd_max(mx, p)
        }
        let len = mx - mn
        guard len.x.isFinite, len.y.isFinite, len.z.isFinite else { return }
        if len == .zero { return }
        let mul = SIMD3<Float>(len.x == 0 ? 0 : 1024 / len.x, len.y == 0 ? 0 : 1024 / len.y, len.z == 0 ? 0 : 1024 / len.z)
        func q(_ v: Float) -> UInt32 { let t = min(1023, v); return t.isNaN || t <= 0 ? 0 : UInt32(t) }
        var keys = [UInt32](repeating: 0, count: n)
        var src = [UInt32](idx[range])
        for i in 0..<n {
            let b = Int(src[i]) * 3
            let ix = q((pos[b] - mn.x) * mul.x), iy = q((pos[b+1] - mn.y) * mul.y), iz = q((pos[b+2] - mn.z) * mul.z)
            keys[i] = (part1By2(iz) << 2) + (part1By2(iy) << 1) + part1By2(ix)
        }
        // Stable LSD radix sort, three 10-bit passes.
        var dstI = [UInt32](repeating: 0, count: n), dstK = [UInt32](repeating: 0, count: n)
        var counts = [Int](repeating: 0, count: 1024)
        for shift in stride(from: 0, to: 30, by: 10) {
            for d in 0..<1024 { counts[d] = 0 }
            for i in 0..<n { counts[Int((keys[i] >> UInt32(shift)) & 1023)] += 1 }
            var sum = 0
            for d in 0..<1024 { let c = counts[d]; counts[d] = sum; sum += c }
            for i in 0..<n {
                let k = Int((keys[i] >> UInt32(shift)) & 1023)
                let o = counts[k]; counts[k] += 1
                dstI[o] = src[i]; dstK[o] = keys[i]
            }
            swap(&src, &dstI); swap(&keys, &dstK)
        }
        for i in 0..<n { idx[range.lowerBound + i] = src[i] }
        var start = 0, end = 1
        while start < n {
            while end < n && keys[end] == keys[start] { end += 1 }
            if end - start > 256 { mortonSort(&idx, (range.lowerBound + start)..<(range.lowerBound + end), pos) }
            start = end
        }
    }

    // MARK: - Write

    static func write(_ g: GaussianCloud, to url: URL, shDegree: Int? = nil) throws {
        let degree = max(0, min(shDegree ?? g.shDegree, 3))
        let coeffs = [0, 3, 8, 15][degree]
        let shBytes = coeffs * 3
        let n = g.count
        let numChunks = (n + chunkSize - 1) / chunkSize

        var header = ["ply", "format binary_little_endian 1.0", "comment Generated by Splat Studio"]
        if g.antialiased { header.append("comment splatstudio_antialiased 1") }
        header.append("element chunk \(numChunks)")
        header += chunkProps.map { "property float \($0)" }
        header.append("element vertex \(n)")
        header += vertexProps.map { "property uint \($0)" }
        if degree > 0 {
            header.append("element sh \(n)")
            header += (0..<shBytes).map { "property uchar f_rest_\($0)" }
        }
        header.append("end_header\n")

        var order = (0..<UInt32(n)).map { $0 }
        g.positions.withUnsafeBufferPointer { mortonSort(&order, 0..<n, $0) }

        var chunkData = [Float](repeating: 0, count: numChunks * 18)
        var packed = [UInt32](repeating: 0, count: n * 4)
        var sh = [UInt8](repeating: 0, count: n * shBytes)

        // Per-chunk scratch: x y z s0 s1 s2 r g b opacity q0 q1 q2 q3
        var m = [[Double]](repeating: [Double](repeating: 0, count: chunkSize), count: 14)
        let C0 = Double(GaussianCloud.C0)

        @inline(__always) func packUnorm(_ v: Double, _ bits: Int) -> UInt32 {
            let t = Double((1 << bits) - 1)
            return UInt32(max(0, min(t, (v * t + 0.5).rounded(.down))))
        }
        @inline(__always) func normalize(_ x: Double, _ lo: Double, _ hi: Double) -> Double {
            if x <= lo { return 0 }
            if x >= hi { return 1 }
            return hi - lo < 0.00001 ? 0 : (x - lo) / (hi - lo)
        }
        @inline(__always) func pack111011(_ x: Double, _ y: Double, _ z: Double) -> UInt32 {
            (packUnorm(x, 11) << 21) | (packUnorm(y, 10) << 11) | packUnorm(z, 11)
        }
        func packRot(_ q0: Double, _ q1: Double, _ q2: Double, _ q3: Double) -> UInt32 {
            var a = [q0, q1, q2, q3]
            let l = (q0*q0 + q1*q1 + q2*q2 + q3*q3).squareRoot()
            if l == 0 { a = [0, 0, 0, 1] } else { for i in 0..<4 { a[i] /= l } } // pc.Quat.normalize
            var largest = 0
            for i in 1..<4 where abs(a[i]) > abs(a[largest]) { largest = i }
            if a[largest] < 0 { for i in 0..<4 { a[i] = -a[i] } }
            let norm = 2.0.squareRoot() * 0.5
            var r = UInt32(largest)
            for i in 0..<4 where i != largest { r = (r << 10) | packUnorm(a[i] * norm + 0.5, 10) }
            return r
        }

        for c in 0..<numChunks {
            let num = min(n, (c + 1) * chunkSize) - c * chunkSize
            for j in 0..<chunkSize {
                let i = Int(order[c * chunkSize + min(j, num - 1)]) // pad with the last splat
                m[0][j] = Double(g.positions[i*3]); m[1][j] = Double(g.positions[i*3+1]); m[2][j] = Double(g.positions[i*3+2])
                m[3][j] = Double(g.scales[i*3]); m[4][j] = Double(g.scales[i*3+1]); m[5][j] = Double(g.scales[i*3+2])
                let b = i * GaussianCloud.shStride
                m[6][j] = Double(g.sh[b]) * C0 + 0.5; m[7][j] = Double(g.sh[b+1]) * C0 + 0.5; m[8][j] = Double(g.sh[b+2]) * C0 + 0.5
                m[9][j] = Double(g.opacities[i])
                for k in 0..<4 { m[10+k][j] = Double(g.rotations[i*4+k]) }
                if j < num, shBytes > 0 {
                    var o = (c * chunkSize + j) * shBytes
                    for ch in 0..<3 {
                        for k in 0..<coeffs {
                            let nv = Double(g.sh[b + (k + 1) * 3 + ch]) / 8 + 0.5
                            sh[o] = UInt8(max(0, min(255, (nv * 256).rounded(.towardZero)))); o += 1
                        }
                    }
                }
            }
            var lo = [Double](repeating: 0, count: 9), hi = [Double](repeating: 0, count: 9)
            for a in 0..<9 {
                var mn = m[a][0], mx = m[a][0]
                for j in 1..<chunkSize { mn = min(mn, m[a][j]); mx = max(mx, m[a][j]) }
                if (3..<6).contains(a) { mn = max(-20, min(20, mn)); mx = max(-20, min(20, mx)) }
                lo[a] = mn; hi[a] = mx
            }
            let cd = [lo[0], lo[1], lo[2], hi[0], hi[1], hi[2], lo[3], lo[4], lo[5], hi[3], hi[4], hi[5],
                      lo[6], lo[7], lo[8], hi[6], hi[7], hi[8]]
            for k in 0..<18 { chunkData[c * 18 + k] = Float(cd[k]) }
            for j in 0..<num {
                let o = (c * chunkSize + j) * 4
                packed[o] = pack111011(normalize(m[0][j], lo[0], hi[0]), normalize(m[1][j], lo[1], hi[1]), normalize(m[2][j], lo[2], hi[2]))
                packed[o+1] = packRot(m[10][j], m[11][j], m[12][j], m[13][j])
                packed[o+2] = pack111011(normalize(m[3][j], lo[3], hi[3]), normalize(m[4][j], lo[4], hi[4]), normalize(m[5][j], lo[5], hi[5]))
                let op = 1 / (1 + exp(-m[9][j]))
                packed[o+3] = (packUnorm(normalize(m[6][j], lo[6], hi[6]), 8) << 24) | (packUnorm(normalize(m[7][j], lo[7], hi[7]), 8) << 16)
                    | (packUnorm(normalize(m[8][j], lo[8], hi[8]), 8) << 8) | packUnorm(op, 8)
            }
        }

        var data = Data(header.joined(separator: "\n").utf8)
        data.reserveCapacity(data.count + chunkData.count * 4 + packed.count * 4 + sh.count)
        chunkData.withUnsafeBytes { data.append(contentsOf: $0) }   // host is little endian (Apple silicon / x86)
        packed.withUnsafeBytes { data.append(contentsOf: $0) }
        data.append(contentsOf: sh)
        try data.write(to: url, options: .atomic)
    }

    // MARK: - Read

    static func read(url: URL) throws -> GaussianCloud {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        guard let he = data.range(of: Data("end_header".utf8)),
              let headerText = String(data: data[..<he.lowerBound], encoding: .ascii) else { throw FormatError.invalid("missing PLY header") }
        var bodyStart = he.upperBound
        if bodyStart < data.count, data[bodyStart] == 0x0d { bodyStart += 1 }
        if bodyStart < data.count, data[bodyStart] == 0x0a { bodyStart += 1 }

        struct El { var name: String; var count: Int; var props: [(name: String, type: String)] }
        var els: [El] = []
        var antialiased = false
        for line in headerText.split(whereSeparator: { $0 == "\n" || $0 == "\r" }) {
            let t = line.split(separator: " ").map(String.init)
            guard let kw = t.first else { continue }
            switch kw {
            case "format":
                guard t.count > 1, t[1] == "binary_little_endian" else { throw FormatError.unsupported("compressed PLY must be binary little endian") }
            case "comment":
                if t.count > 1, t[1] == "splatstudio_antialiased" { antialiased = true }
            case "element":
                guard t.count >= 3, let c = Int(t[2]) else { throw FormatError.invalid("bad element line") }
                els.append(El(name: t[1], count: c, props: []))
            case "property":
                guard !els.isEmpty, t.count >= 3, t[1] != "list" else { throw FormatError.invalid("unexpected property in compressed PLY") }
                els[els.count - 1].props.append((t[2], t[1]))
            default: break
            }
        }
        func size(_ ty: String) throws -> Int {
            switch ty {
            case "float", "float32", "uint", "uint32", "int", "int32": return 4
            case "uchar", "uint8", "char", "int8": return 1
            case "ushort", "uint16", "short", "int16": return 2
            case "double", "float64": return 8
            default: throw FormatError.invalid("property type \(ty)")
            }
        }
        guard let ci = els.firstIndex(where: { $0.name == "chunk" }), let vi = els.firstIndex(where: { $0.name == "vertex" }) else {
            throw FormatError.invalid("not a compressed PLY (no chunk/vertex elements)")
        }
        let chunkEl = els[ci], vertexEl = els[vi]
        let n = vertexEl.count
        guard (n + chunkSize - 1) / chunkSize == chunkEl.count else { throw FormatError.invalid("chunk count does not match vertex count") }

        // Element byte offsets in file order.
        var offsets: [Int] = [], strides: [Int] = []
        var off = bodyStart
        for e in els {
            let s = try e.props.reduce(0) { $0 + (try size($1.type)) }
            offsets.append(off); strides.append(s); off += s * e.count
        }
        guard off <= data.count else { throw FormatError.invalid("compressed PLY truncated") }

        func propOffset(_ e: El, _ name: String, _ type: Set<String>) throws -> Int? {
            var o = 0
            for p in e.props {
                if p.name == name { guard type.contains(p.type) else { throw FormatError.invalid("\(name) has type \(p.type)") }; return o }
                o += try size(p.type)
            }
            return nil
        }
        let floatT: Set<String> = ["float", "float32"], uintT: Set<String> = ["uint", "uint32"]
        var cOff: [Int] = []
        for name in chunkProps.prefix(12) {
            guard let o = try propOffset(chunkEl, name, floatT) else { throw FormatError.invalid("chunk lacks \(name)") }
            cOff.append(o)
        }
        var colorOff: [Int]? = []
        for name in chunkProps.suffix(6) {
            if let o = try propOffset(chunkEl, name, floatT) { colorOff?.append(o) } else { colorOff = nil; break }
        }
        var vOff: [Int] = []
        for name in vertexProps {
            guard let o = try propOffset(vertexEl, name, uintT) else { throw FormatError.invalid("vertex lacks \(name)") }
            vOff.append(o)
        }
        var shOffs: [Int] = []
        var shStrideBytes = 0, shBase = 0, coeffs = 0
        if let si = els.firstIndex(where: { $0.name == "sh" }) {
            let e = els[si]
            guard e.count == n, [9, 24, 45].contains(e.props.count) else { throw FormatError.invalid("bad sh element") }
            for k in 0..<e.props.count {
                guard let o = try propOffset(e, "f_rest_\(k)", ["uchar", "uint8"]) else { throw FormatError.invalid("sh lacks f_rest_\(k)") }
                shOffs.append(o)
            }
            shStrideBytes = strides[si]; shBase = offsets[si]; coeffs = e.props.count / 3
        }
        let degree = [0: 0, 3: 1, 8: 2, 15: 3][coeffs] ?? 0
        var g = GaussianCloud(count: n, shDegree: degree)
        g.antialiased = antialiased
        let C0 = GaussianCloud.C0

        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            let base = raw.baseAddress!
            @inline(__always) func unorm(_ v: UInt32, _ bits: UInt32) -> Float { let t = (UInt32(1) << bits) - 1; return Float(v & t) / Float(t) }
            @inline(__always) func lerp(_ a: Float, _ b: Float, _ t: Float) -> Float { a * (1 - t) + b * t }
            let norm: Float = 1 / (Float(2).squareRoot() * 0.5)
            var cv = [Float](repeating: 0, count: 18)
            var lastChunk = -1
            for i in 0..<n {
                let c = i / chunkSize
                if c != lastChunk {
                    let cp = base + offsets[ci] + c * strides[ci]
                    for k in 0..<12 { cv[k] = cp.loadUnaligned(fromByteOffset: cOff[k], as: Float.self) }
                    if let colorOff { for k in 0..<6 { cv[12+k] = cp.loadUnaligned(fromByteOffset: colorOff[k], as: Float.self) } }
                    lastChunk = c
                }
                let vp = base + offsets[vi] + i * strides[vi]
                let pp = vp.loadUnaligned(fromByteOffset: vOff[0], as: UInt32.self)
                let pr = vp.loadUnaligned(fromByteOffset: vOff[1], as: UInt32.self)
                let ps = vp.loadUnaligned(fromByteOffset: vOff[2], as: UInt32.self)
                let pc = vp.loadUnaligned(fromByteOffset: vOff[3], as: UInt32.self)

                g.positions[i*3] = lerp(cv[0], cv[3], unorm(pp >> 21, 11))
                g.positions[i*3+1] = lerp(cv[1], cv[4], unorm(pp >> 11, 10))
                g.positions[i*3+2] = lerp(cv[2], cv[5], unorm(pp, 11))
                g.scales[i*3] = lerp(cv[6], cv[9], unorm(ps >> 21, 11))
                g.scales[i*3+1] = lerp(cv[7], cv[10], unorm(ps >> 11, 10))
                g.scales[i*3+2] = lerp(cv[8], cv[11], unorm(ps, 11))

                let a = (unorm(pr >> 20, 10) - 0.5) * norm
                let b = (unorm(pr >> 10, 10) - 0.5) * norm
                let cc = (unorm(pr, 10) - 0.5) * norm
                let mm = max(0, 1 - (a*a + b*b + cc*cc)).squareRoot()
                let q: (Float, Float, Float, Float)
                switch pr >> 30 {
                case 0: q = (mm, a, b, cc)
                case 1: q = (a, mm, b, cc)
                case 2: q = (a, b, mm, cc)
                default: q = (a, b, cc, mm)
                }
                g.rotations[i*4] = q.0; g.rotations[i*4+1] = q.1; g.rotations[i*4+2] = q.2; g.rotations[i*4+3] = q.3

                var r = unorm(pc >> 24, 8), gg = unorm(pc >> 16, 8), bb = unorm(pc >> 8, 8)
                if colorOff != nil { r = lerp(cv[12], cv[15], r); gg = lerp(cv[13], cv[16], gg); bb = lerp(cv[14], cv[17], bb) }
                let sb = i * GaussianCloud.shStride
                g.sh[sb] = (r - 0.5) / C0; g.sh[sb+1] = (gg - 0.5) / C0; g.sh[sb+2] = (bb - 0.5) / C0
                // Reference: -log(1/a - 1); clamped here so a = 0 or 1 stays finite.
                g.opacities[i] = GaussianCloud.invSigmoid(unorm(pc, 8))

                if coeffs > 0 {
                    let sp = base + shBase + i * shStrideBytes
                    for ch in 0..<3 {
                        for k in 0..<coeffs {
                            let v = sp.load(fromByteOffset: shOffs[ch * coeffs + k], as: UInt8.self)
                            let nv: Float = v == 0 ? 0 : v == 255 ? 1 : (Float(v) + 0.5) / 256
                            g.sh[sb + (k + 1) * 3 + ch] = (nv - 0.5) * 8
                        }
                    }
                }
            }
        }
        return g
    }
}
