import Foundation
import Compression
import simd

// MARK: - antimatter15 `.splat` (32 bytes per gaussian, no SH)

enum DotSplatCodec {
    static func read(url: URL) throws -> GaussianCloud {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        guard data.count % 32 == 0 else { throw FormatError.invalid(".splat size is not a multiple of 32") }
        let n = data.count / 32
        var g = GaussianCloud(count: n, shDegree: 0)
        data.withUnsafeBytes { raw in
            let b = raw.baseAddress!
            for i in 0..<n {
                let p = b + i * 32
                for k in 0..<3 {
                    g.positions[i*3+k] = p.loadUnaligned(fromByteOffset: k*4, as: Float.self)
                    g.scales[i*3+k] = log(max(p.loadUnaligned(fromByteOffset: 12 + k*4, as: Float.self), 1e-8))
                }
                let c = p.advanced(by: 24).assumingMemoryBound(to: UInt8.self)
                for k in 0..<3 { g.sh[i*48+k] = (Float(c[k]) / 255 - 0.5) / GaussianCloud.C0 }
                g.opacities[i] = GaussianCloud.invSigmoid(Float(c[3]) / 255)
                let r = p.advanced(by: 28).assumingMemoryBound(to: UInt8.self)
                for k in 0..<4 { g.rotations[i*4+k] = (Float(r[k]) - 128) / 128 }
            }
        }
        return g
    }

    static func write(_ g: GaussianCloud, to url: URL) throws {
        // Sorted by importance (opacity × volume) like the reference converter, so progressive
        // web viewers show the most important splats first.
        let order = (0..<g.count).sorted { a, b in
            let va = exp(g.scales[a*3] + g.scales[a*3+1] + g.scales[a*3+2]) * g.opacity(a)
            let vb = exp(g.scales[b*3] + g.scales[b*3+1] + g.scales[b*3+2]) * g.opacity(b)
            return va > vb
        }
        var data = Data(count: g.count * 32)
        data.withUnsafeMutableBytes { raw in
            let b = raw.baseAddress!
            for (o, i) in order.enumerated() {
                let p = b + o * 32
                for k in 0..<3 {
                    p.storeBytes(of: g.positions[i*3+k], toByteOffset: k*4, as: Float.self)
                    p.storeBytes(of: exp(g.scales[i*3+k]), toByteOffset: 12 + k*4, as: Float.self)
                }
                let col = g.baseColor(i)
                for k in 0..<3 { p.storeBytes(of: UInt8(clamping: Int(col[k] * 255 + 0.5)), toByteOffset: 24 + k, as: UInt8.self) }
                p.storeBytes(of: UInt8(clamping: Int(g.opacity(i) * 255 + 0.5)), toByteOffset: 27, as: UInt8.self)
                var q = SIMD4(g.rotations[i*4], g.rotations[i*4+1], g.rotations[i*4+2], g.rotations[i*4+3])
                let len = simd_length(q); q = len > 0 ? q / len : SIMD4(1, 0, 0, 0)
                for k in 0..<4 { p.storeBytes(of: UInt8(clamping: Int(q[k] * 128 + 128)), toByteOffset: 28 + k, as: UInt8.self) }
            }
        }
        try data.write(to: url, options: .atomic)
    }
}

// MARK: - Niantic `.spz` (gzip compressed, quantised; versions 2 and 3)

enum SPZCodec {
    private static let magic: UInt32 = 0x5053_474e
    private static let colorScale: Float = 0.15
    private static let sqrt1_2: Float = 0.70710678
    /// SPZ stores RUB coordinates, splat PLYs are RDF. These SH coefficient indices (0-based
    /// within the 15 higher-order coefficients) change sign when y and z are flipped.
    private static let flipSH: Set<Int> = [0, 1, 3, 6, 8, 10, 11, 13]

    static func read(url: URL) throws -> GaussianCloud {
        let raw = try Gzip.decompress(Data(contentsOf: url))
        guard raw.count >= 16 else { throw FormatError.invalid("spz header") }
        return try raw.withUnsafeBytes { (buf: UnsafeRawBufferPointer) -> GaussianCloud in
            let b = buf.baseAddress!
            guard b.loadUnaligned(as: UInt32.self) == magic else { throw FormatError.invalid("spz magic") }
            let version = b.loadUnaligned(fromByteOffset: 4, as: UInt32.self)
            guard version >= 1 && version <= 3 else { throw FormatError.unsupported("spz version \(version)") }
            let n = Int(b.loadUnaligned(fromByteOffset: 8, as: UInt32.self))
            let degree = Int(b.load(fromByteOffset: 12, as: UInt8.self))
            let frac = Int(b.load(fromByteOffset: 13, as: UInt8.self))
            let flags = b.load(fromByteOffset: 14, as: UInt8.self)
            let shDim = [0, 3, 8, 15][min(degree, 3)]
            let posBytes = version == 1 ? 6 : 9
            let rotBytes = version >= 3 ? 4 : 3
            let need = 16 + n * (posBytes + 1 + 3 + 3 + rotBytes + shDim * 3)
            guard buf.count >= need else { throw FormatError.invalid("spz truncated") }
            var g = GaussianCloud(count: n, shDegree: degree)
            g.antialiased = (flags & 1) != 0
            let u8 = buf.bindMemory(to: UInt8.self)
            var o = 16
            let scale = 1 / Float(1 << frac)
            for i in 0..<n {
                for k in 0..<3 {
                    if version == 1 {
                        let h = UInt16(u8[o]) | UInt16(u8[o+1]) << 8
                        g.positions[i*3+k] = Float(Float16(bitPattern: h)); o += 2
                    } else {
                        var v = Int32(u8[o]) | Int32(u8[o+1]) << 8 | Int32(u8[o+2]) << 16
                        if v & 0x800000 != 0 { v |= Int32(bitPattern: 0xff00_0000) }
                        g.positions[i*3+k] = Float(v) * scale; o += 3
                    }
                }
            }
            for i in 0..<n { g.opacities[i] = GaussianCloud.invSigmoid(Float(u8[o]) / 255); o += 1 }
            for i in 0..<n { for k in 0..<3 { g.sh[i*48+k] = (Float(u8[o]) / 255 - 0.5) / colorScale; o += 1 } }
            for i in 0..<n { for k in 0..<3 { g.scales[i*3+k] = Float(u8[o]) / 16 - 10; o += 1 } }
            for i in 0..<n {
                var q = SIMD4<Float>(0, 0, 0, 1) // x y z w
                if version >= 3 {
                    var comp = UInt32(u8[o]) | UInt32(u8[o+1]) << 8 | UInt32(u8[o+2]) << 16 | UInt32(u8[o+3]) << 24
                    let mask: UInt32 = (1 << 9) - 1
                    let largest = Int(comp >> 30)
                    var ss: Float = 0
                    for c in stride(from: 3, through: 0, by: -1) where c != largest {
                        let mag = comp & mask, neg = (comp >> 9) & 1
                        comp >>= 10
                        var v = sqrt1_2 * Float(mag) / Float(mask)
                        if neg == 1 { v = -v }
                        q[c] = v; ss += v * v
                    }
                    q[largest] = sqrt(max(0, 1 - ss))
                    o += 4
                } else {
                    let x = Float(u8[o]) / 127.5 - 1, y = Float(u8[o+1]) / 127.5 - 1, z = Float(u8[o+2]) / 127.5 - 1
                    q = SIMD4(x, y, z, sqrt(max(0, 1 - x*x - y*y - z*z))); o += 3
                }
                // RUB -> RDF: flip y and z axes. Position (x,-y,-z); quaternion (w, x, -y, -z).
                g.rotations[i*4] = q.w; g.rotations[i*4+1] = q.x; g.rotations[i*4+2] = -q.y; g.rotations[i*4+3] = -q.z
                g.positions[i*3+1] = -g.positions[i*3+1]; g.positions[i*3+2] = -g.positions[i*3+2]
            }
            for i in 0..<n {
                for j in 0..<shDim {
                    let sign: Float = flipSH.contains(j) ? -1 : 1
                    for c in 0..<3 { g.sh[i*48 + (j+1)*3 + c] = sign * (Float(u8[o]) - 128) / 128; o += 1 }
                }
            }
            return g
        }
    }

    /// Writes SPZ version 2 (the most widely supported variant).
    static func write(_ g: GaussianCloud, to url: URL, shDegree: Int? = nil, fractionalBits: Int = 12) throws {
        let degree = min(shDegree ?? g.shDegree, 3)
        let shDim = [0, 3, 8, 15][degree]
        let n = g.count
        var out = [UInt8](repeating: 0, count: 16 + n * (9 + 1 + 3 + 3 + 3 + shDim * 3))
        func put32(_ v: UInt32, _ at: Int) { for k in 0..<4 { out[at + k] = UInt8((v >> (8 * UInt32(k))) & 0xff) } }
        put32(magic, 0); put32(2, 4); put32(UInt32(n), 8)
        out[12] = UInt8(degree); out[13] = UInt8(fractionalBits); out[14] = g.antialiased ? 1 : 0
        var o = 16
        let sc = Float(1 << fractionalBits)
        for i in 0..<n {
            for k in 0..<3 {
                var v = g.positions[i*3+k]; if k > 0 { v = -v }
                let q = Int32(clamping: Int((v * sc).rounded()))
                let c = max(-0x800000, min(0x7fffff, q))
                out[o] = UInt8(c & 0xff); out[o+1] = UInt8((c >> 8) & 0xff); out[o+2] = UInt8((c >> 16) & 0xff); o += 3
            }
        }
        func q8(_ v: Float) -> UInt8 { UInt8(clamping: Int(v.rounded())) }
        for i in 0..<n { out[o] = q8(g.opacity(i) * 255); o += 1 }
        for i in 0..<n { for k in 0..<3 { out[o] = q8(g.sh[i*48+k] * colorScale * 255 + 127.5); o += 1 } }
        for i in 0..<n { for k in 0..<3 { out[o] = q8((g.scales[i*3+k] + 10) * 16); o += 1 } }
        for i in 0..<n {
            // RDF -> RUB
            var q = SIMD4(g.rotations[i*4+1], -g.rotations[i*4+2], -g.rotations[i*4+3], g.rotations[i*4]) // x y z w
            let len = simd_length(q); q = len > 0 ? q / len : SIMD4(0, 0, 0, 1)
            if q.w < 0 { q = -q }
            for k in 0..<3 { out[o] = q8((q[k] + 1) * 127.5); o += 1 }
        }
        for i in 0..<n {
            for j in 0..<shDim {
                let sign: Float = flipSH.contains(j) ? -1 : 1
                for c in 0..<3 {
                    // Quantise like the reference encoder: 5 bits for degree 1, 4 bits for higher degrees.
                    let bucket: Float = j < 3 ? 8 : 16
                    var v = (sign * g.sh[i*48 + (j+1)*3 + c] * 128 + 128)
                    v = ((v + bucket / 2) / bucket).rounded(.down) * bucket
                    out[o] = UInt8(clamping: Int(min(max(v, 0), 255))); o += 1
                }
            }
        }
        try Gzip.compress(Data(out)).write(to: url, options: .atomic)
    }
}

// MARK: - gzip using the system Compression framework (raw deflate) + manual framing

enum Gzip {
    static func decompress(_ data: Data) throws -> Data {
        guard data.count > 18, data[0] == 0x1f, data[1] == 0x8b, data[2] == 8 else {
            // Not gzip – maybe already raw.
            return data
        }
        let flags = data[3]
        var off = 10
        if flags & 4 != 0 { let xlen = Int(data[off]) | Int(data[off+1]) << 8; off += 2 + xlen }
        if flags & 8 != 0 { while off < data.count && data[off] != 0 { off += 1 }; off += 1 }
        if flags & 16 != 0 { while off < data.count && data[off] != 0 { off += 1 }; off += 1 }
        if flags & 2 != 0 { off += 2 }
        let isize = Int(UInt32(data[data.count-4]) | UInt32(data[data.count-3]) << 8 | UInt32(data[data.count-2]) << 16 | UInt32(data[data.count-1]) << 24)
        let deflate = data.subdata(in: off..<(data.count - 8))
        return try inflate(deflate, sizeHint: isize)
    }

    static func inflate(_ src: Data, sizeHint: Int) throws -> Data {
        var result = Data()
        result.reserveCapacity(max(sizeHint, src.count * 3))
        let chunk = 1 << 20
        let dst = UnsafeMutablePointer<UInt8>.allocate(capacity: chunk)
        defer { dst.deallocate() }
        var stream = compression_stream(dst_ptr: dst, dst_size: chunk, src_ptr: dst, src_size: 0, state: nil)
        guard compression_stream_init(&stream, COMPRESSION_STREAM_DECODE, COMPRESSION_ZLIB) == COMPRESSION_STATUS_OK else { throw FormatError.invalid("zlib init") }
        defer { compression_stream_destroy(&stream) }
        try src.withUnsafeBytes { (s: UnsafeRawBufferPointer) in
            stream.src_ptr = s.bindMemory(to: UInt8.self).baseAddress!
            stream.src_size = s.count
            while true {
                stream.dst_ptr = dst; stream.dst_size = chunk
                let st = compression_stream_process(&stream, Int32(COMPRESSION_STREAM_FINALIZE.rawValue))
                result.append(dst, count: chunk - stream.dst_size)
                if st == COMPRESSION_STATUS_END { break }
                if st == COMPRESSION_STATUS_ERROR { throw FormatError.invalid("corrupt deflate stream") }
            }
        }
        return result
    }

    static func compress(_ src: Data) throws -> Data {
        var out = Data([0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 0xff])
        let chunk = 1 << 20
        let dst = UnsafeMutablePointer<UInt8>.allocate(capacity: chunk)
        defer { dst.deallocate() }
        var stream = compression_stream(dst_ptr: dst, dst_size: chunk, src_ptr: dst, src_size: 0, state: nil)
        guard compression_stream_init(&stream, COMPRESSION_STREAM_ENCODE, COMPRESSION_ZLIB) == COMPRESSION_STATUS_OK else { throw FormatError.invalid("zlib init") }
        defer { compression_stream_destroy(&stream) }
        src.withUnsafeBytes { (s: UnsafeRawBufferPointer) in
            stream.src_ptr = s.bindMemory(to: UInt8.self).baseAddress!
            stream.src_size = s.count
            while true {
                stream.dst_ptr = dst; stream.dst_size = chunk
                let st = compression_stream_process(&stream, Int32(COMPRESSION_STREAM_FINALIZE.rawValue))
                out.append(dst, count: chunk - stream.dst_size)
                if st != COMPRESSION_STATUS_OK { break }
            }
        }
        var crc = CRC32.checksum(src)
        var size = UInt32(truncatingIfNeeded: src.count)
        withUnsafeBytes(of: &crc) { out.append(contentsOf: $0) }
        withUnsafeBytes(of: &size) { out.append(contentsOf: $0) }
        return out
    }
}

enum CRC32 {
    static let table: [UInt32] = (0..<256).map { i -> UInt32 in
        var c = UInt32(i)
        for _ in 0..<8 { c = (c & 1) != 0 ? 0xEDB8_8320 ^ (c >> 1) : c >> 1 }
        return c
    }
    static func checksum(_ d: Data) -> UInt32 {
        var c: UInt32 = 0xffff_ffff
        d.withUnsafeBytes { (b: UnsafeRawBufferPointer) in
            for byte in b { c = table[Int((c ^ UInt32(byte)) & 0xff)] ^ (c >> 8) }
        }
        return c ^ 0xffff_ffff
    }
}

// MARK: - Format dispatch

enum SplatFileFormat: String, CaseIterable, Identifiable {
    case ply, splat, spz
    /// PlayCanvas compressed PLY (`.compressed.ply`); see CompressedPLY.swift.
    case compressedPly = "compressed.ply"
    var id: String { rawValue }
    var title: String {
        switch self {
        case .ply: return "PLY (3DGS standard, lossless)"
        case .splat: return ".splat (web viewers, no SH, ~32 B/splat)"
        case .spz: return "SPZ (compressed, ~10× smaller)"
        case .compressedPly: return "Compressed PLY (PlayCanvas/SuperSplat, ~16 B/splat + SH)"
        }
    }
    /// Short label for segmented pickers.
    var shortName: String {
        switch self {
        case .ply: return "PLY"
        case .splat: return "SPLAT"
        case .spz: return "SPZ"
        case .compressedPly: return "C-PLY"
        }
    }
    /// Full file extension without the leading dot (may be compound, e.g. "compressed.ply").
    var fileExtension: String { rawValue }
    /// Last path extension, for UTType / save panels ("ply" for compressed PLY).
    var baseExtension: String { self == .compressedPly ? "ply" : rawValue }

    /// Format implied by a file name, honouring the compound `.compressed.ply` extension.
    static func forURL(_ url: URL) -> SplatFileFormat? {
        if CompressedPLYCodec.hasCompressedName(url) { return .compressedPly }
        return SplatFileFormat(rawValue: url.pathExtension.lowercased())
    }

    static func load(url: URL) throws -> GaussianCloud {
        switch url.pathExtension.lowercased() {
        case "ply":
            // Compressed PLY is detected from its header, whatever the file is called.
            if CompressedPLYCodec.isCompressedPLY(url: url) { return try CompressedPLYCodec.read(url: url) }
            return try PLYCodec.readGaussians(url: url)
        case "splat": return try DotSplatCodec.read(url: url)
        case "spz": return try SPZCodec.read(url: url)
        default: throw FormatError.unsupported(url.pathExtension)
        }
    }

    func save(_ g: GaussianCloud, to url: URL, shDegree: Int?) throws {
        switch self {
        case .ply:
            // Callers that map `url.pathExtension` to a format (e.g. the CLI) get `.ply` for "x.compressed.ply".
            if CompressedPLYCodec.hasCompressedName(url) { try CompressedPLYCodec.write(g, to: url, shDegree: shDegree) }
            else { try PLYCodec.writeGaussians(g, to: url, shDegree: shDegree) }
        case .splat: try DotSplatCodec.write(g, to: url)
        case .spz: try SPZCodec.write(g, to: url, shDegree: shDegree)
        case .compressedPly: try CompressedPLYCodec.write(g, to: url, shDegree: shDegree)
        }
    }
}
