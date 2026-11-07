import Foundation
import simd

enum FormatError: LocalizedError {
    case invalid(String)
    case unsupported(String)
    var errorDescription: String? {
        switch self {
        case .invalid(let s): return "Invalid file: \(s)"
        case .unsupported(let s): return "Unsupported file: \(s)"
        }
    }
}

/// Minimal but complete PLY reader: ascii, binary little/big endian, any scalar property types.
/// List properties are skipped (faces etc.).
struct PLYFile {
    enum ScalarType: String {
        case char, uchar, short, ushort, int, uint, float, double
        case int8, uint8, int16, uint16, int32, uint32, float32, float64
        var size: Int {
            switch self {
            case .char, .uchar, .int8, .uint8: return 1
            case .short, .ushort, .int16, .uint16: return 2
            case .int, .uint, .int32, .uint32, .float, .float32: return 4
            case .double, .float64: return 8
            }
        }
    }
    struct Property { var name: String; var type: ScalarType; var listCount: ScalarType? }
    struct Element { var name: String; var count: Int; var properties: [Property] }
    enum Encoding { case ascii, binaryLE, binaryBE }

    var elements: [Element] = []
    var encoding: Encoding = .binaryLE
    /// Decoded vertex properties, by name, as Float columns.
    var vertexColumns: [String: [Float]] = [:]
    var vertexCount: Int = 0

    init(url: URL, wantedVertexProperties: Set<String>? = nil) throws {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        try self.init(data: data, wanted: wantedVertexProperties)
    }

    init(data: Data, wanted: Set<String>? = nil) throws {
        // --- header
        guard let headerEnd = data.range(of: Data("end_header".utf8)) else { throw FormatError.invalid("missing PLY header") }
        var bodyStart = headerEnd.upperBound
        // skip the line ending after end_header
        while bodyStart < data.count, data[bodyStart] == 0x0d || data[bodyStart] == 0x0a {
            bodyStart += 1
            if data[bodyStart - 1] == 0x0a { break }
        }
        guard let header = String(data: data[..<headerEnd.lowerBound], encoding: .ascii) else { throw FormatError.invalid("header not ascii") }
        let lines = header.split(whereSeparator: { $0 == "\n" || $0 == "\r" })
        guard lines.first?.trimmingCharacters(in: .whitespaces) == "ply" else { throw FormatError.invalid("not a PLY file") }
        for raw in lines.dropFirst() {
            let t = raw.split(separator: " ").map(String.init)
            guard let kw = t.first else { continue }
            switch kw {
            case "format":
                switch t.count > 1 ? t[1] : "" {
                case "ascii": encoding = .ascii
                case "binary_little_endian": encoding = .binaryLE
                case "binary_big_endian": encoding = .binaryBE
                default: throw FormatError.unsupported("PLY format \(t)")
                }
            case "element":
                guard t.count >= 3, let n = Int(t[2]) else { throw FormatError.invalid("bad element line") }
                elements.append(Element(name: t[1], count: n, properties: []))
            case "property":
                guard !elements.isEmpty else { continue }
                if t.count >= 5, t[1] == "list" {
                    guard let ct = ScalarType(rawValue: t[2]), let it = ScalarType(rawValue: t[3]) else { throw FormatError.invalid("bad list property") }
                    elements[elements.count - 1].properties.append(Property(name: t[4], type: it, listCount: ct))
                } else if t.count >= 3, let ty = ScalarType(rawValue: t[1]) {
                    elements[elements.count - 1].properties.append(Property(name: t[2], type: ty, listCount: nil))
                } else { throw FormatError.invalid("bad property line \(t)") }
            default: break
            }
        }
        try decodeBody(data: data, start: bodyStart, wanted: wanted)
    }

    private mutating func decodeBody(data: Data, start: Int, wanted: Set<String>?) throws {
        if encoding == .ascii {
            guard let text = String(data: data[start...], encoding: .ascii) else { throw FormatError.invalid("ascii body") }
            var tokens = text.split(whereSeparator: { $0 == " " || $0 == "\n" || $0 == "\r" || $0 == "\t" }).makeIterator()
            for el in elements {
                let isVertex = el.name == "vertex"
                var cols: [[Float]] = isVertex ? el.properties.map { _ in [Float](repeating: 0, count: el.count) } : []
                for r in 0..<el.count {
                    for (pi, p) in el.properties.enumerated() {
                        if let _ = p.listCount {
                            let n = Int(Float(tokens.next() ?? "0") ?? 0)
                            for _ in 0..<n { _ = tokens.next() }
                        } else {
                            let v = Float(tokens.next() ?? "0") ?? 0
                            if isVertex { cols[pi][r] = v }
                        }
                    }
                }
                if isVertex {
                    vertexCount = el.count
                    for (pi, p) in el.properties.enumerated() where p.listCount == nil && (wanted?.contains(p.name) ?? true) {
                        vertexColumns[p.name] = cols[pi]
                    }
                }
            }
            return
        }
        let bigEndian = encoding == .binaryBE
        try data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            guard let base = raw.baseAddress else { return }
            var offset = start
            for el in elements {
                let fixed = el.properties.allSatisfy { $0.listCount == nil }
                if el.name == "vertex" {
                    guard fixed else { throw FormatError.unsupported("list properties on vertices") }
                    vertexCount = el.count
                    var propOffsets: [Int] = []
                    var stride = 0
                    for p in el.properties { propOffsets.append(stride); stride += p.type.size }
                    guard offset + stride * el.count <= raw.count else { throw FormatError.invalid("file truncated") }
                    for (pi, p) in el.properties.enumerated() where wanted?.contains(p.name) ?? true {
                        var col = [Float](repeating: 0, count: el.count)
                        let po = propOffsets[pi]
                        col.withUnsafeMutableBufferPointer { dst in
                            for r in 0..<el.count {
                                dst[r] = Self.readScalar(base + offset + r * stride + po, p.type, bigEndian)
                            }
                        }
                        vertexColumns[p.name] = col
                    }
                    offset += stride * el.count
                } else if fixed {
                    offset += el.properties.reduce(0) { $0 + $1.type.size } * el.count
                } else {
                    for _ in 0..<el.count {
                        for p in el.properties {
                            if let ct = p.listCount {
                                let n = Int(Self.readScalar(base + offset, ct, bigEndian)); offset += ct.size
                                offset += n * p.type.size
                            } else { offset += p.type.size }
                        }
                    }
                }
            }
        }
    }

    @inline(__always)
    static func readScalar(_ p: UnsafeRawPointer, _ t: ScalarType, _ be: Bool) -> Float {
        switch t {
        case .char, .int8: return Float(p.loadUnaligned(as: Int8.self))
        case .uchar, .uint8: return Float(p.loadUnaligned(as: UInt8.self))
        case .short, .int16: let v = p.loadUnaligned(as: Int16.self); return Float(be ? v.byteSwapped : v)
        case .ushort, .uint16: let v = p.loadUnaligned(as: UInt16.self); return Float(be ? v.byteSwapped : v)
        case .int, .int32: let v = p.loadUnaligned(as: Int32.self); return Float(be ? v.byteSwapped : v)
        case .uint, .uint32: let v = p.loadUnaligned(as: UInt32.self); return Float(be ? v.byteSwapped : v)
        case .float, .float32:
            let v = p.loadUnaligned(as: UInt32.self); return Float(bitPattern: be ? v.byteSwapped : v)
        case .double, .float64:
            let v = p.loadUnaligned(as: UInt64.self); return Float(Double(bitPattern: be ? v.byteSwapped : v))
        }
    }
}

enum PLYCodec {
    /// Returns true if the PLY header describes a Gaussian splat (as opposed to a plain point cloud).
    static func isGaussianPLY(url: URL) -> Bool {
        guard let h = try? FileHandle(forReadingFrom: url), let d = try? h.read(upToCount: 64 * 1024) else { return false }
        try? h.close()
        guard let s = String(data: d, encoding: .ascii) ?? String(data: d.prefix(8000), encoding: .isoLatin1) else { return false }
        return s.contains("f_dc_0") && s.contains("scale_0")
    }

    static func readGaussians(url: URL) throws -> GaussianCloud {
        let ply = try PLYFile(url: url)
        let c = ply.vertexColumns
        let n = ply.vertexCount
        guard let x = c["x"], let y = c["y"], let z = c["z"] else { throw FormatError.invalid("PLY has no x/y/z") }
        guard c["f_dc_0"] != nil, c["scale_0"] != nil else { throw FormatError.invalid("PLY is not a gaussian splat (no f_dc_0/scale_0)") }
        var restCount = 0
        while c["f_rest_\(restCount)"] != nil { restCount += 1 }
        let coeffsPerChannel = restCount / 3
        let degree: Int = coeffsPerChannel >= 15 ? 3 : coeffsPerChannel >= 8 ? 2 : coeffsPerChannel >= 3 ? 1 : 0
        var g = GaussianCloud(count: n, shDegree: degree)
        let s0 = c["scale_0"]!, s1 = c["scale_1"] ?? s0, s2 = c["scale_2"] ?? s0
        let r0 = c["rot_0"], r1 = c["rot_1"], r2 = c["rot_2"], r3 = c["rot_3"]
        let op = c["opacity"]
        let dc0 = c["f_dc_0"]!, dc1 = c["f_dc_1"] ?? dc0, dc2 = c["f_dc_2"] ?? dc0
        let rest: [[Float]] = (0..<restCount).map { c["f_rest_\($0)"]! }
        let usedCoeffs = min(coeffsPerChannel, 15)
        for i in 0..<n {
            g.positions[i*3] = x[i]; g.positions[i*3+1] = y[i]; g.positions[i*3+2] = z[i]
            g.scales[i*3] = s0[i]; g.scales[i*3+1] = s1[i]; g.scales[i*3+2] = s2[i]
            if let r0, let r1, let r2, let r3 {
                g.rotations[i*4] = r0[i]; g.rotations[i*4+1] = r1[i]; g.rotations[i*4+2] = r2[i]; g.rotations[i*4+3] = r3[i]
            }
            g.opacities[i] = op?[i] ?? 10
            let b = i * GaussianCloud.shStride
            g.sh[b] = dc0[i]; g.sh[b+1] = dc1[i]; g.sh[b+2] = dc2[i]
            // f_rest is channel-major: [R coeffs..., G coeffs..., B coeffs...]
            for k in 0..<usedCoeffs {
                for ch in 0..<3 {
                    g.sh[b + (k + 1) * 3 + ch] = rest[ch * coeffsPerChannel + k][i]
                }
            }
        }
        return g
    }

    /// Writes the standard INRIA 3DGS binary PLY (readable by every splat viewer).
    static func writeGaussians(_ g: GaussianCloud, to url: URL, shDegree: Int? = nil, up: SIMD3<Float> = Orientation.standardUp) throws {
        let degree = min(shDegree ?? g.shDegree, 3)
        let restPerChannel = [0, 3, 8, 15][degree]
        var header = "ply\nformat binary_little_endian 1.0\ncomment Generated by Splat Studio\n"
        header += "comment splatstudio_up \(up.x) \(up.y) \(up.z)\n"
        header += "element vertex \(g.count)\n"
        for p in ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"] { header += "property float \(p)\n" }
        for i in 0..<(restPerChannel * 3) { header += "property float f_rest_\(i)\n" }
        for p in ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"] { header += "property float \(p)\n" }
        header += "end_header\n"
        let floatsPer = 9 + restPerChannel * 3 + 8
        var body = [Float](repeating: 0, count: g.count * floatsPer)
        body.withUnsafeMutableBufferPointer { o in
            for i in 0..<g.count {
                var k = i * floatsPer
                o[k] = g.positions[i*3]; o[k+1] = g.positions[i*3+1]; o[k+2] = g.positions[i*3+2]
                k += 6 // normals zero
                let b = i * GaussianCloud.shStride
                o[k] = g.sh[b]; o[k+1] = g.sh[b+1]; o[k+2] = g.sh[b+2]; k += 3
                for ch in 0..<3 { for j in 0..<restPerChannel { o[k] = g.sh[b + (j + 1) * 3 + ch]; k += 1 } }
                o[k] = g.opacities[i]; k += 1
                o[k] = g.scales[i*3]; o[k+1] = g.scales[i*3+1]; o[k+2] = g.scales[i*3+2]; k += 3
                o[k] = g.rotations[i*4]; o[k+1] = g.rotations[i*4+1]; o[k+2] = g.rotations[i*4+2]; o[k+3] = g.rotations[i*4+3]
            }
        }
        var data = Data(header.utf8)
        body.withUnsafeBytes { data.append(contentsOf: $0) }
        try data.write(to: url, options: .atomic)
    }

    static func readPointCloud(url: URL) throws -> PointCloud {
        let ply = try PLYFile(url: url, wantedVertexProperties: ["x", "y", "z", "red", "green", "blue", "r", "g", "b", "f_dc_0", "f_dc_1", "f_dc_2"])
        let c = ply.vertexColumns
        guard let x = c["x"], let y = c["y"], let z = c["z"] else { throw FormatError.invalid("PLY has no x/y/z") }
        var pc = PointCloud()
        pc.points.reserveCapacity(ply.vertexCount)
        let r = c["red"] ?? c["r"], gg = c["green"] ?? c["g"], b = c["blue"] ?? c["b"]
        let dc = c["f_dc_0"] != nil ? (c["f_dc_0"]!, c["f_dc_1"]!, c["f_dc_2"]!) : nil
        for i in 0..<ply.vertexCount {
            pc.points.append(SIMD3(x[i], y[i], z[i]))
            if let r, let gg, let b {
                pc.colors.append(SIMD3(UInt8(clamping: Int(r[i])), UInt8(clamping: Int(gg[i])), UInt8(clamping: Int(b[i]))))
            } else if let dc {
                let col = (SIMD3(dc.0[i], dc.1[i], dc.2[i]) * GaussianCloud.C0 + 0.5) * 255
                pc.colors.append(SIMD3(UInt8(clamping: Int(col.x)), UInt8(clamping: Int(col.y)), UInt8(clamping: Int(col.z))))
            } else { pc.colors.append(SIMD3(128, 128, 128)) }
        }
        return pc
    }

    static func writePointCloud(_ pc: PointCloud, to url: URL) throws {
        let header = "ply\nformat binary_little_endian 1.0\nelement vertex \(pc.count)\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        var data = Data(header.utf8)
        data.reserveCapacity(header.utf8.count + pc.count * 15)
        var rec = [UInt8](repeating: 0, count: 15)
        for i in 0..<pc.count {
            let p = pc.points[i]
            withUnsafeBytes(of: p.x) { for j in 0..<4 { rec[j] = $0[j] } }
            withUnsafeBytes(of: p.y) { for j in 0..<4 { rec[4+j] = $0[j] } }
            withUnsafeBytes(of: p.z) { for j in 0..<4 { rec[8+j] = $0[j] } }
            let c = i < pc.colors.count ? pc.colors[i] : SIMD3(128, 128, 128)
            rec[12] = c.x; rec[13] = c.y; rec[14] = c.z
            data.append(contentsOf: rec)
        }
        try data.write(to: url, options: .atomic)
    }
}
