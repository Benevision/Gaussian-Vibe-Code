import Foundation

/// Minimal streaming zip writer (STORE method — JPEG/HEIC/float16 data barely compresses).
/// Supports Zip64 for archives / entries larger than 4 GB and more than 65 535 entries.
/// CRC and sizes are patched into each local header after the data is streamed, so the
/// output has no data descriptors and is readable by every unzip implementation.
enum ZipWriter {
    enum ZipError: LocalizedError {
        case cannotOpen(String)
        case cancelled
        case fileChanged(String)
        var errorDescription: String? {
            switch self {
            case .cannotOpen(let p): "Could not open \(p) for zipping."
            case .cancelled: "Cancelled."
            case .fileChanged(let n): "\(n) changed while it was being zipped."
            }
        }
    }

    private struct Entry {
        let name: Data
        let crc: UInt32
        let size: UInt64
        let offset: UInt64
        let dosTime: UInt16
        let dosDate: UInt16
    }

    /// Zips `directory` so that it is the single top-level entry of the archive
    /// (e.g. `Kitchen.splatcapture/manifest.json`).
    /// - Parameter progress: called with (bytesDone, bytesTotal) from the calling thread.
    /// - Parameter keepParent: false = the directory's contents at the top level (no folder entry prefix).
    static func zipDirectory(_ directory: URL, to output: URL, keepParent: Bool = true,
                             progress: ((Int64, Int64) -> Void)? = nil) throws {
        let fm = FileManager.default
        let root = directory.lastPathComponent
        var files: [(URL, String, UInt64, Date)] = []
        let keys: [URLResourceKey] = [.isRegularFileKey, .fileSizeKey, .contentModificationDateKey]
        let basePath = directory.standardizedFileURL.resolvingSymlinksInPath().path
        if let e = fm.enumerator(at: directory, includingPropertiesForKeys: keys) {
            for case let url as URL in e {
                let v = try url.resourceValues(forKeys: Set(keys))
                guard v.isRegularFile == true else { continue }
                let full = url.standardizedFileURL.resolvingSymlinksInPath().path
                var rel = full.hasPrefix(basePath) ? String(full.dropFirst(basePath.count)) : url.lastPathComponent
                if rel.hasPrefix("/") { rel.removeFirst() }
                files.append((url, keepParent ? "\(root)/\(rel)" : rel, UInt64(v.fileSize ?? 0), v.contentModificationDate ?? Date()))
            }
        }
        // manifest first is friendlier for streaming readers.
        files.sort { a, b in
            let am = a.1.hasSuffix("/manifest.json"), bm = b.1.hasSuffix("/manifest.json")
            if am != bm { return am }
            return a.1 < b.1
        }
        let total = files.reduce(Int64(0)) { $0 + Int64($1.2) }

        try? fm.removeItem(at: output)
        guard fm.createFile(atPath: output.path, contents: nil) else { throw ZipError.cannotOpen(output.path) }
        let out = try FileHandle(forWritingTo: output)
        defer { try? out.close() }

        var entries: [Entry] = []
        var offset: UInt64 = 0
        var done: Int64 = 0
        let chunkSize = 1 << 20

        for (url, name, size, date) in files {
            if Task.isCancelled { throw ZipError.cancelled }
            let nameData = Data(name.utf8)
            let (dt, dd) = dosDateTime(date)
            let zip64 = size >= 0xFFFF_FFFF
            let headerOffset = offset

            var h = Data()
            h.le32(0x0403_4b50)
            h.le16(zip64 ? 45 : 20)        // version needed
            h.le16(0x0800)                 // UTF-8 names
            h.le16(0)                      // store
            h.le16(dt); h.le16(dd)
            h.le32(0)                      // crc (patched)
            h.le32(zip64 ? 0xFFFF_FFFF : UInt32(size))
            h.le32(zip64 ? 0xFFFF_FFFF : UInt32(size))
            h.le16(UInt16(nameData.count))
            h.le16(zip64 ? 20 : 0)
            h.append(nameData)
            if zip64 {
                h.le16(0x0001); h.le16(16)
                h.le64(size); h.le64(size)
            }
            try out.write(contentsOf: h)
            offset += UInt64(h.count)

            guard let input = try? FileHandle(forReadingFrom: url) else { throw ZipError.cannotOpen(url.path) }
            var crc = ZipCRC32()
            var written: UInt64 = 0
            while true {
                let chunk = try autoreleasepool { try input.read(upToCount: chunkSize) } ?? Data()
                if chunk.isEmpty { break }
                crc.update(chunk)
                try out.write(contentsOf: chunk)
                written += UInt64(chunk.count)
                done += Int64(chunk.count)
                progress?(done, total)
                if Task.isCancelled { try? input.close(); throw ZipError.cancelled }
            }
            try? input.close()
            // A file that changed size while zipping would make the header lie.
            if written != size { throw ZipError.fileChanged(name) }
            offset += written

            // Patch CRC into the local header.
            let crcValue = crc.value
            try out.seek(toOffset: headerOffset + 14)
            var c = Data(); c.le32(crcValue)
            try out.write(contentsOf: c)
            try out.seekToEnd()

            entries.append(Entry(name: nameData, crc: crcValue, size: written, offset: headerOffset,
                                 dosTime: dt, dosDate: dd))
        }

        // Central directory
        let cdStart = offset
        var cd = Data()
        for e in entries {
            let needSize = e.size >= 0xFFFF_FFFF
            let needOffset = e.offset >= 0xFFFF_FFFF
            var extra = Data()
            if needSize || needOffset {
                var payload = Data()
                if needSize { payload.le64(e.size); payload.le64(e.size) }
                if needOffset { payload.le64(e.offset) }
                extra.le16(0x0001); extra.le16(UInt16(payload.count)); extra.append(payload)
            }
            cd.le32(0x0201_4b50)
            cd.le16(0x0300 | 45)           // made by: UNIX, 4.5
            cd.le16(extra.isEmpty ? 20 : 45)
            cd.le16(0x0800)
            cd.le16(0)
            cd.le16(e.dosTime); cd.le16(e.dosDate)
            cd.le32(e.crc)
            cd.le32(needSize ? 0xFFFF_FFFF : UInt32(e.size))
            cd.le32(needSize ? 0xFFFF_FFFF : UInt32(e.size))
            cd.le16(UInt16(e.name.count))
            cd.le16(UInt16(extra.count))
            cd.le16(0)                     // comment
            cd.le16(0)                     // disk
            cd.le16(0)                     // internal attrs
            cd.le32(UInt32(0o100644) << 16) // external attrs: -rw-r--r--
            cd.le32(needOffset ? 0xFFFF_FFFF : UInt32(e.offset))
            cd.append(e.name)
            cd.append(extra)
            if cd.count > 4 << 20 {
                try out.write(contentsOf: cd)
                offset += UInt64(cd.count)
                cd.removeAll(keepingCapacity: true)
            }
        }
        try out.write(contentsOf: cd)
        offset += UInt64(cd.count)
        let cdSize = offset - cdStart
        let count = UInt64(entries.count)

        var tail = Data()
        let needZip64 = count >= 0xFFFF || cdSize >= 0xFFFF_FFFF || cdStart >= 0xFFFF_FFFF
        if needZip64 {
            let z64Offset = offset
            tail.le32(0x0606_4b50)
            tail.le64(44)
            tail.le16(0x0300 | 45); tail.le16(45)
            tail.le32(0); tail.le32(0)
            tail.le64(count); tail.le64(count)
            tail.le64(cdSize); tail.le64(cdStart)
            // locator
            tail.le32(0x0706_4b50)
            tail.le32(0)
            tail.le64(z64Offset)
            tail.le32(1)
        }
        tail.le32(0x0605_4b50)
        tail.le16(0); tail.le16(0)
        tail.le16(count >= 0xFFFF ? 0xFFFF : UInt16(count))
        tail.le16(count >= 0xFFFF ? 0xFFFF : UInt16(count))
        tail.le32(cdSize >= 0xFFFF_FFFF ? 0xFFFF_FFFF : UInt32(cdSize))
        tail.le32(cdStart >= 0xFFFF_FFFF ? 0xFFFF_FFFF : UInt32(cdStart))
        tail.le16(0)
        try out.write(contentsOf: tail)
        try out.synchronize()
    }

    private static func dosDateTime(_ date: Date) -> (UInt16, UInt16) {
        let c = Calendar(identifier: .gregorian).dateComponents([.year, .month, .day, .hour, .minute, .second], from: date)
        let year = max(1980, min(2107, c.year ?? 1980))
        let time = UInt16((c.hour ?? 0) << 11 | (c.minute ?? 0) << 5 | ((c.second ?? 0) / 2))
        let day = UInt16((year - 1980) << 9 | (c.month ?? 1) << 5 | (c.day ?? 1))
        return (time, day)
    }
}

/// CRC-32 (IEEE 802.3), slicing-by-8.
struct ZipCRC32 {
    private static let tables: [[UInt32]] = {
        var t = [[UInt32]](repeating: [UInt32](repeating: 0, count: 256), count: 8)
        for i in 0..<256 {
            var c = UInt32(i)
            for _ in 0..<8 { c = (c & 1) != 0 ? 0xEDB8_8320 ^ (c >> 1) : c >> 1 }
            t[0][i] = c
        }
        for i in 0..<256 {
            for k in 1..<8 {
                let prev = t[k - 1][i]
                t[k][i] = (prev >> 8) ^ t[0][Int(prev & 0xFF)]
            }
        }
        return t
    }()

    private var crc: UInt32 = 0xFFFF_FFFF
    var value: UInt32 { crc ^ 0xFFFF_FFFF }

    mutating func update(_ data: Data) {
        var c = crc
        let t = Self.tables
        t[0].withUnsafeBufferPointer { t0 in
        t[1].withUnsafeBufferPointer { t1 in
        t[2].withUnsafeBufferPointer { t2 in
        t[3].withUnsafeBufferPointer { t3 in
        t[4].withUnsafeBufferPointer { t4 in
        t[5].withUnsafeBufferPointer { t5 in
        t[6].withUnsafeBufferPointer { t6 in
        t[7].withUnsafeBufferPointer { t7 in
            data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                guard let base = raw.baseAddress else { return }
                let n = raw.count
                var i = 0
                while i + 8 <= n {
                    let lo = base.loadUnaligned(fromByteOffset: i, as: UInt32.self).littleEndian ^ c
                    let hi = base.loadUnaligned(fromByteOffset: i + 4, as: UInt32.self).littleEndian
                    c = t7[Int(lo & 0xFF)] ^ t6[Int((lo >> 8) & 0xFF)] ^ t5[Int((lo >> 16) & 0xFF)] ^ t4[Int(lo >> 24)]
                      ^ t3[Int(hi & 0xFF)] ^ t2[Int((hi >> 8) & 0xFF)] ^ t1[Int((hi >> 16) & 0xFF)] ^ t0[Int(hi >> 24)]
                    i += 8
                }
                while i < n {
                    c = t0[Int((c ^ UInt32(base.load(fromByteOffset: i, as: UInt8.self))) & 0xFF)] ^ (c >> 8)
                    i += 1
                }
            }
        }}}}}}}}
        crc = c
    }
}

private extension Data {
    mutating func le16(_ v: UInt16) { var x = v.littleEndian; Swift.withUnsafeBytes(of: &x) { append(contentsOf: $0) } }
    mutating func le32(_ v: UInt32) { var x = v.littleEndian; Swift.withUnsafeBytes(of: &x) { append(contentsOf: $0) } }
    mutating func le64(_ v: UInt64) { var x = v.littleEndian; Swift.withUnsafeBytes(of: &x) { append(contentsOf: $0) } }
}
