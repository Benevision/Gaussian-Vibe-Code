import Compression
import Foundation

/// Minimal ZIP extractor for dataset uploads from Splat Studio (`ditto -c -k --sequesterRsrc --keepParent`):
/// stored (0) and deflate (8) entries, data descriptors (sizes come from the central directory), ZIP64.
/// Rejects encrypted entries, absolute paths and `..`; skips symlinks, `__MACOSX/` and AppleDouble `._*` files.
enum ZipReader {
    struct Entry: Sendable, Equatable {
        var path: String
        var method: UInt16
        var flags: UInt16
        var crc32: UInt32
        var compressedSize: UInt64
        var uncompressedSize: UInt64
        var localHeaderOffset: UInt64
        var isDirectory: Bool
        var isSymlink: Bool
    }

    enum ZipError: LocalizedError {
        case notAZip, corrupt(String), unsupported(String), unsafePath(String), tooLarge(String)
        var errorDescription: String? {
            switch self {
            case .notAZip: "not a zip archive"
            case .corrupt(let s): "corrupt zip: \(s)"
            case .unsupported(let s): "unsupported zip: \(s)"
            case .unsafePath(let s): "unsafe path in zip: \(s)"
            case .tooLarge(let s): "zip too large: \(s)"
            }
        }
    }

    // MARK: central directory

    static func entries(of url: URL) throws -> [Entry] {
        let fh = try FileHandle(forReadingFrom: url)
        defer { try? fh.close() }
        let size = try fh.seekToEnd()
        guard size >= 22 else { throw ZipError.notAZip }
        let tailLen = min(size, 65_557 + 20)
        try fh.seek(toOffset: size - tailLen)
        let tail = try fh.read(upToCount: Int(tailLen)) ?? Data()
        guard let eocd = lastIndex(of: 0x0605_4b50, in: tail) else { throw ZipError.notAZip }
        var count = UInt64(tail.readLE16(eocd + 10))
        var cdSize = UInt64(tail.readLE32(eocd + 12))
        var cdOffset = UInt64(tail.readLE32(eocd + 16))
        // ZIP64 end of central directory (locator sits right before the classic record)
        if eocd >= 20, tail.readLE32(eocd - 20) == 0x0706_4b50 {
            let z64Offset = tail.readLE64(eocd - 20 + 8)
            try fh.seek(toOffset: z64Offset)
            let z = try fh.read(upToCount: 56) ?? Data()
            guard z.count >= 56, z.readLE32(0) == 0x0606_4b50 else { throw ZipError.corrupt("bad ZIP64 end record") }
            count = z.readLE64(32); cdSize = z.readLE64(40); cdOffset = z.readLE64(48)
        }
        guard cdOffset + cdSize <= size else { throw ZipError.corrupt("central directory out of range") }
        guard cdSize < 512 * 1024 * 1024 else { throw ZipError.tooLarge("central directory") }
        try fh.seek(toOffset: cdOffset)
        let cd = try fh.read(upToCount: Int(cdSize)) ?? Data()
        guard cd.count == Int(cdSize) else { throw ZipError.corrupt("short central directory") }
        var out: [Entry] = []
        out.reserveCapacity(Int(min(count, 1_000_000)))
        var p = 0
        while p + 46 <= cd.count, cd.readLE32(p) == 0x0201_4b50 {
            let flags = cd.readLE16(p + 8), method = cd.readLE16(p + 10)
            let crc = cd.readLE32(p + 16)
            var csize = UInt64(cd.readLE32(p + 20)), usize = UInt64(cd.readLE32(p + 24))
            let nameLen = Int(cd.readLE16(p + 28)), extraLen = Int(cd.readLE16(p + 30)), commentLen = Int(cd.readLE16(p + 32))
            let extAttr = cd.readLE32(p + 38)
            var offset = UInt64(cd.readLE32(p + 42))
            guard p + 46 + nameLen + extraLen + commentLen <= cd.count else { throw ZipError.corrupt("truncated entry") }
            let nameData = cd.subdata(in: (p + 46)..<(p + 46 + nameLen))
            let name = String(data: nameData, encoding: .utf8) ?? String(decoding: nameData, as: UTF8.self)
            // ZIP64 extended information: only the fields whose 32-bit value is saturated, in this order.
            var e = p + 46 + nameLen
            let extraEnd = e + extraLen
            while e + 4 <= extraEnd {
                let id = cd.readLE16(e), len = Int(cd.readLE16(e + 2))
                if id == 0x0001 {
                    var q = e + 4
                    if usize == 0xFFFF_FFFF, q + 8 <= e + 4 + len { usize = cd.readLE64(q); q += 8 }
                    if csize == 0xFFFF_FFFF, q + 8 <= e + 4 + len { csize = cd.readLE64(q); q += 8 }
                    if offset == 0xFFFF_FFFF, q + 8 <= e + 4 + len { offset = cd.readLE64(q); q += 8 }
                }
                e += 4 + len
            }
            let mode = extAttr >> 16
            out.append(Entry(path: name, method: method, flags: flags, crc32: crc, compressedSize: csize, uncompressedSize: usize,
                             localHeaderOffset: offset, isDirectory: name.hasSuffix("/"),
                             isSymlink: (mode & 0xF000) == 0xA000))
            p += 46 + nameLen + extraLen + commentLen
        }
        return out
    }

    /// Whether an entry is Finder / ditto metadata rather than dataset content.
    static func isMetadata(_ path: String) -> Bool {
        let parts = path.split(separator: "/")
        return parts.first == "__MACOSX" || parts.last?.hasPrefix("._") == true || parts.last == ".DS_Store"
    }

    /// Normalised relative path, or throws for absolute paths / `..` components.
    static func safeRelativePath(_ path: String) throws -> String {
        let p = path.replacingOccurrences(of: "\\", with: "/")
        guard !p.hasPrefix("/"), !p.contains("\0") else { throw ZipError.unsafePath(path) }
        let parts = p.split(separator: "/", omittingEmptySubsequences: true).filter { $0 != "." }
        guard !parts.contains("..") else { throw ZipError.unsafePath(path) }
        return parts.joined(separator: "/")
    }

    // MARK: extraction

    /// Extracts `zip` into `directory`. `maxBytes` bounds the total uncompressed size. Returns bytes written.
    @discardableResult
    static func extract(_ zip: URL, to directory: URL, maxBytes: UInt64, isCancelled: () -> Bool = { false }) throws -> UInt64 {
        let list = try entries(of: zip)
        guard !list.isEmpty else { throw ZipError.corrupt("no entries") }
        let total = list.reduce(UInt64(0)) { $0 &+ $1.uncompressedSize }
        guard total <= maxBytes else { throw ZipError.tooLarge("\(total / 1_048_576) MB uncompressed, \(maxBytes / 1_048_576) MB free") }
        let fm = FileManager.default
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)
        let fh = try FileHandle(forReadingFrom: zip)
        defer { try? fh.close() }
        let zipSize = try fh.seekToEnd()
        var written: UInt64 = 0
        for entry in list {
            if isCancelled() { throw CancellationError() }
            if isMetadata(entry.path) || entry.isSymlink { continue }
            let rel = try safeRelativePath(entry.path)
            if rel.isEmpty { continue }
            let dst = directory.appendingPathComponent(rel)
            if entry.isDirectory { try fm.createDirectory(at: dst, withIntermediateDirectories: true); continue }
            guard entry.flags & 1 == 0 else { throw ZipError.unsupported("encrypted entry \(entry.path)") }
            guard entry.method == 0 || entry.method == 8 else { throw ZipError.unsupported("compression method \(entry.method) (\(entry.path))") }
            // local header: the name / extra lengths there may differ from the central directory's
            try fh.seek(toOffset: entry.localHeaderOffset)
            let lh = try fh.read(upToCount: 30) ?? Data()
            guard lh.count == 30, lh.readLE32(0) == 0x0403_4b50 else { throw ZipError.corrupt("bad local header for \(entry.path)") }
            let dataStart = entry.localHeaderOffset + 30 + UInt64(lh.readLE16(26)) + UInt64(lh.readLE16(28))
            guard dataStart + entry.compressedSize <= zipSize else { throw ZipError.corrupt("\(entry.path) runs past the end") }
            try fh.seek(toOffset: dataStart)
            try fm.createDirectory(at: dst.deletingLastPathComponent(), withIntermediateDirectories: true)
            _ = fm.createFile(atPath: dst.path, contents: nil)
            let out = try FileHandle(forWritingTo: dst)
            var crc = ZipCRC32()
            var produced: UInt64 = 0
            let sink: (Data) throws -> Void = { d in
                produced += UInt64(d.count)
                guard produced <= entry.uncompressedSize else { throw ZipError.corrupt("\(entry.path) inflates past its size") }
                crc.update(d)
                try out.write(contentsOf: d)
            }
            do {
                var remaining = entry.compressedSize
                let chunk = 1 << 20
                if entry.method == 0 {
                    while remaining > 0 {
                        let d = try fh.read(upToCount: Int(min(UInt64(chunk), remaining))) ?? Data()
                        guard !d.isEmpty else { throw ZipError.corrupt("short read in \(entry.path)") }
                        remaining -= UInt64(d.count)
                        try sink(d)
                    }
                } else {
                    // Apple's COMPRESSION_ZLIB is raw DEFLATE (RFC 1951) — exactly zip method 8.
                    var sinkError: Error?
                    let filter = try OutputFilter(.decompress, using: .zlib, bufferCapacity: chunk) { d in
                        guard let d, sinkError == nil else { return }
                        do { try sink(d) } catch { sinkError = error }
                    }
                    while remaining > 0 {
                        let d = try fh.read(upToCount: Int(min(UInt64(chunk), remaining))) ?? Data()
                        guard !d.isEmpty else { throw ZipError.corrupt("short read in \(entry.path)") }
                        remaining -= UInt64(d.count)
                        try filter.write(d)
                        if let sinkError { throw sinkError }
                        if isCancelled() { throw CancellationError() }
                    }
                    try filter.finalize()
                    if let sinkError { throw sinkError }
                }
                try out.close()
            } catch {
                try? out.close()
                throw error
            }
            guard produced == entry.uncompressedSize else { throw ZipError.corrupt("\(entry.path): \(produced) of \(entry.uncompressedSize) bytes") }
            guard crc.value == entry.crc32 else { throw ZipError.corrupt("CRC mismatch in \(entry.path)") }
            written += produced
        }
        return written
    }

    private static func lastIndex(of sig: UInt32, in d: Data) -> Int? {
        guard d.count >= 22 else { return nil }
        var i = d.count - 22
        while i >= 0 {
            if d.readLE32(i) == sig { return i }
            i -= 1
        }
        return nil
    }

    /// Where the dataset lives in an extracted upload (same rule as the Linux worker's `find_upload_root`):
    /// the archive root when it has `dataset.json`, else the single top-level folder that has one, else the
    /// single top-level folder.
    static func uploadRoot(_ dir: URL) -> URL {
        let fm = FileManager.default
        func isRoot(_ u: URL) -> Bool { fm.fileExists(atPath: u.appendingPathComponent("dataset.json").path) }
        if isRoot(dir) { return dir }
        let visible = ((try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.isDirectoryKey])) ?? [])
            .filter { !$0.lastPathComponent.hasPrefix(".") && $0.lastPathComponent != "__MACOSX" }
        let dirs = visible.filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true }
        let candidates = dirs.filter(isRoot)
        if candidates.count == 1 { return candidates[0] }
        if visible.count == 1, dirs.count == 1 { return dirs[0] }
        return dir
    }
}

extension Data {
    // ZipCRC32 is shared with ZipWriter.swift.
    /// Little-endian reads at an offset from `startIndex`.
    func readLE16(_ o: Int) -> UInt16 {
        let i = startIndex + o
        return UInt16(self[i]) | UInt16(self[i + 1]) << 8
    }
    func readLE32(_ o: Int) -> UInt32 {
        let i = startIndex + o
        return UInt32(self[i]) | UInt32(self[i + 1]) << 8 | UInt32(self[i + 2]) << 16 | UInt32(self[i + 3]) << 24
    }
    func readLE64(_ o: Int) -> UInt64 { UInt64(readLE32(o)) | UInt64(readLE32(o + 4)) << 32 }
}
