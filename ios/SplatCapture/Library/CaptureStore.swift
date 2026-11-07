import Foundation

/// File-system layout: Documents/Captures/<name>.splatcapture
enum CaptureStore {
    static let bundleExtension = "splatcapture"

    static var capturesDirectory: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dir = docs.appendingPathComponent("Captures", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    static var exportsDirectory: URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("Exports", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    static func defaultName(for date: Date = Date()) -> String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyy-MM-dd HH.mm.ss"
        return "Capture \(f.string(from: date))"
    }

    /// Removes characters that are illegal or awkward in file names.
    static func sanitize(_ name: String) -> String {
        let bad = CharacterSet(charactersIn: "/\\:?%*|\"<>\n\r\t")
        let cleaned = name.components(separatedBy: bad).joined(separator: "-")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmed = cleaned.hasPrefix(".") ? String(cleaned.drop(while: { $0 == "." })) : cleaned
        return trimmed.isEmpty ? "Capture" : String(trimmed.prefix(100))
    }

    /// A non-existing bundle URL for `name` (appends " 2", " 3"… if needed).
    static func uniqueURL(for name: String, excluding: URL? = nil) -> (URL, String) {
        let base = sanitize(name)
        var candidate = base
        var n = 2
        while true {
            let url = capturesDirectory.appendingPathComponent("\(candidate).\(bundleExtension)", isDirectory: true)
            if url.standardizedFileURL == excluding?.standardizedFileURL || !FileManager.default.fileExists(atPath: url.path) {
                return (url, candidate)
            }
            candidate = "\(base) \(n)"
            n += 1
        }
    }

    static func directorySize(_ url: URL) -> Int64 {
        guard let e = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey]) else { return 0 }
        var total: Int64 = 0
        for case let f as URL in e {
            let v = try? f.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
            if v?.isRegularFile == true { total += Int64(v?.fileSize ?? 0) }
        }
        return total
    }

    static var freeSpace: Int64 {
        let v = try? capturesDirectory.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return v?.volumeAvailableCapacityForImportantUsage ?? Int64.max
    }
}

extension Int64 {
    var formattedBytes: String { ByteCountFormatter.string(fromByteCount: self, countStyle: .file) }
}
