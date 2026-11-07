import Foundation
import Observation

struct CaptureItem: Identifiable, Hashable, Sendable {
    var id: URL { url }
    let url: URL
    var name: String
    var createdAt: Date
    var frameCount: Int
    var bytes: Int64
    var hasLiDAR: Bool
    var hasPointCloud: Bool
    var resolution: String?
    var device: String?

    var thumbnailURL: URL { url.appendingPathComponent("thumbnail.jpg") }
}

/// The list of capture bundles in Documents/Captures.
@MainActor
@Observable
final class CaptureLibrary {
    private(set) var items: [CaptureItem] = []
    private(set) var isLoading = false

    var totalBytes: Int64 { items.reduce(0) { $0 + $1.bytes } }

    func reload() async {
        isLoading = true
        let loaded = await Task.detached(priority: .userInitiated) { Self.scan() }.value
        items = loaded
        isLoading = false
    }

    func delete(_ item: CaptureItem) throws {
        try FileManager.default.removeItem(at: item.url)
        items.removeAll { $0.url == item.url }
    }

    /// Renames the bundle folder and the `name` in its manifest.
    func rename(_ item: CaptureItem, to newName: String) throws {
        let trimmed = newName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, trimmed != item.name else { return }
        let (dest, finalName) = CaptureStore.uniqueURL(for: trimmed, excluding: item.url)
        if dest.standardizedFileURL != item.url.standardizedFileURL {
            try FileManager.default.moveItem(at: item.url, to: dest)
        }
        let manifestURL = dest.appendingPathComponent("manifest.json")
        if let data = try? Data(contentsOf: manifestURL),
           var json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            // Edit as a dictionary so unknown keys are preserved untouched.
            json["name"] = finalName
            let out = try JSONSerialization.data(withJSONObject: json, options: [.withoutEscapingSlashes])
            try out.write(to: manifestURL, options: .atomic)
        }
        if let i = items.firstIndex(where: { $0.url == item.url }) {
            var updated = items[i]
            updated = CaptureItem(url: dest, name: finalName, createdAt: updated.createdAt,
                                  frameCount: updated.frameCount, bytes: updated.bytes,
                                  hasLiDAR: updated.hasLiDAR, hasPointCloud: updated.hasPointCloud,
                                  resolution: updated.resolution, device: updated.device)
            items[i] = updated
        }
    }

    func item(at url: URL) -> CaptureItem? { items.first { $0.url.standardizedFileURL == url.standardizedFileURL } }

    nonisolated static func scan() -> [CaptureItem] {
        let fm = FileManager.default
        let dir = CaptureStore.capturesDirectory
        guard let urls = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.creationDateKey],
                                                     options: [.skipsHiddenFiles]) else { return [] }
        var out: [CaptureItem] = []
        for url in urls where url.pathExtension == CaptureStore.bundleExtension {
            let manifestURL = url.appendingPathComponent("manifest.json")
            let summary = (try? Data(contentsOf: manifestURL)).flatMap { try? ManifestCoding.decoder().decode(ManifestSummary.self, from: $0) }
            let created = summary?.createdAt
                ?? (try? url.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? Date.distantPast
            var res: String?
            if let w = summary?.imageWidth, let h = summary?.imageHeight, w > 0 { res = "\(w)×\(h)" }
            out.append(CaptureItem(
                url: url,
                name: summary?.name ?? url.deletingPathExtension().lastPathComponent,
                createdAt: created,
                frameCount: summary?.frames?.count ?? 0,
                bytes: CaptureStore.directorySize(url),
                hasLiDAR: summary?.hasLiDAR ?? false,
                hasPointCloud: fm.fileExists(atPath: url.appendingPathComponent("points.ply").path),
                resolution: res,
                device: summary?.device))
        }
        return out.sorted { $0.createdAt > $1.createdAt }
    }
}
