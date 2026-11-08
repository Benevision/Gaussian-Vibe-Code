import Foundation

/// "These folders of this project already exist on that worker, as dataset X" — written next to `frames.json`
/// when a video was split on a worker. Later steps (masks, camera poses, training) run on the worker's copy and
/// only send what changed, instead of shipping gigabytes of frames back and forth; the frames are downloaded
/// here in the background for anything that needs them on this Mac.
struct RemoteStore: Codable, Hashable, Sendable {
    /// The worker's base URL (`WorkerClient.baseURL.absoluteString`).
    var worker: String
    var datasetId: String
    /// Top-level folders the worker dataset holds for this project (e.g. ["images"]).
    var dirs: [String]
    /// True once those folders are complete here too.
    var local: Bool = false

    static let fileName = ".remote-store.json"

    static func load(_ root: URL) -> RemoteStore? {
        guard let d = try? Data(contentsOf: root.appendingPathComponent(fileName)) else { return nil }
        return try? JSONDecoder().decode(RemoteStore.self, from: d)
    }

    func save(_ root: URL) throws {
        try JSONEncoder().encode(self).write(to: root.appendingPathComponent(Self.fileName), options: .atomic)
    }

    /// The store, if `client` is the worker that holds it.
    static func usable(at root: URL, by client: WorkerClient) -> RemoteStore? {
        guard let s = load(root), s.worker == client.baseURL.absoluteString else { return nil }
        return s
    }
}

/// Background download of a project's frames from the worker that split them (see `RemoteStore`).
/// Anything on this Mac that reads the images calls `ensure(_:)` first; it returns at once when they're here.
actor RemoteImageFetch {
    static let shared = RemoteImageFetch()

    private var tasks: [String: Task<Void, Error>] = [:]
    /// Latest progress per project (0…1), for status displays.
    private(set) var progress: [String: Double] = [:]

    /// Starts fetching `store`'s folders into `root` (no-op if already running or complete).
    func start(root: URL, client: WorkerClient, store: RemoteStore) {
        let key = root.standardizedFileURL.path
        guard tasks[key] == nil, !store.local else { return }
        tasks[key] = Task.detached(priority: .utility) {
            // Unzip next to the project, then move into place, so a half-finished download is never mistaken
            // for the frames.
            let tmp = root.appendingPathComponent(".remote-fetch")
            try? FileManager.default.removeItem(at: tmp)
            defer { try? FileManager.default.removeItem(at: tmp) }
            try await client.downloadDatasetFiles(store.datasetId, only: store.dirs, into: tmp) { f in
                Task { await RemoteImageFetch.shared.setProgress(key, f) }
            }
            let fm = FileManager.default
            for d in store.dirs {
                let src = tmp.appendingPathComponent(d), dst = root.appendingPathComponent(d)
                guard fm.fileExists(atPath: src.path) else { continue }
                if fm.fileExists(atPath: dst.path) {
                    // The worker's copy is the one masks / camera solves were computed from: it replaces any local
                    // file of the same name (e.g. from an earlier split of the video on this Mac).
                    for f in try fm.contentsOfDirectory(atPath: src.path) {
                        let t = dst.appendingPathComponent(f)
                        if fm.fileExists(atPath: t.path) { try fm.removeItem(at: t) }
                        try fm.moveItem(at: src.appendingPathComponent(f), to: t)
                    }
                } else {
                    try fm.moveItem(at: src, to: dst)
                }
            }
            var s = store; s.local = true
            try s.save(root)
        }
    }

    private func setProgress(_ key: String, _ f: Double) { progress[key] = f }

    /// Waits until the project's frames are on this Mac (starting the download if the app was restarted
    /// meanwhile). Returns at once for projects that never had a worker store.
    func ensure(_ root: URL) async throws {
        let key = root.standardizedFileURL.path
        guard let store = RemoteStore.load(root), !store.local else { return }
        if tasks[key] == nil {
            guard let client = await Self.client(for: store) else {
                throw PipelineError.processFailed("This project's frames are still on \(URL(string: store.worker)?.host ?? "the worker"), "
                    + "which isn't available — connect to it (Compute) to download them.")
            }
            start(root: root, client: client, store: store)
        }
        guard let t = tasks[key] else { return }
        do { try await t.value } catch {
            tasks[key] = nil      // allow a retry
            throw PipelineError.processFailed("Couldn't download this project's frames from the worker: \(error.localizedDescription)")
        }
    }

    /// Resumes an unfinished download for a project being opened (fire and forget).
    func resumeIfNeeded(_ root: URL) async {
        guard let store = RemoteStore.load(root), !store.local, tasks[root.standardizedFileURL.path] == nil,
              let client = await Self.client(for: store) else { return }
        start(root: root, client: client, store: store)
    }

    @MainActor private static func client(for store: RemoteStore) -> WorkerClient? {
        WorkerBrowser.shared.client(forBaseURL: store.worker)
    }
}

/// Processing jobs this project sent to workers (`.remote-jobs.json` next to `frames.json`), so a step whose
/// job already ran — or is still running — is picked up again instead of being uploaded and computed twice:
/// after the Mac slept or lost the network during the download, or the app was quit while the worker worked.
/// Keyed by step (`task/outputDir`); an entry is only reused when its fingerprint (solve settings + the exact
/// image list) matches and the job is still on the same worker.
struct RemoteJobLedger: Codable, Sendable {
    struct Entry: Codable, Sendable { var worker: String; var jobId: String; var fingerprint: String; var createdAt: Date }
    var entries: [String: Entry] = [:]

    static let fileName = ".remote-jobs.json"

    static func load(_ root: URL) -> RemoteJobLedger {
        guard let d = try? Data(contentsOf: root.appendingPathComponent(fileName)),
              let l = try? JSONDecoder.iso.decode(RemoteJobLedger.self, from: d) else { return RemoteJobLedger() }
        return l
    }

    func save(_ root: URL) {
        if let d = try? JSONEncoder.iso.encode(self) { try? d.write(to: root.appendingPathComponent(Self.fileName), options: .atomic) }
    }

    static func key(_ r: ProcessingRequest) -> String { "\(r.task.rawValue)/\(r.outputDir)" }

    /// Settings that change the result, plus the sorted image basenames. Reproducible outside the app:
    /// SHA-256 of these lines joined by "\n".
    static func fingerprint(_ r: ProcessingRequest, images: [String]) -> String {
        var lines = [r.task.rawValue, r.outputDir, r.features ?? "", r.matching ?? "", r.mapper ?? "", r.cameraModel ?? "",
                     r.resizeMax.map(String.init) ?? "", r.maxKeypoints.map(String.init) ?? "",
                     r.masksDir == nil ? "nomasks" : "masks", r.rig?.kind ?? "norig"]
        lines += images.map { ($0 as NSString).lastPathComponent }.sorted()
        return sha256Hex(Data(lines.joined(separator: "\n").utf8))
    }

    static func record(_ root: URL, request: ProcessingRequest, fingerprint: String, worker: String, jobId: String) {
        var l = load(root)
        l.entries[key(request)] = Entry(worker: worker, jobId: jobId, fingerprint: fingerprint, createdAt: Date())
        l.save(root)
    }

    static func lookup(_ root: URL, request: ProcessingRequest, fingerprint: String, worker: String) -> String? {
        guard let e = load(root).entries[key(request)], e.fingerprint == fingerprint, e.worker == worker else { return nil }
        return e.jobId
    }
}

import CryptoKit

func sha256Hex(_ d: Data) -> String { SHA256.hash(data: d).map { String(format: "%02x", $0) }.joined() }

extension JSONEncoder { static var iso: JSONEncoder { let e = JSONEncoder(); e.dateEncodingStrategy = .iso8601; return e } }
extension JSONDecoder { static var iso: JSONDecoder { let d = JSONDecoder(); d.dateDecodingStrategy = .iso8601; return d } }

/// "These frames were split from this video with these settings" (`.frames-source.json` next to `frames.json`).
/// A rerun of the same project reuses the frames instead of splitting the video again — which also keeps the
/// frames identical to the ones earlier masks / camera solves on a worker were computed from.
struct FrameSourceStamp: Codable, Equatable, Sendable {
    var video: String
    var size: Int64
    var modified: Double
    var settings: String

    static let fileName = ".frames-source.json"

    static func make(video u: URL, video vo: VideoExtractionOptions, equirect: Equirect360Options?) -> FrameSourceStamp? {
        guard let a = try? FileManager.default.attributesOfItem(atPath: u.path),
              let size = (a[.size] as? NSNumber)?.int64Value, let mod = a[.modificationDate] as? Date else { return nil }
        let e = JSONEncoder(); e.outputFormatting = .sortedKeys
        let s = [(try? e.encode(vo)), equirect.flatMap { try? e.encode($0) }].compactMap { $0 }
        return FrameSourceStamp(video: u.standardizedFileURL.path, size: size, modified: mod.timeIntervalSince1970.rounded(),
                                settings: sha256Hex(s.reduce(Data(), +)))
    }

    func save(_ root: URL) { if let d = try? JSONEncoder().encode(self) { try? d.write(to: root.appendingPathComponent(Self.fileName), options: .atomic) } }

    /// The earlier frames, when they came from the same video and settings and are still complete
    /// (images here, or kept on the worker this project uses).
    static func reusable(_ root: URL, stamp: FrameSourceStamp?, worker: WorkerClient?) -> FrameSet? {
        guard let stamp, let d = try? Data(contentsOf: root.appendingPathComponent(fileName)),
              let old = try? JSONDecoder().decode(FrameSourceStamp.self, from: d), old == stamp,
              let fs = try? FrameSet.load(root), !fs.frames.isEmpty else { return nil }
        if let w = worker, let s = RemoteStore.usable(at: root, by: w), s.dirs.contains("images") { return fs }
        let fm = FileManager.default
        return fs.frames.allSatisfy { fm.fileExists(atPath: root.appendingPathComponent($0.image).path) } ? fs : nil
    }
}
