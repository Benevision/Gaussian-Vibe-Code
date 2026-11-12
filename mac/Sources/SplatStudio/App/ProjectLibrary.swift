import Foundation
import Observation
import AppKit

/// `<project>/project.json`: what a project was made from, with which settings, and how far its processing and its
/// latest training run got. Written at every change of state, so it is right even after a crash or a force quit:
/// a run recorded as running that no session is running any more was interrupted and can be resumed.
struct ProjectRecord: Codable, Sendable {
    var name: String
    var created = Date()
    var updated = Date()
    /// Nil for projects made before records existed (their dataset can still be trained).
    var source: SourceDescription? = nil
    var options: PrepOptions? = nil
    /// Start training as soon as processing is done.
    var trainAfter = false

    enum RunState: String, Codable, Sendable { case running, stopped, failed, finished }

    struct Prep: Codable, Sendable {
        var state: RunState
        /// The stage that was running (or failed / was stopped).
        var stage: PrepStage? = nil
        var message: String? = nil
        var updated = Date()
    }
    var prep: Prep? = nil

    struct Training: Codable, Sendable {
        var state: RunState
        var started = Date()
        var updated = Date()
        var config: TrainingConfig
        var placement: ComputePlacement
        var deviceIds: [String]? = nil
        var workerId: String? = nil
        /// Where the result goes (path relative to the project).
        var output: String
        var step = 0
        var totalSteps = 0
        var message: String? = nil
        /// This Mac's last checkpoint of a single-piece run (relative path) and its step.
        var checkpoint: String? = nil
        var checkpointStep: Int? = nil
        /// Block-parallel runs: how many blocks (a resumed run splits the same way and reuses finished blocks).
        var blockCount: Int? = nil
        var blocksDone: Int? = nil
    }
    var training: Training? = nil

    static let fileName = "project.json"
    static func url(_ project: URL) -> URL { project.appendingPathComponent(fileName) }

    static func load(_ project: URL) -> ProjectRecord? {
        guard let d = try? Data(contentsOf: url(project)) else { return nil }
        return try? JSONDecoder.iso.decode(ProjectRecord.self, from: d)
    }

    func save(_ project: URL) {
        var r = self; r.updated = Date()
        if let d = try? JSONEncoder.iso.encode(r) { try? d.write(to: Self.url(project), options: .atomic) }
    }
}

/// Every project and splat this Mac has worked on, newest first (`~/Library/Application Support/SplatWork/
/// library.json`). Unlike the system's recent-documents list it includes projects made with New Project and the
/// results of training, and it survives relaunches.
@MainActor
@Observable
final class ProjectLibrary {
    static let shared = ProjectLibrary()

    struct Item: Codable, Hashable { var path: String; var opened: Date }
    /// Project folders (each has `dataset/` and/or `project.json`).
    private(set) var projects: [Item] = []
    /// Splat files outside any project.
    private(set) var files: [Item] = []
    /// Bumped whenever a project's record changes, so views reload it.
    private(set) var revision = 0

    private struct Stored: Codable { var projects: [Item]; var files: [Item] }

    private static var storeURL: URL {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("SplatWork", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("library.json")
    }

    private init() {
        if let d = try? Data(contentsOf: Self.storeURL), let s = try? JSONDecoder.iso.decode(Stored.self, from: d) {
            projects = s.projects; files = s.files
        } else {
            // First run: start from what macOS remembered, and the default projects folder.
            for u in NSDocumentController.shared.recentDocumentURLs.reversed() { note(u, save: false) }
            let def = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("Splat Studio")
            for p in (try? FileManager.default.contentsOfDirectory(at: def, includingPropertiesForKeys: [.contentModificationDateKey])) ?? []
            where Self.isProject(p) && !projects.contains(where: { $0.path == p.path }) {
                let mod = (try? p.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                projects.append(Item(path: p.path, opened: mod))
            }
            projects.sort { $0.opened > $1.opened }
            save()
        }
    }

    private func save() {
        if let d = try? JSONEncoder.iso.encode(Stored(projects: projects, files: files)) { try? d.write(to: Self.storeURL, options: .atomic) }
    }

    static func isProject(_ dir: URL) -> Bool {
        let fm = FileManager.default
        return fm.fileExists(atPath: ProjectRecord.url(dir).path)
            || fm.fileExists(atPath: dir.appendingPathComponent("dataset").appendingPathComponent(PreparedDataset.fileName).path)
            || fm.fileExists(atPath: dir.appendingPathComponent("dataset").appendingPathComponent(".prep").path)
    }

    /// The project a file or folder belongs to: the project folder itself, its `dataset/`, or a splat in `outputs/`.
    static func project(containing url: URL) -> URL? {
        let u = url.standardizedFileURL
        var candidates = [u, u.deletingLastPathComponent()]
        if u.lastPathComponent == PreparedDataset.fileName { candidates.append(u.deletingLastPathComponent().deletingLastPathComponent()) }
        if u.deletingLastPathComponent().lastPathComponent == "outputs" || u.lastPathComponent == "dataset" || u.lastPathComponent == "outputs" {
            candidates.append(u.deletingLastPathComponent().deletingLastPathComponent())
        }
        return candidates.first { isProject($0) && $0.lastPathComponent != "dataset" && $0.lastPathComponent != "outputs" }
    }

    /// Records that `url` (a project, its dataset, a splat) was opened or worked on now.
    func note(_ url: URL, save doSave: Bool = true) {
        let now = Date()
        if let p = Self.project(containing: url) {
            projects.removeAll { $0.path == p.path }
            projects.insert(Item(path: p.path, opened: now), at: 0)
        } else if ["ply", "spz", "splat"].contains(url.pathExtension.lowercased()) {
            files.removeAll { $0.path == url.path }
            files.insert(Item(path: url.path, opened: now), at: 0)
            if files.count > 30 { files.removeLast(files.count - 30) }
        } else { return }
        if doSave { save() }
    }

    /// Changes a project's record (creating it when missing) and moves the project to the top.
    func update(_ project: URL, _ change: (inout ProjectRecord) -> Void) {
        var r = ProjectRecord.load(project) ?? ProjectRecord(name: project.lastPathComponent)
        change(&r)
        r.save(project)
        note(project)
        revision += 1
    }

    /// Takes a project or file off the list (nothing on disk is touched).
    func remove(_ path: String) {
        projects.removeAll { $0.path == path }; files.removeAll { $0.path == path }
        save(); revision += 1
    }

    func refresh() { revision += 1 }
}

/// What the Projects list and the project page show about one project, read from its folder.
struct ProjectSummary: Identifiable {
    var id: String { url.path }
    let url: URL
    let record: ProjectRecord?
    let exists: Bool
    let datasetReady: Bool
    let cameras: Int
    /// Finished splats in `outputs/`, newest first.
    let results: [URL]
    let checkpoints: [(stage: PrepStage, entry: PrepCheckpoints.Entry)]

    var name: String { record?.name ?? url.lastPathComponent }
    var datasetDir: URL { url.appendingPathComponent("dataset", isDirectory: true) }
    var canResumeProcessing: Bool { record?.source != nil && record?.options != nil }

    /// - Parameter full: also read the dataset (camera count); the sidebar skips that.
    init(_ url: URL, full: Bool = true) {
        self.url = url
        let fm = FileManager.default
        exists = fm.fileExists(atPath: url.path)
        record = ProjectRecord.load(url)
        let dsFile = url.appendingPathComponent("dataset").appendingPathComponent(PreparedDataset.fileName)
        let ds = full ? try? PreparedDataset.load(url.appendingPathComponent("dataset")) : nil
        let prepRan = record?.prep.map { $0.state != .finished } ?? false
        // A dataset.json that a run is rebuilding (or stopped rebuilding) isn't ready.
        datasetReady = (full ? ds != nil : fm.fileExists(atPath: dsFile.path)) && !prepRan
        cameras = ds?.manifest.cameras.count ?? 0
        let out = url.appendingPathComponent("outputs")
        results = ((try? fm.contentsOfDirectory(at: out, includingPropertiesForKeys: [.contentModificationDateKey])) ?? [])
            .filter { ["ply", "spz", "splat"].contains($0.pathExtension.lowercased()) && !$0.lastPathComponent.hasPrefix(".") }
            .sorted {
                let a = (try? $0.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                let b = (try? $1.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                return a > b
            }
        checkpoints = PrepCheckpoints(root: url.appendingPathComponent("dataset")).saved
    }
}
