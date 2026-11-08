import Foundation
import Observation
import SplatTrainKit
import UIKit

/// A finished splat: Documents/Splats/<capture name>/splat.ply (+ dataset.json for the viewer's start view).
struct SplatOutput: Identifiable, Hashable, Sendable {
    var folder: URL
    var id: URL { folder }
    var plyURL: URL { folder.appendingPathComponent("splat.ply") }
    var datasetURL: URL { folder }
    var notesURL: URL { folder.appendingPathComponent("notes.txt") }
    var exists: Bool { FileManager.default.fileExists(atPath: plyURL.path) }
    var notes: [String] { ((try? String(contentsOf: notesURL, encoding: .utf8)) ?? "").split(separator: "\n").map(String.init) }

    static var directory: URL {
        let d = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("Splats", isDirectory: true)
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }
    static func `for`(_ capture: URL) -> SplatOutput {
        SplatOutput(folder: directory.appendingPathComponent(capture.deletingPathExtension().lastPathComponent, isDirectory: true))
    }
}

/// Runs "Make splat" for one capture at a time and keeps its state while the user moves between screens.
@MainActor
@Observable
final class SplatJobs {
    enum State: Equatable {
        case idle
        case running(SplatMakeProgress)
        case finished(SplatOutput, splats: Int, seconds: Double)
        case failed(String)
    }

    private(set) var capture: URL?
    private(set) var state: State = .idle
    private(set) var notes: [String] = []
    var options = SplatMakeOptions()
    private var task: Task<Void, Never>?
    /// Whether the running job trains on this iPhone (iOS stops GPU work in the background).
    private var trainsHere = false

    var isRunning: Bool { if case .running = state { return true }; return false }

    func state(for capture: URL) -> State { self.capture?.standardizedFileURL == capture.standardizedFileURL ? state : .idle }

    func start(_ item: CaptureItem, devices: SplatDevices) {
        guard !isRunning else { return }
        capture = item.url
        notes = []
        let opts = options
        trainsHere = opts.deviceIds.isEmpty || opts.deviceIds.contains(SplatDeviceInfo.localId)
            || !devices.remote.contains { opts.deviceIds.contains($0.id) && $0.isOnline }
        state = .running(SplatMakeProgress(stage: "Starting", fraction: 0, message: ""))
        UIApplication.shared.isIdleTimerDisabled = true
        let work = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("MakeSplat/\(UUID().uuidString)", isDirectory: true)
        let out = SplatOutput.for(item.url)
        let captureURL = item.url
        task = Task { [weak self] in
            do {
                let r = try await SplatMaker.make(capture: captureURL, workDir: work, options: opts, devices: devices) { p in
                    Task { @MainActor in
                        guard let self, self.isRunning else { return }
                        // keep the last known fraction for "unknown" (-1) reports
                        var p = p
                        if p.fraction < 0, case .running(let old) = self.state { p.fraction = old.fraction }
                        self.state = .running(p)
                    }
                }
                try Self.save(r, to: out, work: work)
                self?.notes = r.notes
                self?.state = .finished(out, splats: r.splatCount, seconds: r.seconds)
            } catch {
                let cancelled = error is CancellationError || Task.isCancelled
                self?.state = cancelled ? .failed(self?.stopReason ?? "Stopped.") : .failed(error.localizedDescription)
            }
            try? FileManager.default.removeItem(at: work)
            self?.stopReason = nil
            UIApplication.shared.isIdleTimerDisabled = false
        }
    }

    private var stopReason: String?

    func cancel(reason: String = "Stopped.") {
        stopReason = reason
        task?.cancel()
    }

    func reset() { if !isRunning { state = .idle; capture = nil } }

    /// iOS doesn't let a backgrounded app use the GPU: stop on-phone training cleanly instead of crashing into it.
    /// (Jobs on a Mac / PC keep running there; the app picks up their progress when it returns.)
    func appDidEnterBackground() {
        guard isRunning, trainsHere else { return }
        cancel(reason: "Stopped because the app left the screen — iOS doesn't allow GPU training in the background. Keep Splat Capture open (plug the phone in) and try again.")
    }

    private nonisolated static func save(_ r: SplatMakeResult, to out: SplatOutput, work: URL) throws {
        let fm = FileManager.default
        try? fm.removeItem(at: out.folder)
        try fm.createDirectory(at: out.folder, withIntermediateDirectories: true)
        // splat.ply and its skybox side files (splat.sky.png / splat.sky.json)
        for f in (try? fm.contentsOfDirectory(at: r.plyURL.deletingLastPathComponent(), includingPropertiesForKeys: nil)) ?? []
        where f.lastPathComponent.hasPrefix("splat.") {
            try fm.moveItem(at: f, to: out.folder.appendingPathComponent(f.lastPathComponent))
        }
        try? fm.copyItem(at: r.datasetURL.appendingPathComponent("dataset.json"), to: out.folder.appendingPathComponent("dataset.json"))
        try? r.notes.joined(separator: "\n").write(to: out.notesURL, atomically: true, encoding: .utf8)
    }
}
