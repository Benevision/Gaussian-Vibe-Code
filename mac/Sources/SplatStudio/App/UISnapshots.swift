import AppKit
import SwiftUI

/// Developer aid: `SPLAT_UI_SNAPSHOTS=<dir> SplatStudio [file]` renders the main window at several
/// sizes and pages to PNGs (no Screen Recording permission needed — the window draws itself), then quits.
/// Used to check layout and scaling; the Metal viewport itself may appear blank in these images.
@MainActor
enum UISnapshots {
    static var directory: URL? { ProcessInfo.processInfo.environment["SPLAT_UI_SNAPSHOTS"].map { URL(fileURLWithPath: $0) } }

    static func run(model: AppModel) {
        guard let dir = directory else { return }
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        Task { @MainActor in
            try? await Task.sleep(for: .seconds(2.5))
            guard let win = NSApp.windows.first(where: { $0.isVisible && $0.contentView != nil && !($0 is NSPanel) }) else { NSApp.terminate(nil); return }
            var sizes: [(String, CGSize)] = [("min", CGSize(width: 1080, height: 680)), ("laptop", CGSize(width: 1440, height: 860)), ("large", CGSize(width: 1920, height: 1150))]
            if ProcessInfo.processInfo.environment["SPLAT_UI_QUICK"] != nil { sizes = [sizes[0]] }
            let modeNames: [ExperienceMode] = [.easy, .advanced]
            for (sname, size) in sizes {
                win.setContentSize(size)
                for mode in modeNames {
                    model.mode = mode
                    for page in pages(model) {
                        page.apply(model)
                        try? await Task.sleep(for: .seconds(page.wait))
                        snap(win, to: dir.appendingPathComponent("\(sname)-\(mode.rawValue)-\(page.name).png"))
                        if let sheet = win.attachedSheet {
                            snap(sheet, to: dir.appendingPathComponent("\(sname)-\(mode.rawValue)-\(page.name)-sheet.png"))
                        }
                        page.undo(model)
                        try? await Task.sleep(for: .milliseconds(300))
                    }
                }
            }
            // Home: back to the start page with everything unloaded.
            win.setContentSize(sizes[1].1)
            model.mode = .easy
            model.goHome(force: true)
            try? await Task.sleep(for: .milliseconds(800))
            snap(win, to: dir.appendingPathComponent("after-home.png"))
            print("after home: scene \(model.scene == nil ? "unloaded" : "STILL LOADED"), renderer scene \(model.renderer.scene == nil ? "nil" : "SET"), cloud \(model.cloud == nil ? "nil" : "SET")")
            NSApp.terminate(nil)
        }
    }

    struct Page { var name: String; var apply: (AppModel) -> Void; var undo: (AppModel) -> Void = { _ in }; var wait: Double = 0.7 }

    static func pages(_ m: AppModel) -> [Page] {
        var p: [Page] = [Page(name: "viewer") { $0.sidebar = .viewer; $0.inspectorTab = .view }]
        if m.scene != nil {
            p += [Page(name: "edit") { $0.inspectorTab = .edit }, Page(name: "train") { $0.inspectorTab = .train },
                  Page(name: "export") { $0.inspectorTab = .export }, Page(name: "shortcuts", apply: { $0.inspectorTab = .view; $0.showShortcuts = true }, undo: { $0.showShortcuts = false })]
        }
        if let ds = m.dataset {
            p += [Page(name: "maskeditor", apply: { $0.inspectorTab = .train; $0.maskEditor = MaskEditorModel(dataset: ds) }, undo: { $0.maskEditor = nil })]
        }
        p += [Page(name: "wizard-cameras", apply: { $0.newProjectSource = .video(URL(fileURLWithPath: "/tmp/example.mov")); $0.newProjectStartStep = 1; $0.showNewProject = true }, undo: { $0.showNewProject = false }),
              Page(name: "wizard-training", apply: { $0.newProjectSource = .video(URL(fileURLWithPath: "/tmp/example.mov")); $0.newProjectStartStep = 2; $0.showNewProject = true }, undo: { $0.showNewProject = false })]
        if let drone = ProcessInfo.processInfo.environment["SPLAT_UI_DRONE"] {   // a folder of drone photos
            p += [Page(name: "wizard-drone", apply: { $0.newProjectSource = .photos(folder: URL(fileURLWithPath: drone)); $0.newProjectStartStep = 0; $0.showNewProject = true },
                       undo: { $0.showNewProject = false })]
        }
        if let tape = ProcessInfo.processInfo.environment["SPLAT_UI_HOMEVIDEO"] {   // an old home video (analyse it first: --home-video-clips)
            p += [Page(name: "wizard-homevideo", apply: { $0.newProjectSource = .homeVideo(URL(fileURLWithPath: tape)); $0.newProjectStartStep = 0; $0.showNewProject = true },
                       undo: { $0.showNewProject = false }, wait: 8)]
        }
        if let proj = ProcessInfo.processInfo.environment["SPLAT_UI_PROJECT"] {
            p += [Page(name: "project") { $0.sidebar = .project(URL(fileURLWithPath: proj)) }]
        }
        p += [Page(name: "inbox") { $0.sidebar = .inbox }, Page(name: "compute") { $0.sidebar = .workers },
              Page(name: "models") { $0.sidebar = .models }, Page(name: "reference") { $0.sidebar = .reference },
              Page(name: "newproject", apply: { $0.sidebar = .viewer; $0.showNewProject = true }, undo: { $0.showNewProject = false })]
        return p
    }

    static func snap(_ win: NSWindow, to url: URL) {
        guard let v = win.contentView?.superview ?? win.contentView,
              let rep = v.bitmapImageRepForCachingDisplay(in: v.bounds) else { return }
        v.cacheDisplay(in: v.bounds, to: rep)
        try? rep.representation(using: .png, properties: [:])?.write(to: url)
    }
}
