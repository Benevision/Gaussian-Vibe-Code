import SwiftUI
import AppKit

@main
struct SplatStudioApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate
    @State private var model = AppModel()

    init() {
        CLI.runIfRequested()
    }

    var body: some Scene {
        Window("Splat Studio", id: "main") {
            ContentView(model: model)
                .frame(minWidth: 1080, minHeight: 680)
                .onAppear {
                    delegate.model = model
                    // Files passed on the command line, plus any Finder / Dock open requests that arrived
                    // before the window existed.
                    var urls = CommandLine.arguments.dropFirst().filter { !$0.hasPrefix("-") }.map { URL(fileURLWithPath: $0) }
                    urls += delegate.pendingURLs; delegate.pendingURLs.removeAll()
                    for u in Set(urls.map { $0.standardizedFileURL }) { model.open(u) }
                    UISnapshots.run(model: model)
                    WorkerBrowser.shared.start()
                    if UserDefaults.standard.object(forKey: "receiveFromiPhone") as? Bool ?? true { CaptureReceiverHolder.shared.receiver.start() }
                    CaptureReceiverHolder.shared.receiver.onCapture = { _ in NSSound(named: "Purr")?.play() }
                }
        }
        .defaultLaunchBehavior(.presented)   // also when launched to open a file/folder
        .commands { AppCommands(model: model) }

        Window("Settings Reference", id: "reference") {
            SettingsReferenceView().frame(minWidth: 760, minHeight: 520)
        }
        .keyboardShortcut("/", modifiers: [.command, .shift])

        Settings {
            PreferencesView(model: model)
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    weak var model: AppModel?
    /// Open requests received before the main window (and its model) appeared.
    var pendingURLs: [URL] = []
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
        NSWindow.allowsAutomaticWindowTabbing = false
    }
    func application(_ application: NSApplication, open urls: [URL]) {
        Task { @MainActor in
            if let model { for u in urls { model.open(u) } } else { pendingURLs += urls }
            NSApp.windows.first { $0.canBecomeMain }?.makeKeyAndOrderFront(nil)
        }
    }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

struct AppCommands: Commands {
    let model: AppModel
    @Environment(\.openWindow) private var openWindow

    var body: some Commands {
        CommandGroup(replacing: .newItem) {
            Button("Home") { model.goHome() }.keyboardShortcut("h", modifiers: [.command, .shift])
            Button("New Project…") { model.newProjectSource = nil; model.showNewProject = true }.keyboardShortcut("n")
            Button("Open…") {
                let p = NSOpenPanel(); p.canChooseDirectories = true
                if p.runModal() == .OK, let u = p.url { model.open(u) }
            }.keyboardShortcut("o")
        }
        CommandGroup(after: .saveItem) {
            Button("Export…") { model.inspectorTab = .export; model.showInspector = true }.keyboardShortcut("e")
        }
        CommandGroup(replacing: .undoRedo) {
            Button("Undo Edit") { model.undo() }.keyboardShortcut("z").disabled(!model.canUndo)
            Button("Redo Edit") { model.redo() }.keyboardShortcut("z", modifiers: [.command, .shift]).disabled(!model.canRedo)
        }
        CommandMenu("View") {
            Picker("Display", selection: Binding(get: { model.viewer.displayMode }, set: { model.viewer.displayMode = $0 })) {
                ForEach(ViewerSettings.DisplayMode.allCases) { Text($0.title).tag($0) }
            }
            Divider()
            Button("Frame Scene") { model.frameScene() }.keyboardShortcut("f", modifiers: [.command, .shift])
            Button("Reset View") { model.resetCamera() }.keyboardShortcut("0")
            Button("Flip Upside Down") { model.flipUpsideDown() }.keyboardShortcut("u", modifiers: [.command, .option])
            Button("Level to Current View") { model.levelToCurrentView() }
            Button(model.cameraMode == .orbit ? "Switch to Fly Mode" : "Switch to Orbit Mode") { model.cameraMode = model.cameraMode == .orbit ? .fly : .orbit }
            Toggle("Show Training Cameras", isOn: Binding(get: { model.viewer.showCameras }, set: { model.viewer.showCameras = $0 }))
            Divider()
            Picker("Experience", selection: Binding(get: { model.mode }, set: { model.mode = $0 })) {
                ForEach(ExperienceMode.allCases) { Text($0.title).tag($0) }
            }
            Button("Toggle Inspector") { model.showInspector.toggle() }.keyboardShortcut("i", modifiers: [.command, .option])
        }
        CommandMenu("Edit Splats") {
            Button("Delete Selected") { model.deleteSelected() }.keyboardShortcut(.delete, modifiers: [])
            Button("Invert Selection") { model.invertSelection() }.keyboardShortcut("i")
            Button("Clear Selection") { model.clearSelection() }.keyboardShortcut("d")
            Button("Apply Crop") { model.applyCrop() }
            Button("Restore All") { model.restoreAll() }
        }
        CommandMenu("Training") {
            Button("Start") { if let d = model.dataset { model.training.start(dataset: d) } }.keyboardShortcut("r").disabled(model.dataset == nil || model.training.isActive)
            Button(model.training.state == .paused ? "Resume" : "Pause") {
                model.training.state == .paused ? model.training.resume() : model.training.pause()
            }.keyboardShortcut("p", modifiers: [.command, .shift]).disabled(!model.training.isActive)
            Button("Stop") { model.training.cancel() }.keyboardShortcut(".").disabled(!model.training.isActive)
        }
        CommandGroup(replacing: .help) {
            Button("Settings Reference") { openWindow(id: "reference") }.keyboardShortcut("/", modifiers: [.command, .shift])
            Button("Keyboard Shortcuts") { model.showShortcuts = true }
        }
    }
}

struct PreferencesView: View {
    let model: AppModel
    @AppStorage("receiveFromiPhone") private var receive = true
    var body: some View {
        Form {
            Picker("Experience", selection: Binding(get: { model.mode }, set: { model.mode = $0 })) {
                ForEach(ExperienceMode.allCases) { Text($0.title).tag($0) }
            }
            Text("Easy mode shows presets and essential options. Advanced shows every parameter with full documentation.").font(.caption).foregroundStyle(.secondary)
            Toggle("Receive captures from iPhone at launch", isOn: $receive)
        }
        .padding(20)
        .frame(width: 460)
    }
}
