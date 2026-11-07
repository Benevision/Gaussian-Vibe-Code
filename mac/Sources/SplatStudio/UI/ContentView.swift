import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @Bindable var model: AppModel
    @State private var tool: ViewerTool = .navigate
    @State private var brushRadius: CGFloat = 20
    @State private var dropTargeted = false

    enum SidebarItem: Hashable { case viewer, recent(URL), inbox, workers, models, reference }

    var body: some View {
        NavigationSplitView {
            SidebarView(model: model, selection: $model.sidebar)
                .navigationSplitViewColumnWidth(min: 190, ideal: 220, max: 300)
        } detail: {
            detail
        }
        .environment(\.experienceMode, model.mode)
        .toolbar { toolbar }
        .navigationTitle(model.documentName)
        .navigationSubtitle(subtitle)
        .sheet(isPresented: $model.showNewProject) { NewProjectSheet(model: model).environment(\.experienceMode, model.mode) }
        .alert(item: $model.alert) { a in Alert(title: Text(a.title), message: Text(a.message)) }
        .confirmationDialog("Training is still running", isPresented: $model.confirmHome) {
            Button("Stop training and go home", role: .destructive) { model.goHome(force: true) }
            Button("Keep training", role: .cancel) {}
        } message: { Text("Going home stops the current training run. Checkpoints already saved are kept.") }
        .onDrop(of: [.fileURL], isTargeted: $dropTargeted) { providers in
            for p in providers {
                _ = p.loadObject(ofClass: URL.self) { url, _ in
                    if let url { Task { @MainActor in model.open(url); model.sidebar = .viewer } }
                }
            }
            return true
        }
        .overlay { if dropTargeted { DropOverlay() } }
        .onChange(of: model.viewer.showCameras) { model.updateCameraLines() }
        .onChange(of: model.viewer.cropEnabled) { model.updateCameraLines() }
        .onChange(of: model.viewer.showCropBox) { model.updateCameraLines() }
        .onChange(of: model.viewer.cropMin) { model.updateCameraLines() }
        .onChange(of: model.viewer.cropMax) { model.updateCameraLines() }
    }

    private var subtitle: String {
        guard let s = model.scene else { return model.dataset.map { "\($0.manifest.cameras.count) cameras" } ?? "" }
        var t = "\(formatCount(s.count)) splats"
        if model.deletedCount > 0 { t += " · \(formatCount(model.deletedCount)) deleted" }
        return t
    }

    @ViewBuilder private var detail: some View {
        switch model.sidebar {
        case .inbox: InboxView(model: model, onOpen: { model.sidebar = .viewer })
        case .workers: WorkersView(model: model)
        case .models: ModelsView()
        case .reference: SettingsReferenceView()
        default: viewerWithInspector
        }
    }

    private var hasContent: Bool { model.scene != nil || model.dataset != nil || model.training.isActive }

    private var viewerWithInspector: some View {
        ZStack(alignment: .topLeading) {
            // Clipped to its own column: the Metal layer must never extend under the inspector.
            ViewerView(model: model, tool: tool, brushRadius: brushRadius)
                .frame(minWidth: 380, minHeight: 300)   // never let the layout squeeze the inspector shut
                .clipped()
            if model.scene == nil && !model.isLoading && !model.training.isActive {
                WelcomeView(model: model).transition(.opacity)
            }
            if model.isLoading { ProgressView("Loading…").padding().background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10)).frame(maxWidth: .infinity, maxHeight: .infinity) }
            if model.scene != nil { ViewerHUD(model: model, tool: tool).padding(10) }
            if model.training.isActive { TrainingBanner(session: model.training).padding(10).frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottom) }
            if model.showShortcuts { ShortcutsOverlay { model.showShortcuts = false } }
        }
        // SwiftUI writes `false` back when it auto-collapses the inspector during window resizing / launch;
        // only the toolbar button and menu command (which set showInspector directly) may hide it.
        .inspector(isPresented: Binding(get: { model.showInspector && hasContent }, set: { if $0 { model.showInspector = true } })) {
            InspectorView(model: model, tool: $tool, brushRadius: $brushRadius)
                .inspectorColumnWidth(min: 300, ideal: 340, max: 460)
                .environment(\.experienceMode, model.mode)
        }
    }

    @ToolbarContentBuilder private var toolbar: some ToolbarContent {
        ToolbarItemGroup(placement: .navigation) {
            Button { model.goHome() } label: { Label("Home", systemImage: "house") }
                .help("Back to the start page and unload the current splat to free memory (⇧⌘H)")
                .disabled(!hasContent && model.sidebar == .viewer)
            Button { model.newProjectSource = nil; model.showNewProject = true } label: { Label("New", systemImage: "plus.square.on.square") }
                .help("Create a splat from photos, video, 360° or iPhone captures (⌘N)")
            Button { openPanel() } label: { Label("Open", systemImage: "folder") }.help("Open a splat or dataset (⌘O)")
        }
        ToolbarItemGroup(placement: .principal) {
            Button { model.resetCamera() } label: { Label("Reset view", systemImage: "arrow.counterclockwise.circle") }
                .help("Reset the camera to the starting view (R)")
                .disabled(model.scene == nil)
            Menu {
                Button("Flip upside down") { model.flipUpsideDown() }
                Button("Level to current view") { model.levelToCurrentView() }
                Divider()
                ForEach(OrientationChoice.allCases) { o in Button(o.title) { model.setWorldUp(o.vector) } }
            } label: { Label("Orientation", systemImage: "arrow.up.and.down.righttriangle.up.righttriangle.down") }
                .help("Fix a scene that is upside down or tilted")
                .disabled(model.scene == nil)
            Picker("Tool", selection: $tool) {
                Label("Navigate", systemImage: "rotate.3d").tag(ViewerTool.navigate)
                Label("Select", systemImage: "paintbrush.pointed").tag(ViewerTool.select)
            }
            .pickerStyle(.segmented)
            .help("Navigate: orbit / pan / zoom. Select: brush to select splats (⌥ to deselect).")
        }
        ToolbarItemGroup(placement: .primaryAction) {
            Picker("Mode", selection: $model.mode) {
                ForEach(ExperienceMode.allCases) { Text($0.title).tag($0) }
            }
            .pickerStyle(.segmented)
            .frame(width: 150)
            .help("Easy shows presets and essentials. Advanced reveals every parameter — nothing is ever hidden permanently.")
            Button { model.showInspector.toggle() } label: { Label("Inspector", systemImage: "sidebar.trailing") }
        }
    }

    private func openPanel() {
        let p = NSOpenPanel()
        p.canChooseDirectories = true
        p.allowsMultipleSelection = false
        p.message = "Choose a splat (.ply/.spz/.splat), a dataset folder, photos, a video or an iPhone capture"
        if p.runModal() == .OK, let u = p.url { model.open(u); model.sidebar = .viewer }
    }
}

private struct DropOverlay: View {
    var body: some View {
        RoundedRectangle(cornerRadius: 16)
            .strokeBorder(style: StrokeStyle(lineWidth: 3, dash: [10, 6]))
            .foregroundStyle(.tint)
            .background(.tint.opacity(0.08))
            .overlay(Label("Drop to open or create a splat", systemImage: "square.and.arrow.down").font(.title2.bold()))
            .padding(20)
            .allowsHitTesting(false)
    }
}

// MARK: - Sidebar

struct SidebarView: View {
    let model: AppModel
    @Binding var selection: ContentView.SidebarItem?
    @State private var receiver = CaptureReceiverHolder.shared.receiver
    @State private var workers = WorkerBrowser.shared

    var body: some View {
        List(selection: $selection) {
            Section("Workspace") {
                Label("Viewer", systemImage: "cube.transparent").tag(ContentView.SidebarItem.viewer)
                Label {
                    HStack { Text("iPhone Inbox"); Spacer(); if !receiver.received.isEmpty { Text("\(receiver.received.count)").foregroundStyle(.secondary) } }
                } icon: { Image(systemName: "iphone.radiowaves.left.and.right") }
                .tag(ContentView.SidebarItem.inbox)
                Label {
                    HStack { Text("Compute"); Spacer(); Circle().fill(onlineWorkers > 0 ? .green : .secondary.opacity(0.4)).frame(width: 7, height: 7) }
                } icon: { Image(systemName: "server.rack") }
                .tag(ContentView.SidebarItem.workers)
                Label("Models & Benchmarks", systemImage: "chart.bar.doc.horizontal").tag(ContentView.SidebarItem.models)
                Label("Settings Reference", systemImage: "book").tag(ContentView.SidebarItem.reference)
            }
            Section("Recent") {
                ForEach(model.recents.prefix(12), id: \.self) { url in
                    Label(url.deletingPathExtension().lastPathComponent, systemImage: icon(for: url))
                        .tag(ContentView.SidebarItem.recent(url))
                        .help(url.path)
                }
            }
        }
        .onChange(of: selection) { _, new in
            if case .recent(let url) = new { model.open(url); selection = .viewer }
        }
    }

    private var onlineWorkers: Int {
        workers.workers.filter { if case .online = workers.health[$0.id]?.status { return true }; return false }.count
    }
    private func icon(for url: URL) -> String {
        switch url.pathExtension.lowercased() {
        case "ply", "spz", "splat": return "circle.hexagongrid"
        case "mov", "mp4", "m4v": return "film"
        case "splatcapture": return "iphone"
        default: return "folder"
        }
    }
}

/// Keeps one receiver per process.
@MainActor final class CaptureReceiverHolder {
    static let shared = CaptureReceiverHolder()
    let receiver = CaptureReceiver()
}

// MARK: - Welcome

struct WelcomeView: View {
    let model: AppModel
    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                VStack(spacing: 6) {
                    Image(systemName: "circle.hexagongrid.fill").font(.system(size: 52)).foregroundStyle(.tint)
                    Text("Splat Studio").font(.largeTitle.bold())
                    Text("View, create and edit 3D Gaussian splats — accelerated by Metal on Apple silicon.").foregroundStyle(.secondary)
                }
                VStack(alignment: .leading, spacing: 10) {
                    Text("Create").font(.headline).foregroundStyle(.secondary)
                    HStack(spacing: 12) {
                        ActionCard(title: "Photos", subtitle: "A folder of overlapping photos of an object or place.", systemImage: "photo.stack") { pick(.photos) }
                        ActionCard(title: "Video", subtitle: "Walk slowly around the subject. Frames are picked for sharpness.", systemImage: "video") { pick(.video) }
                        ActionCard(title: "360° video", subtitle: "Equirectangular footage from Insta360, GoPro Max, Theta…", systemImage: "globe") { pick(.video360) }
                    }
                    HStack(spacing: 12) {
                        ActionCard(title: "iPhone LiDAR", subtitle: "A Splat Capture recording: posed frames + depth. Fastest, metric scale.", systemImage: "iphone.gen3", tint: .green) { pick(.capture) }
                        ActionCard(title: "Stereo footage", subtitle: "Side-by-side, top-bottom, spatial video or Arcturus Vision recordings.", systemImage: "eyeglasses", tint: .purple) { pick(.stereo) }
                        ActionCard(title: "COLMAP / Nerfstudio", subtitle: "Existing posed datasets from other tools.", systemImage: "camera.metering.matrix", tint: .orange) { pick(.posed) }
                    }
                }
                .frame(width: 624, alignment: .leading)
                VStack(alignment: .leading, spacing: 10) {
                    Text("View").font(.headline).foregroundStyle(.secondary)
                    HStack(spacing: 12) {
                        ActionCard(title: "Open splat", subtitle: ".ply, .spz or .splat from any 3DGS tool.", systemImage: "cube.transparent", tint: .teal) { pick(.splat) }
                        ActionCard(title: "Open dataset", subtitle: "A prepared Splat Studio dataset to (re)train.", systemImage: "tray.full", tint: .teal) { pick(.dataset) }
                    }
                }
                .frame(width: 624, alignment: .leading)
                Text("Tip: drag anything onto this window. Press ? in the viewer for shortcuts.").font(.callout).foregroundStyle(.secondary)
            }
            .padding(40)
            .frame(maxWidth: .infinity)
        }
        .background(.background)
    }

    enum Kind { case photos, video, video360, capture, stereo, posed, splat, dataset }

    private func pick(_ k: Kind) {
        let p = NSOpenPanel()
        p.canChooseDirectories = [.photos, .capture, .posed, .dataset].contains(k)
        p.canChooseFiles = k != .photos && k != .dataset
        p.allowsMultipleSelection = k == .stereo
        switch k {
        case .video, .video360, .stereo: p.allowedContentTypes = [.movie, .mpeg4Movie, .quickTimeMovie]
        case .splat: p.allowedContentTypes = AppModel.openableTypes
        default: break
        }
        p.message = k == .stereo ? "Choose one stereo video, or the left and right videos" : ""
        guard p.runModal() == .OK, let u = p.url else { return }
        switch k {
        case .splat, .dataset: model.open(u)
        case .photos: model.newProjectSource = .photos(folder: u); model.showNewProject = true
        case .video: model.newProjectSource = .video(u); model.showNewProject = true
        case .video360: model.newProjectSource = .video360(u); model.showNewProject = true
        case .capture: model.newProjectSource = .captureBundle(u); model.showNewProject = true
        case .stereo: model.newProjectSource = .stereoVideo(left: p.urls[0], right: p.urls.count > 1 ? p.urls[1] : nil); model.showNewProject = true
        case .posed: model.newProjectSource = SourceDescription.guess(u) ?? .colmap(project: u, images: nil); model.showNewProject = true
        }
    }
}

// MARK: - HUD / banners / shortcuts

struct ViewerHUD: View {
    let model: AppModel
    let tool: ViewerTool
    var body: some View {
        HStack(spacing: 10) {
            Text(String(format: "%.0f fps", model.stats.fps)).monospacedDigit()
            Divider().frame(height: 12)
            Text(String(format: "GPU %.1f ms", model.stats.gpuMs)).monospacedDigit()
            Divider().frame(height: 12)
            Text("\(formatCount(model.stats.visible)) / \(formatCount(model.stats.total)) visible").monospacedDigit()
            if model.selectedCount > 0 { Divider().frame(height: 12); Text("\(formatCount(model.selectedCount)) selected").foregroundStyle(.orange) }
            if tool == .select { Divider().frame(height: 12); Text("Brush: drag to select, ⌥ deselect, ⌫ delete").foregroundStyle(.secondary) }
        }
        .font(.caption)
        .padding(.horizontal, 10).padding(.vertical, 5)
        .background(.ultraThinMaterial, in: Capsule())
    }
}

struct TrainingBanner: View {
    let session: TrainingSession
    var body: some View {
        HStack(spacing: 12) {
            ProgressView(value: session.fraction).frame(width: 180)
            switch session.state {
            case .preparing(let m): Text(m)
            default:
                Text("Step \(session.status.step)/\(session.status.totalSteps)").monospacedDigit()
                Text(String(format: "PSNR %.1f dB", session.status.evalPSNR ?? session.status.psnr)).monospacedDigit()
                Text("\(formatCount(session.status.numGaussians)) splats").monospacedDigit()
                Text("ETA \(formatDuration(session.status.eta))").monospacedDigit().foregroundStyle(.secondary)
            }
            if session.state == .paused {
                Button("Resume") { session.resume() }
            } else {
                Button("Pause") { session.pause() }.disabled(session.state != .training)
            }
            Button("Stop", role: .destructive) { session.cancel() }
        }
        .font(.callout)
        .padding(.horizontal, 14).padding(.vertical, 8)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
        .shadow(radius: 8, y: 2)
    }
}

struct ShortcutsOverlay: View {
    let close: () -> Void
    private let items: [(String, String)] = [
        ("Drag", "Orbit (Navigate) / paint selection (Select)"), ("⇧ Drag · Right-drag", "Pan"),
        ("Scroll · Pinch", "Zoom"), ("Two-finger drag", "Orbit (trackpad)"), ("Double-click", "Set orbit centre"),
        ("Tab", "Toggle Orbit / Fly"), ("W A S D Q E", "Fly (hold ⇧ for faster)"), ("F", "Frame scene"),
        ("1 – 5", "Colour · Depth · Centres · Ellipses · Opacity"), ("⌫", "Delete selected splats"),
        ("⌘Z / ⇧⌘Z", "Undo / redo edits"), ("⌘N · ⌘O · ⌘E", "New · Open · Export"), ("?", "Show / hide this panel"),
    ]
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack { Text("Keyboard & mouse").font(.title3.bold()); Spacer(); Button("Close", action: close).keyboardShortcut(.cancelAction) }
            Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 6) {
                ForEach(items, id: \.0) { k, v in GridRow { Text(k).font(.callout.monospaced()).foregroundStyle(.secondary); Text(v) } }
            }
        }
        .padding(20)
        .frame(width: 460)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14))
        .shadow(radius: 20)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

/// Axis-aligned "up" choices for the Orientation menu.
enum OrientationChoice: String, CaseIterable, Identifiable {
    case negY, posY, posZ, negZ, posX, negX
    var id: String { rawValue }
    var vector: SIMD3<Float> {
        switch self {
        case .negY: return SIMD3(0, -1, 0)
        case .posY: return SIMD3(0, 1, 0)
        case .posZ: return SIMD3(0, 0, 1)
        case .negZ: return SIMD3(0, 0, -1)
        case .posX: return SIMD3(1, 0, 0)
        case .negX: return SIMD3(-1, 0, 0)
        }
    }
    var title: String {
        switch self {
        case .negY: return "Up is −Y (3DGS / COLMAP, default)"
        case .posY: return "Up is +Y (OpenGL, ARKit, Polycam)"
        case .posZ: return "Up is +Z (Blender, drones, LiDAR)"
        case .negZ: return "Up is −Z"
        case .posX: return "Up is +X"
        case .negX: return "Up is −X"
        }
    }
}
