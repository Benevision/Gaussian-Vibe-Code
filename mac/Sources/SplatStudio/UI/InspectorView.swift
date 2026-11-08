import SwiftUI
import simd
import Charts
import UniformTypeIdentifiers

struct InspectorView: View {
    @Bindable var model: AppModel
    @Binding var tool: ViewerTool
    @Binding var brushRadius: CGFloat

    var body: some View {
        VStack(spacing: 0) {
            Picker("", selection: $model.inspectorTab) {
                ForEach(AppModel.InspectorTab.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented).labelsHidden().padding(10)
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 4) {
                    switch model.inspectorTab {
                    case .view: ViewPanel(model: model)
                    case .edit: EditPanel(model: model, tool: $tool, brushRadius: $brushRadius)
                    case .train: TrainPanel(model: model)
                    case .export: ExportPanel(model: model)
                    }
                }
                .padding(12)
            }
        }
    }
}

// MARK: - View

struct ViewPanel: View {
    @Bindable var model: AppModel
    @State private var fovDeg: Double = 50

    private var currentUpTitle: String {
        OrientationChoice.allCases.first { simd_distance($0.vector, model.worldUp) < 1e-3 }?.title.components(separatedBy: " (").first ?? "Custom (levelled)"
    }

    var body: some View {
        PanelSection(title: "Display", systemImage: "eye") {
            SettingRow(id: "viewer.displayMode") {
                Picker("", selection: $model.viewer.displayMode) {
                    ForEach(ViewerSettings.DisplayMode.allCases) { Text($0.title).tag($0) }
                }.labelsHidden().fixedSize()
            }
            if model.viewer.displayMode == .coverage {
                VStack(alignment: .leading, spacing: 6) {
                    HStack(spacing: 0) {
                        LinearGradient(colors: [Color(red: 0.86, green: 0.16, blue: 0.12), Color(red: 0.96, green: 0.78, blue: 0.12),
                                                Color(red: 0.20, green: 0.72, blue: 0.32)], startPoint: .leading, endPoint: .trailing)
                            .frame(height: 8).clipShape(Capsule())
                    }
                    HStack { Text("Seen by 1–2 photos, or only from far away"); Spacer(); Text("Seen closely by many") }
                        .font(.caption2).foregroundStyle(.secondary)
                    if model.coverageBusy { ProgressView().controlSize(.small) }
                    if let info = model.coverageInfo { Text(info).font(.caption).fixedSize(horizontal: false, vertical: true) }
                    Text("Red areas will stay blurry however long you train: capture them again from closer, from a few more angles.")
                        .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
            }
            SettingRow(id: "viewer.maxSHDegree") {
                Picker("", selection: $model.viewer.maxSHDegree) { ForEach(0...3, id: \.self) { Text("\($0)").tag($0) } }
                    .pickerStyle(.segmented).labelsHidden().frame(width: 140)
            }
            SettingRow(id: "viewer.background") {
                ColorPicker("", selection: Binding(
                    get: { Color(red: Double(model.viewer.background.x), green: Double(model.viewer.background.y), blue: Double(model.viewer.background.z)) },
                    set: { c in
                        let n = NSColor(c).usingColorSpace(.sRGB) ?? .black
                        model.viewer.background = SIMD3(Float(n.redComponent), Float(n.greenComponent), Float(n.blueComponent))
                    }), supportsOpacity: false).labelsHidden()
            }
            SettingRow(id: "viewer.splatScale") { NumberSlider(value: $model.viewer.splatScale.double, range: 0.1...3, width: 110, decimals: 2) }
            SettingRow(id: "viewer.alphaCutoff") { NumberSlider(value: $model.viewer.alphaCutoff.double, range: 0.002...0.5, log: true, width: 110) }
            SettingRow(id: "viewer.showCameras") { Toggle("", isOn: $model.viewer.showCameras).labelsHidden()
                    .disabled(model.dataset == nil && (model.remoteWatch.cameras ?? []).isEmpty)
                    .help(model.dataset == nil && (model.remoteWatch.cameras ?? []).isEmpty
                          ? "No camera positions for this splat: open its project (or a splat from a project's outputs folder)" : "") }
            SettingRow(id: "viewer.antialias") {
                Picker("", selection: Binding(get: { model.viewer.antialiasOverride.map { $0 ? 1 : 0 } ?? -1 },
                                              set: { model.viewer.antialiasOverride = $0 < 0 ? nil : $0 == 1 })) {
                    Text("Automatic").tag(-1); Text("On").tag(1); Text("Off").tag(0)
                }.labelsHidden().fixedSize()
            }
        }
        PanelSection(title: "Navigation", systemImage: "move.3d") {
            HStack {
                Button { model.resetCamera() } label: { Label("Reset view", systemImage: "arrow.counterclockwise") }.help("R")
                Button("Frame all") { model.frameScene() }.help("F — with a project open, jumps to the most central capture viewpoint; otherwise frames the whole splat")
                Menu("Views") {
                    Button("Front") { model.setView("front") }; Button("Back") { model.setView("back") }
                    Button("Left") { model.setView("left") }; Button("Right") { model.setView("right") }; Button("Top") { model.setView("top") }
                }.fixedSize()
            }
            SettingRow(id: "viewer.cameraMode") {
                Picker("", selection: $model.cameraMode) { Text("Orbit").tag(ViewerCamera.Mode.orbit); Text("Fly").tag(ViewerCamera.Mode.fly) }
                    .pickerStyle(.segmented).labelsHidden().frame(width: 140)
            }
            SettingRow(id: "nav.rotateSpeed") { NumberSlider(value: $model.navigation.rotateSpeed.double, range: 0.2...3, width: 110, decimals: 1) }
            SettingRow(id: "nav.panSpeed") { NumberSlider(value: $model.navigation.panSpeed.double, range: 0.2...3, width: 110, decimals: 1) }
            SettingRow(id: "nav.zoomSpeed") { NumberSlider(value: $model.navigation.zoomSpeed.double, range: 0.2...3, width: 110, decimals: 1) }
            SettingRow(id: "nav.flySpeed") { NumberSlider(value: $model.navigation.flySpeed.double, range: 0.1...5, log: true, width: 110, decimals: 1) }
            SettingRow(id: "nav.trackpad") {
                Picker("", selection: $model.navigation.trackpadTwoFinger) { Text("Orbit").tag(NavigationSettings.TwoFinger.orbit); Text("Pan").tag(NavigationSettings.TwoFinger.pan) }
                    .pickerStyle(.segmented).labelsHidden().frame(width: 140)
            }
            SettingRow(id: "nav.invertVertical") { Toggle("", isOn: $model.navigation.invertVertical).labelsHidden() }
            SettingRow(id: "nav.smoothing") { Toggle("", isOn: $model.navigation.smoothing).labelsHidden() }
            SettingRow(id: "viewer.fov") {
                NumberSlider(value: $fovDeg, range: 20...110, width: 110, decimals: 0)
                    .onChange(of: fovDeg) { _, v in var c = model.renderer.camera; c.fovY = Float(v) * .pi / 180; model.renderer.camera = c }
            }
            Button("Keyboard & mouse shortcuts") { model.showShortcuts = true }
        }
        PanelSection(title: "Orientation", systemImage: "arrow.up.and.down.righttriangle.up.righttriangle.down") {
            SettingRow(id: "viewer.orientation") {
                Menu(currentUpTitle) {
                    ForEach(OrientationChoice.allCases) { o in Button(o.title) { model.setWorldUp(o.vector) } }
                }.fixedSize()
            }
            HStack {
                Button { model.flipUpsideDown() } label: { Label("Flip upside down", systemImage: "arrow.up.arrow.down") }
                Button { model.levelToCurrentView() } label: { Label("Level to view", systemImage: "level") }
                    .help("Rotate the view until the scene looks level, then press this: what is up on screen becomes the scene's up.")
            }
        }
        PanelSection(title: "Performance", systemImage: "speedometer") {
            SettingRow(id: "viewer.renderScale") {
                Picker("", selection: $model.viewer.renderScale) { Text("100 %").tag(Float(1)); Text("75 %").tag(Float(0.75)); Text("50 %").tag(Float(0.5)); Text("33 %").tag(Float(0.33)) }
                    .labelsHidden().fixedSize()
            }
            SettingRow(id: "viewer.sortPrecision") { Toggle("", isOn: $model.viewer.sixteenBitSort).labelsHidden() }
            SettingRow(id: "viewer.lazySort") { Toggle("", isOn: $model.viewer.lazySort).labelsHidden() }
            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 3) {
                GridRow { Text("Frame rate").foregroundStyle(.secondary); Text(String(format: "%.0f fps", model.stats.fps)) }
                GridRow { Text("GPU time").foregroundStyle(.secondary); Text(String(format: "%.2f ms", model.stats.gpuMs)) }
                GridRow { Text("Visible").foregroundStyle(.secondary); Text("\(formatCount(model.stats.visible)) of \(formatCount(model.stats.total))") }
                if let c = model.cloud { GridRow { Text("Memory").foregroundStyle(.secondary); Text(formatBytes(c.byteSize)) } }
            }
            .font(.callout).monospacedDigit()
        }
        if let ds = model.dataset {
            PanelSection(title: "Training cameras", systemImage: "camera", subtitle: "\(ds.manifest.cameras.count) cameras — double-click to look through one") {
                List(ds.manifest.cameras) { cam in
                    HStack {
                        Text((cam.image as NSString).lastPathComponent).lineLimit(1)
                        Spacer()
                        if cam.holdout == true { Text("holdout").font(.caption2).foregroundStyle(.orange) }
                        if cam.depth != nil { Image(systemName: "cube.transparent").help("Has depth") }
                    }
                    .contentShape(Rectangle())
                    .onTapGesture(count: 2) { model.lookThrough(cam) }
                }
                .frame(height: 200)
            }
        }
    }
}

// MARK: - Edit

struct EditPanel: View {
    @Bindable var model: AppModel
    @Binding var tool: ViewerTool
    @Binding var brushRadius: CGFloat
    @State private var opacityT: Double = 0.05
    @State private var sizeT: Double = 0.1
    @State private var outlierN = 4
    @State private var outlierR: Double = 0.01

    var body: some View {
        if model.scene == nil {
            Text("Open or train a splat to edit it.").foregroundStyle(.secondary)
        } else {
            HStack {
                Button { model.undo() } label: { Label("Undo", systemImage: "arrow.uturn.backward") }.disabled(!model.canUndo)
                Button { model.redo() } label: { Label("Redo", systemImage: "arrow.uturn.forward") }.disabled(!model.canRedo)
                Spacer()
                Button("Restore all", role: .destructive) { model.restoreAll() }.disabled(model.deletedCount == 0)
            }
            Text("\(formatCount(model.deletedCount)) deleted · \(formatCount(model.selectedCount)) selected").font(.caption).foregroundStyle(.secondary)

            PanelSection(title: "Selection brush", systemImage: "paintbrush.pointed") {
                SettingRow(id: "edit.brush") {
                    Toggle("Brush", isOn: Binding(get: { tool == .select }, set: { tool = $0 ? .select : .navigate })).toggleStyle(.button)
                }
                HStack { Text("Radius"); Slider(value: $brushRadius, in: 4...120); Text("\(Int(brushRadius)) px").monospacedDigit().frame(width: 50) }
                HStack {
                    Button("Delete selected") { model.deleteSelected() }.disabled(model.selectedCount == 0)
                    Button("Invert") { model.invertSelection() }
                    Button("Clear") { model.clearSelection() }.disabled(model.selectedCount == 0)
                }
            }
            PanelSection(title: "Crop box", systemImage: "cube") {
                SettingRow(id: "edit.cropBox") { Toggle("", isOn: $model.viewer.cropEnabled).labelsHidden() }
                if model.viewer.cropEnabled || model.mode == .advanced {
                    axisRow("X", 0); axisRow("Y", 1); axisRow("Z", 2)
                    HStack {
                        Button("Fit to scene") { if let s = model.scene { model.viewer.cropMin = s.boundsMin; model.viewer.cropMax = s.boundsMax } }
                        Spacer()
                        Button("Apply crop") { model.applyCrop() }.buttonStyle(.borderedProminent).disabled(!model.viewer.cropEnabled)
                    }
                }
            }
            PanelSection(title: "Clean up", systemImage: "sparkles") {
                SettingRow(id: "edit.autoClean") {
                    HStack {
                        Button { model.autoClean() } label: { Label("Auto clean", systemImage: "wand.and.stars") }
                            .disabled(model.autoCleanBusy || model.scene == nil)
                        if model.selectedCount > 0 {
                            Button("Delete selected") { model.deleteSelected() }.buttonStyle(.borderedProminent)
                        }
                    }
                }
                SettingRow(id: "edit.autoCleanFewViews") { Toggle("", isOn: $model.autoCleanFewViews).labelsHidden() }
                if model.autoCleanBusy { ProgressView().controlSize(.small) }
                if let info = model.autoCleanInfo { Text(info).font(.caption).fixedSize(horizontal: false, vertical: true) }
                SettingRow(id: "edit.opacityThreshold") {
                    HStack { NumberSlider(value: $opacityT, range: 0.005...0.5, log: true, width: 90); Button("Delete") { model.deleteBelowOpacity(Float(opacityT)) } }
                }
                SettingRow(id: "edit.sizeThreshold") {
                    HStack { NumberSlider(value: $sizeT, range: 0.005...1, log: true, width: 90); Button("Delete") { model.deleteLargerThan(fraction: Float(sizeT)) } }
                }
                SettingRow(id: "edit.outlierRadius") {
                    HStack {
                        Stepper("N \(outlierN)", value: $outlierN, in: 1...30).fixedSize()
                        Button("Remove") { model.removeOutliers(minNeighbours: outlierN, radiusFraction: Float(outlierR)) }
                    }
                }
            }
        }
    }

    private func axisRow(_ name: String, _ a: Int) -> some View {
        let lo = model.scene.map { $0.center[a] - $0.radius * 1.5 } ?? -10
        let hi = model.scene.map { $0.center[a] + $0.radius * 1.5 } ?? 10
        return HStack(spacing: 6) {
            Text(name).frame(width: 14)
            Slider(value: Binding(get: { Double(model.viewer.cropMin[a]) }, set: { model.viewer.cropMin[a] = min(Float($0), model.viewer.cropMax[a]) }), in: Double(lo)...Double(hi))
            Slider(value: Binding(get: { Double(model.viewer.cropMax[a]) }, set: { model.viewer.cropMax[a] = max(Float($0), model.viewer.cropMin[a]) }), in: Double(lo)...Double(hi))
        }
    }
}

// MARK: - Train

struct TrainPanel: View {
    @Bindable var model: AppModel
    @State private var prepPlacement: ComputePlacement = .thisMac

    var body: some View {
        let s = model.training
        if let w = model.remoteWatch.watched { RemoteWatchDashboard(watcher: model.remoteWatch, watch: w) }
        if let ds = model.dataset {
            PanelSection(title: "Dataset", systemImage: "tray.full") {
                Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 3) {
                    GridRow { Text("Folder").foregroundStyle(.secondary); Text(ds.root.deletingLastPathComponent().lastPathComponent).lineLimit(1) }
                    GridRow { Text("Cameras").foregroundStyle(.secondary); Text("\(ds.manifest.cameras.count) (\(ds.manifest.cameras.filter { $0.holdout == true }.count) held out)") }
                    GridRow { Text("Depth").foregroundStyle(.secondary); Text(ds.hasDepth ? "yes — depth supervision available" : "no") }
                    GridRow { Text("Masks").foregroundStyle(.secondary); Text(ds.hasMasks ? "yes" : "no") }
                    GridRow { Text("Scene size").foregroundStyle(.secondary); Text(String(format: "%.2f", ds.manifest.sceneExtent)) }
                }.font(.callout)
                HStack {
                    Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([ds.root]) }
                    Button { model.maskEditor = MaskEditorModel(dataset: ds) } label: { Label("Edit masks…", systemImage: "theatermask.and.paintbrush") }
                        .help("Review and edit which parts of each image training ignores (moving objects, people, the operator)")
                }
            }
            .sheet(item: $model.maskEditor) { m in MaskEditorView(model: m) }
            if s.isActive || s.state == .finished || s.history.count > 0 { TrainingDashboard(session: s) }
            if !s.isActive {
                PanelSection(title: "Settings", systemImage: "gearshape.2") {
                    TrainingPresetPicker(session: s)
                    ComputePicker(session: s, prepPlacement: $prepPlacement, dataset: ds)
                    if model.mode == .advanced { AdvancedTrainingSettings(session: s) }
                }
                Button { s.start(dataset: ds) } label: { Label("Start training", systemImage: "play.fill").frame(maxWidth: .infinity) }
                    .buttonStyle(.borderedProminent).controlSize(.large)
                if case .failed(let e) = s.state { Label(e, systemImage: "xmark.octagon").foregroundStyle(.red).font(.callout) }
                if let m = s.remoteMessage { Text(m).font(.caption).foregroundStyle(.orange) }
            }
        } else {
            VStack(alignment: .leading, spacing: 10) {
                Text("No dataset loaded").font(.headline)
                Text("Create one from photos, video, 360° footage or an iPhone capture, or open an existing dataset folder.").foregroundStyle(.secondary)
                Button("New project…") { model.showNewProject = true }.buttonStyle(.borderedProminent)
            }
        }
    }
}

/// PSNR (train line + holdout points) and splat-count charts — local training and watched worker jobs.
struct TrainingCharts: View {
    let history: [TrainingSession.Sample]
    let evalHistory: [TrainingSession.Sample]
    var body: some View {
        if history.count > 2 {
            Chart {
                ForEach(history) { LineMark(x: .value("Step", $0.step), y: .value("PSNR", $0.psnr)).foregroundStyle(by: .value("Series", "Train")) }
                ForEach(evalHistory) { PointMark(x: .value("Step", $0.step), y: .value("PSNR", $0.psnr)).foregroundStyle(by: .value("Series", "Holdout")) }
            }
            .chartYAxisLabel("PSNR (dB)")
            .frame(height: 140)
            Chart(history) { AreaMark(x: .value("Step", $0.step), y: .value("Splats", $0.gaussians)).foregroundStyle(.tint.opacity(0.4)) }
                .chartYAxisLabel("Splats").frame(height: 90)
        }
    }
}

/// The Train tab while watching a job on a Linux worker: the same numbers and charts as local training.
struct RemoteWatchDashboard: View {
    let watcher: RemoteJobWatcher
    let watch: RemoteJobWatcher.Watched
    var body: some View {
        PanelSection(title: "Watching on \(watch.workerName)", systemImage: "dot.radiowaves.left.and.right") {
            Text(watch.name).font(.callout.weight(.semibold)).lineLimit(2)
            if watch.totalSteps > 0 { ProgressView(value: Double(watch.step), total: Double(max(watch.totalSteps, 1))) }
            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 3) {
                GridRow { Text("Status").foregroundStyle(.secondary); Text(watch.state.capitalized + (watch.message.map { $0 == watch.state ? "" : " — \($0)" } ?? "")).lineLimit(2) }
                GridRow { Text("Step").foregroundStyle(.secondary); Text("\(watch.step) / \(watch.totalSteps)") }
                GridRow { Text("Speed").foregroundStyle(.secondary); Text(watch.stepsPerSecond.map { String(format: "%.1f steps/s", $0) } ?? "–") }
                GridRow { Text("Elapsed / ETA").foregroundStyle(.secondary); Text("\(formatDuration(watch.elapsed ?? 0)) / \(formatDuration(watch.eta ?? 0))") }
                GridRow { Text("Splats").foregroundStyle(.secondary); Text(formatCount(watch.splats)) }
                if let p = watch.psnr { GridRow { Text("Train PSNR").foregroundStyle(.secondary); Text(String(format: "%.2f dB", p)) } }
                if let e = watch.evalPsnr { GridRow { Text("Holdout PSNR").foregroundStyle(.secondary); Text(String(format: "%.2f dB", e)).bold() } }
                if let l = watch.loss { GridRow { Text("Loss").foregroundStyle(.secondary); Text(String(format: "%.4f", l)) } }
                GridRow { Text("Live preview").foregroundStyle(.secondary); Text(watch.previewStep.map { "from step \($0)" } ?? "waiting for the first one") }
            }
            .font(.callout).monospacedDigit()
            TrainingCharts(history: watcher.history, evalHistory: watcher.evalHistory)
        }
    }
}

struct TrainingDashboard: View {
    let session: TrainingSession
    var body: some View {
        PanelSection(title: "Progress", systemImage: "chart.xyaxis.line") {
            ProgressView(value: session.fraction)
            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 3) {
                GridRow { Text("Status").foregroundStyle(.secondary); Text(stateText) }
                GridRow { Text("Step").foregroundStyle(.secondary); Text("\(session.status.step) / \(session.status.totalSteps)") }
                GridRow { Text("Speed").foregroundStyle(.secondary); Text(String(format: "%.0f steps/s", session.status.itersPerSecond)) }
                GridRow { Text("Elapsed / ETA").foregroundStyle(.secondary); Text("\(formatDuration(session.status.elapsed)) / \(formatDuration(session.status.eta))") }
                GridRow { Text("Splats").foregroundStyle(.secondary); Text(formatCount(session.status.numGaussians)) }
                GridRow { Text("Train PSNR").foregroundStyle(.secondary); Text(String(format: "%.2f dB", session.status.psnr)) }
                if let e = session.status.evalPSNR { GridRow { Text("Holdout PSNR").foregroundStyle(.secondary); Text(String(format: "%.2f dB", e)).bold() } }
                GridRow { Text("SH degree").foregroundStyle(.secondary); Text("\(session.status.shDegree)") }
                if session.status.transientFraction > 0.001 {
                    GridRow { Text("Ignored as moving").foregroundStyle(.secondary); Text(String(format: "%.1f %% of pixels", session.status.transientFraction * 100)) }
                }
            }
            .font(.callout).monospacedDigit()
            TrainingCharts(history: session.history, evalHistory: session.evalHistory)
            if session.isActive {
                HStack {
                    if session.state == .paused { Button("Resume") { session.resume() } } else { Button("Pause") { session.pause() }.disabled(session.state != .training) }
                    Button("Save checkpoint…") {
                        let p = NSSavePanel(); p.allowedContentTypes = [UTType(filenameExtension: "ply")!]; p.nameFieldStringValue = "checkpoint-\(session.status.step).ply"
                        if p.runModal() == .OK, let u = p.url { try? session.saveCheckpoint(to: u) }
                    }
                    Spacer()
                    Button("Stop", role: .destructive) { session.cancel() }
                }
                Toggle("Live preview in viewer", isOn: Binding(get: { session.livePreview }, set: { session.livePreview = $0 })).font(.callout)
            }
            if session.state == .finished, let u = session.outputURL {
                Label("Saved \(u.lastPathComponent)", systemImage: "checkmark.circle.fill").foregroundStyle(.green)
                Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([u]) }
            }
        }
    }
    private var stateText: String {
        switch session.state {
        case .idle: return "Idle"
        case .preparing(let m): return m
        case .training: return session.status.phase
        case .paused: return "Paused"
        case .finishing: return "Finishing"
        case .finished: return "Finished"
        case .failed(let e): return "Failed: \(e)"
        case .cancelled: return "Cancelled"
        }
    }
}

struct AdvancedTrainingSettings: View {
    @Bindable var session: TrainingSession
    var body: some View {
        // Only a real change counts as customising (fields write back unchanged values when they appear).
        let c = Binding(get: { session.config }, set: { v in
            guard v != session.config else { return }
            session.config = v; session.customised = true
        })
        Group {
            Text("Basics").font(.subheadline.bold()).padding(.top, 6)
            SettingRow(id: "train.iterations", showSummary: false) { IntField(value: c.iterations, range: 500...200_000, step: 1000) }
            SettingRow(id: "train.maxImageSize", showSummary: false) { IntField(value: c.maxImageSize, range: 256...8192, step: 200) }
            SettingRow(id: "train.shDegree", showSummary: false) { Picker("", selection: c.shDegree) { ForEach(0...3, id: \.self) { Text("\($0)").tag($0) } }.pickerStyle(.segmented).labelsHidden().frame(width: 140) }
            SettingRow(id: "train.shDegreeInterval", showSummary: false) { IntField(value: c.shDegreeInterval, range: 100...10_000, step: 100) }
            SettingRow(id: "train.background", showSummary: false) {
                Picker("", selection: c.background) { ForEach(TrainingConfig.Background.allCases, id: \.self) { Text($0.rawValue.capitalized).tag($0) } }.labelsHidden().fixedSize()
            }
            SettingRow(id: "train.antialiased", showSummary: false) { Toggle("", isOn: c.antialiased).labelsHidden() }
            SettingRow(id: "train.seed", showSummary: false) { IntField(value: c.seed, range: 0...999_999) }
        }
        Group {
            Text("Densification").font(.subheadline.bold()).padding(.top, 6)
            SettingRow(id: "train.strategy", showSummary: false) {
                Picker("", selection: c.strategy) { Text("MCMC").tag(TrainingConfig.Strategy.mcmc); Text("Adaptive").tag(TrainingConfig.Strategy.adaptive) }.pickerStyle(.segmented).labelsHidden().frame(width: 160)
            }
            SettingRow(id: "train.maxGaussians", showSummary: false) { IntField(value: c.maxGaussians, range: 10_000...10_000_000, step: 100_000) }
            SettingRow(id: "train.densifyStart", showSummary: false) { IntField(value: c.densifyStart, range: 0...50_000, step: 100) }
            SettingRow(id: "train.densifyStop", showSummary: false) { IntField(value: c.densifyStop, range: 0...100_000, step: 500) }
            SettingRow(id: "train.densifyInterval", showSummary: false) { IntField(value: c.densifyInterval, range: 10...1000, step: 10) }
            if session.config.strategy == .adaptive {
                SettingRow(id: "train.gradThreshold", showSummary: false) { NumberSlider(value: c.densifyGradThreshold.double, range: 0.00005...0.002, log: true, width: 90) }
                SettingRow(id: "train.absGrad", showSummary: false) { Toggle("", isOn: c.absGrad).labelsHidden() }
                SettingRow(id: "train.opacityReset", showSummary: false) { IntField(value: c.opacityResetInterval, range: 0...20_000, step: 500) }
            }
            SettingRow(id: "train.pruneOpacity", showSummary: false) { NumberSlider(value: c.pruneOpacity.double, range: 0.001...0.1, log: true, width: 90) }
            SettingRow(id: "train.initialOpacity", showSummary: false) { NumberSlider(value: c.initialOpacity.double, range: 0.01...0.9, width: 90) }
            SettingRow(id: "train.randomInitCount", showSummary: false) { IntField(value: c.randomInitCount, range: 1000...2_000_000, step: 10_000) }
        }
        Group {
            Text("Learning rates").font(.subheadline.bold()).padding(.top, 6)
            SettingRow(id: "train.lrPosition", showSummary: false) {
                VStack(alignment: .trailing) {
                    NumberSlider(value: c.lrPosition.double, range: 1e-6...1e-2, log: true, width: 90)
                    NumberSlider(value: c.lrPositionFinal.double, range: 1e-8...1e-3, log: true, width: 90)
                }
            }
            SettingRow(id: "train.lrColor", showSummary: false) {
                VStack(alignment: .trailing) {
                    NumberSlider(value: c.lrColor.double, range: 1e-4...1e-1, log: true, width: 90)
                    NumberSlider(value: c.lrSHRest.double, range: 1e-6...1e-2, log: true, width: 90)
                }
            }
            SettingRow(id: "train.lrOpacity", showSummary: false) { NumberSlider(value: c.lrOpacity.double, range: 1e-3...0.5, log: true, width: 90) }
            SettingRow(id: "train.lrScale", showSummary: false) { NumberSlider(value: c.lrScale.double, range: 1e-4...0.1, log: true, width: 90) }
            SettingRow(id: "train.lrRotation", showSummary: false) { NumberSlider(value: c.lrRotation.double, range: 1e-5...0.05, log: true, width: 90) }
        }
        Group {
            Text("Losses & regularisation").font(.subheadline.bold()).padding(.top, 6)
            SettingRow(id: "train.ssimWeight", showSummary: false) { NumberSlider(value: c.ssimWeight.double, range: 0...1, width: 90) }
            SettingRow(id: "train.depthWeight", showSummary: false) { NumberSlider(value: c.depthWeight.double, range: 0...1, width: 90) }
            SettingRow(id: "train.depthEdgeAware", showSummary: false) { Toggle("", isOn: c.depthEdgeAware).labelsHidden() }
            SettingRow(id: "train.anisotropyReg", showSummary: false) { NumberSlider(value: c.anisotropyReg.double, range: 0...1, width: 90) }
            SettingRow(id: "train.bigSplatReg", showSummary: false) { NumberSlider(value: c.bigSplatReg.double, range: 0...30, width: 90, decimals: 1) }
            SettingRow(id: "train.transientMasking", showSummary: false) { Toggle("", isOn: c.transientMasking).labelsHidden() }
            if session.config.transientMasking {
                SettingRow(id: "train.transientSensitivity", showSummary: false) { NumberSlider(value: c.transientSensitivity.double, range: 2...8, width: 90, decimals: 1) }
                SettingRow(id: "train.transientMaxViewGap", showSummary: false) { NumberSlider(value: c.transientMaxViewGap.double, range: 5...45, width: 90, decimals: 0) }
            }
            SettingRow(id: "train.longAxisSplit", showSummary: false) { Toggle("", isOn: c.longAxisSplit).labelsHidden() }
            SettingRow(id: "train.appearanceCorrection", showSummary: false) { Toggle("", isOn: c.appearanceCorrection).labelsHidden() }
            SettingRow(id: "train.bilateralGrid", showSummary: false) { Toggle("", isOn: c.bilateralGrid).labelsHidden() }
            if session.config.appearanceCorrection {
                SettingRow(id: "train.appearanceLR", showSummary: false) { NumberSlider(value: c.appearanceLR.double, range: 0.001...0.1, log: true, width: 90) }
                SettingRow(id: "train.appearanceReg", showSummary: false) { NumberSlider(value: c.appearanceReg.double, range: 0...0.1, width: 90) }
            }
            if session.config.strategy == .mcmc {
                SettingRow(id: "train.opacityReg", showSummary: false) { NumberSlider(value: c.opacityReg.double, range: 0...0.1, width: 90) }
                SettingRow(id: "train.scaleReg", showSummary: false) { NumberSlider(value: c.scaleReg.double, range: 0...0.1, width: 90) }
                SettingRow(id: "train.noiseLR", showSummary: false) { NumberSlider(value: c.noiseLR.double, range: 0...2_000_000, width: 90, decimals: 0) }
            }
            SettingRow(id: "train.evalInterval", showSummary: false) { IntField(value: c.evalInterval, range: 0...20_000, step: 500) }
            HStack {
                Button("Reset to preset") { session.customised = false; session.config = session.preset.config }
                Spacer()
                Button("Copy as JSON") {
                    let e = JSONEncoder(); e.outputFormatting = [.prettyPrinted, .sortedKeys]
                    if let d = try? e.encode(session.config), let s = String(data: d, encoding: .utf8) {
                        NSPasteboard.general.clearContents(); NSPasteboard.general.setString(s, forType: .string)
                    }
                }
            }
        }
    }
}

// MARK: - Export

struct ExportPanel: View {
    @Bindable var model: AppModel
    @State private var format: SplatFileFormat = .ply
    @State private var shDegree = 3
    @State private var onlyKept = true
    @State private var bake = true
    @State private var busy = false
    @State private var message: String?
    @State private var videoProgress: Double? = nil
    @State private var videoSeconds: Double = 8
    @State private var videoRes = 1080

    var body: some View {
        if model.scene == nil {
            Text("Nothing to export yet.").foregroundStyle(.secondary)
        } else {
            PanelSection(title: "Splat file", systemImage: "square.and.arrow.up") {
                SettingRow(id: "export.format") {
                    Picker("", selection: $format) { ForEach(SplatFileFormat.allCases) { Text($0.shortName).tag($0) } }.pickerStyle(.segmented).labelsHidden().frame(width: 220)
                }
                if format != .splat {
                    SettingRow(id: "export.shDegree") { Picker("", selection: $shDegree) { ForEach(0...3, id: \.self) { Text("\($0)").tag($0) } }.pickerStyle(.segmented).labelsHidden().frame(width: 140) }
                }
                SettingRow(id: "export.onlyVisible") { Toggle("", isOn: $onlyKept).labelsHidden() }
                if simd_distance(model.worldUp, Orientation.standardUp) > 1e-3 {
                    SettingRow(id: "export.bakeOrientation") { Toggle("", isOn: $bake).labelsHidden() }
                }
                Text("Estimated size: \(formatBytes(estimate))").font(.callout).foregroundStyle(.secondary)
                Button { exportFile() } label: { Label("Export…", systemImage: "square.and.arrow.up").frame(maxWidth: .infinity) }
                    .buttonStyle(.borderedProminent).disabled(busy)
            }
            PanelSection(title: "Image & video", systemImage: "film") {
                Button("Save screenshot (4K)…") {
                    let p = NSSavePanel(); p.allowedContentTypes = [.png]; p.nameFieldStringValue = "\(model.documentName).png"
                    if p.runModal() == .OK, let u = p.url {
                        do { try model.screenshot(to: u, width: 3840, height: 2160); message = "Saved \(u.lastPathComponent)" } catch { message = error.localizedDescription }
                    }
                }
                SettingRow(id: "export.video") {
                    Picker("", selection: $videoRes) { Text("720p").tag(720); Text("1080p").tag(1080); Text("4K").tag(2160) }.labelsHidden().fixedSize()
                }
                HStack { Text("Length"); Slider(value: $videoSeconds, in: 3...30); Text("\(Int(videoSeconds)) s").monospacedDigit() }
                if let p = videoProgress { ProgressView(value: p) }
                Button("Render turntable video…") { renderVideo() }.disabled(videoProgress != nil)
            }
            if let message { Text(message).font(.callout).foregroundStyle(.secondary) }
        }
    }

    private var estimate: Int {
        let n = max((model.scene?.count ?? 0) - (onlyKept ? model.deletedCount : 0), 0)
        // PLY: xyz, normals, dc, opacity, scale, rot = 17 floats + 3 × (SH coefficients − 1).
        // SPZ: 19 quantised bytes + SH bytes, then gzip (≈ 1/3 on trained scenes).
        let rest = [0, 9, 24, 45][shDegree]
        switch format {
        case .ply: return n * (17 + rest) * 4
        case .splat: return n * 32
        case .spz: return n * (19 + rest) / 3
        // Compressed PLY: 16 packed bytes + 1 byte per SH value, plus 72 bytes of bounds per 256 splats.
        case .compressedPly: return n * (16 + rest) + (n + 255) / 256 * 72
        }
    }

    private func exportFile() {
        let p = NSSavePanel()
        p.allowedContentTypes = [UTType(filenameExtension: format.baseExtension) ?? .data]
        p.nameFieldStringValue = "\(model.documentName).\(format.fileExtension)"
        guard p.runModal() == .OK, let u = p.url else { return }
        busy = true
        Task {
            do {
                try await model.export(to: u, format: format, shDegree: shDegree, onlyKept: onlyKept, bakeOrientation: bake)
                message = "Exported \(u.lastPathComponent)"
            } catch { message = "Export failed: \(error.localizedDescription)" }
            busy = false
        }
    }

    private func renderVideo() {
        let p = NSSavePanel(); p.allowedContentTypes = [.mpeg4Movie]; p.nameFieldStringValue = "\(model.documentName).mp4"
        guard p.runModal() == .OK, let u = p.url else { return }
        let h = videoRes, w = h * 16 / 9
        var o = VideoExporter.Options(); o.width = w; o.height = h; o.seconds = videoSeconds
        let r = model.renderer, cam = r.camera, st = model.viewer
        videoProgress = 0
        Task.detached {
            do {
                try VideoExporter.renderTurntable(renderer: r, start: cam, settings: st, options: o, to: u) { f in Task { @MainActor in videoProgress = f } }
                await MainActor.run { videoProgress = nil; message = "Rendered \(u.lastPathComponent)" }
            } catch {
                await MainActor.run { videoProgress = nil; message = "Video failed: \(error.localizedDescription)" }
            }
        }
    }
}
