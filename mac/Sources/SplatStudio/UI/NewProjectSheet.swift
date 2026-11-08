import SwiftUI

/// Three-step guided flow: Source → Cameras & masks → Training. Easy mode keeps each page to
/// the decisions that matter; Advanced shows every option. Each step explains what happens.
struct NewProjectSheet: View {
    let model: AppModel
    /// Explicit: Swift ≤ 6.3 makes the memberwise initializer private when any stored property is private.
    init(model: AppModel) { self.model = model }
    @Environment(\.experienceMode) private var mode
    @Environment(\.dismiss) private var dismiss
    @State private var source: SourceDescription?
    @State private var reuseOffer: ReuseOffer? = nil
    @State private var pendingRun: (PrepOptions, URL)? = nil
    @State private var reuseFrames = true
    @State private var reuseMasks = true
    @State private var reusePoses = true
    @State private var options = PrepOptions()
    @State private var step = 0
    @State private var projectName = ""
    @State private var projectParent: URL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("Splat Studio", isDirectory: true)
    @State private var trainAfter = true
    @State private var solverKind = SolverKind.automatic
    @State private var featureModel = "superpoint_lightglue"
    @State private var matching = "auto"
    @State private var mapper = "auto"
    @State private var maskKind = MaskKind.none
    @State private var holdout = true   // a few held-out photos give the dashboard an honest quality trend
    /// Reveals every option in this flow even in Easy mode.
    @State private var showAll = false
    @State private var dynamicClasses = MaskingOptions.defaultDynamicClasses.joined(separator: ", ")
    @State private var showTrainingDetails = false
    private var effectiveMode: ExperienceMode { showAll ? .advanced : mode }
    private var prep: PrepSession { model.prep }
    private var python = PythonEnvironment.shared

    enum SolverKind: String, CaseIterable, Identifiable { case automatic, objectCapture, learned, fromCapture; var id: String { rawValue } }
    enum MaskKind: String, CaseIterable, Identifiable { case none, moving, movingAI, subject, people, sam2, sky; var id: String { rawValue } }

    private let steps = ["Source", "Cameras & masks", "Training"]

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            if prep.running || prep.error != nil {
                progressPage
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 14) {
                        switch step {
                        case 0: sourcePage
                        case 1: camerasPage
                        default: trainingPage
                        }
                    }
                    .padding(20)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .environment(\.experienceMode, effectiveMode)
                }
            }
            Divider()
            footer
        }
        .frame(width: 680, height: 620)
        .sheet(item: $reuseOffer) { offer in reuseSheet(offer) }
        .onAppear {
            prep.reset()   // a new project starts on its settings, not on the last run's result
            source = model.newProjectSource
            step = model.newProjectStartStep; model.newProjectStartStep = 0
            if let s = source { projectName = defaultName(s) }
            if mode == .advanced { holdout = true }
            if case .captureBundle = source { solverKind = .fromCapture }
            Task { await python.refresh() }
        }
    }

    // MARK: header / footer

    private var header: some View {
        HStack(spacing: 18) {
            ForEach(Array(steps.enumerated()), id: \.offset) { i, name in
                HStack(spacing: 6) {
                    ZStack {
                        Circle().fill(i <= step ? AnyShapeStyle(.tint) : AnyShapeStyle(.quaternary)).frame(width: 22, height: 22)
                        if i < step { Image(systemName: "checkmark").font(.caption.bold()).foregroundStyle(.white) }
                        else { Text("\(i + 1)").font(.caption.bold()).foregroundStyle(i == step ? .white : .secondary) }
                    }
                    Text(name).font(.callout.weight(i == step ? .semibold : .regular)).foregroundStyle(i == step ? .primary : .secondary)
                }
                .onTapGesture { if source != nil && !prep.running { step = i } }
                if i < steps.count - 1 { Rectangle().fill(.quaternary).frame(width: 30, height: 1) }
            }
            Spacer()
        }
        .padding(.horizontal, 20).padding(.vertical, 14)
    }

    private var footer: some View {
        HStack {
            Button("Cancel") { if prep.running { prep.cancel() } else { dismiss() } }.keyboardShortcut(.cancelAction)
            if !prep.running && prep.error == nil && mode == .easy {
                Toggle("Show all options", isOn: $showAll)
                    .toggleStyle(.switch).controlSize(.small)
                    .help("Reveal every Advanced setting in this flow without switching the whole app to Advanced mode")
                    .padding(.leading, 12)
            }
            Spacer()
            if prep.error != nil {
                Button("Back to settings") { prep.error = nil }
            } else if !prep.running {
                if step > 0 { Button("Back") { step -= 1 } }
                if step < steps.count - 1 {
                    Button("Next") { step += 1 }.keyboardShortcut(.defaultAction).disabled(source == nil)
                } else {
                    Button(trainAfter ? "Create & train" : "Create dataset") { start() }
                        .keyboardShortcut(.defaultAction).disabled(source == nil || projectName.isEmpty)
                }
            }
        }
        .padding(16)
    }

    // MARK: page 1 — source

    @ViewBuilder private var sourcePage: some View {
        Text("What are you starting from?").font(.title2.bold())
        Text("Splat Studio turns any of these into a set of posed images, then trains a Gaussian splat from them.").foregroundStyle(.secondary)
        GroupBox {
            HStack {
                Image(systemName: sourceIcon).font(.title).foregroundStyle(.tint).frame(width: 40)
                VStack(alignment: .leading) {
                    Text(source.map(kindName) ?? "Nothing chosen yet").font(.headline)
                    Text(source?.displayName ?? "Choose a file or folder").foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
                }
                Spacer()
                Menu("Change…") {
                    Button("Photos folder…") { choose(dir: true) { .photos(folder: $0) } }
                    Button("Video…") { choose(dir: false) { .video($0) } }
                    Button("360° video…") { choose(dir: false) { .video360($0) } }
                    Button("360° photos folder…") { choose(dir: true) { .photos360(folder: $0) } }
                    Button("Stereo video…") { choose(dir: false) { .stereoVideo(left: $0, right: nil) } }
                    Button("iPhone / headset capture…") { choose(dir: true, files: true) { .captureBundle($0) } }
                    Button("COLMAP project…") { choose(dir: true) { .colmap(project: $0, images: nil) } }
                    Button("Nerfstudio transforms.json…") { choose(dir: true, files: true) { .nerfstudio(transforms: $0) } }
                }
                .fixedSize()
            }
            .padding(6)
        }
        HStack {
            Text("Project name")
            TextField("Name", text: $projectName).textFieldStyle(.roundedBorder)
        }
        HStack {
            Text("Save in").foregroundStyle(.secondary)
            Text(projectParent.path).lineLimit(1).truncationMode(.head).foregroundStyle(.secondary)
            Button("Choose…") {
                let p = NSOpenPanel(); p.canChooseDirectories = true; p.canChooseFiles = false; p.canCreateDirectories = true
                if p.runModal() == .OK, let u = p.url { projectParent = u }
            }
        }
        .font(.callout)
        if let s = source { sourceOptions(s) }
    }

    @ViewBuilder private func sourceOptions(_ s: SourceDescription) -> some View {
        PanelSection(title: "Preparation", systemImage: "slider.horizontal.3") {
            SettingRow(id: "source.maxImageSize") {
                Picker("", selection: Binding(get: { options.maxImageSize ?? 0 }, set: { options.maxImageSize = $0 == 0 ? nil : $0 })) {
                    Text("1600 px").tag(1600); Text("2400 px").tag(2400); Text("3200 px").tag(3200); Text("Full").tag(0)
                }.labelsHidden().fixedSize().frame(maxWidth: 380, alignment: .trailing)
            }
            switch s {
            case .video, .video360, .stereoVideo:
                SettingRow(id: "source.videoFPS") { NumberSlider(value: $options.video.targetFPS, range: 0.5...15, width: 110, decimals: 1) }
                SettingRow(id: "source.targetFrames") {
                    IntField(value: Binding(get: { options.video.targetFrameCount ?? 0 }, set: { options.video.targetFrameCount = $0 > 0 ? $0 : nil }), range: 0...3000, step: 50)
                }
                SettingRow(id: "source.sharpestInWindow") { Toggle("", isOn: $options.video.pickSharpest).labelsHidden() }
                SettingRow(id: "source.hdr") { Toggle("Tone-map HDR", isOn: $options.video.toneMapHDR).labelsHidden() }
            case .photos, .photoFiles, .photos360:
                SettingRow(id: "source.blurRejection") {
                    HStack { Toggle("", isOn: $options.photos.rejectBlurry).labelsHidden(); NumberSlider(value: $options.photos.blurRelativeThreshold, range: 0...0.9, width: 90) }
                }
            default: EmptyView()
            }
            if case .video360 = s { equirectOptions } else if case .photos360 = s { equirectOptions }
            if case .stereoVideo = s { stereoOptions }
            if case .captureBundle = s {
                Toggle("Use LiDAR depth for supervision", isOn: $options.capture.copyDepth)
                Toggle("Build initial points from depth", isOn: $options.capture.buildPointsFromDepth)
                Toggle("Drop frames with limited tracking", isOn: $options.capture.dropLimitedTracking)
                if effectiveMode == .advanced {
                    Stepper("Use every \(options.capture.frameStride). frame", value: $options.capture.frameStride, in: 1...10)
                    HStack { Text("Max depth"); NumberSlider(value: $options.capture.maxDepth.double, range: 1...20, width: 110, decimals: 1); Text("m") }
                }
            }
            SettingRow(id: "source.holdout") { Toggle("", isOn: $holdout).labelsHidden() }
        }
    }

    @ViewBuilder private var equirectOptions: some View {
        SettingRow(id: "source.360views") {
            Stepper("\(options.equirect.rings.first?.count ?? 8) around", value: Binding(
                get: { options.equirect.rings.first?.count ?? 8 },
                set: { v in if options.equirect.rings.isEmpty { options.equirect.rings = [.init(pitch: 0, count: v)] } else { options.equirect.rings[0].count = v } }), in: 3...16)
        }
        SettingRow(id: "source.360fov") { NumberSlider(value: $options.equirect.fovDegrees.double, range: 60...120, width: 110, decimals: 0) }
        SettingRow(id: "source.360dropNadir") {
            Toggle("", isOn: Binding(get: { !options.equirect.includeNadir }, set: { options.equirect.includeNadir = !$0 })).labelsHidden()
        }
    }

    @ViewBuilder private var stereoOptions: some View {
        SettingRow(id: "source.stereoLayout") {
            Picker("", selection: $options.stereo.layout) {
                Text("Automatic").tag(StereoLayout.auto); Text("Side by side").tag(StereoLayout.sideBySide)
                Text("Top / bottom").tag(StereoLayout.topBottom); Text("Two files").tag(StereoLayout.separateFiles)
                Text("Apple spatial (MV-HEVC)").tag(StereoLayout.mvhevc); Text("Two video tracks").tag(StereoLayout.dualStream)
            }.labelsHidden().fixedSize().frame(maxWidth: 380, alignment: .trailing)
        }
        Picker("Camera", selection: $options.stereo.device) {
            Text("Automatic").tag(StereoDevice.auto); Text("Arcturus Vision (Steam Frame)").tag(StereoDevice.arcturus)
            Text("Apple spatial video").tag(StereoDevice.appleSpatial); Text("Other").tag(StereoDevice.generic)
        }
        SettingRow(id: "source.stereoBaseline") {
            HStack {
                TextField("auto", value: Binding(get: { options.stereo.baseline.map { $0 * 1000 } }, set: { options.stereo.baseline = $0.map { $0 / 1000 } }), format: .number)
                    .textFieldStyle(.roundedBorder).frame(width: 70)
                Text("mm")
            }
        }
        Toggle("Swap left / right eyes", isOn: $options.stereo.swapEyes)
        SettingRow(id: "source.stereoDepth") { Toggle("", isOn: $options.stereo.computeDepth).labelsHidden() }
    }

    // MARK: page 2 — cameras & masks

    @ViewBuilder private var camerasPage: some View {
        Text("Cameras & masks").font(.title2.bold())
        Text("First, the position of every camera is recovered (structure-from-motion). Masks optionally tell training to ignore moving people, sky or background.").foregroundStyle(.secondary)
        PanelSection(title: "Camera solving", systemImage: "camera.metering.center.weighted") {
            SettingRow(id: "sfm.solver") {
                Picker("", selection: $solverKind) {
                    Text("Automatic").tag(SolverKind.automatic)
                    Text("Apple Object Capture").tag(SolverKind.objectCapture)
                    Text("COLMAP/GLOMAP + learned features").tag(SolverKind.learned)
                    if source?.hasPoses == true { Text("Use poses from source").tag(SolverKind.fromCapture) }
                }.labelsHidden().fixedSize().frame(maxWidth: 380, alignment: .trailing)
            }
            if solverKind == .learned {
                if case .ready = python.state {} else {
                    Label("Learned features need the Models pack. Install it from Models & Benchmarks in the sidebar, or use a Linux worker.", systemImage: "exclamationmark.triangle")
                        .font(.callout).foregroundStyle(.orange)
                }
                SettingRow(id: "sfm.features") {
                    Picker("", selection: $featureModel) {
                        ForEach(featureChoices, id: \.0) { Text($0.1).tag($0.0) }
                    }.labelsHidden().fixedSize().frame(maxWidth: 380, alignment: .trailing)
                }
                SettingRow(id: "sfm.matching") {
                    Picker("", selection: $matching) {
                        Text("Automatic").tag("auto"); Text("Sequential").tag("sequential"); Text("Exhaustive").tag("exhaustive"); Text("Retrieval").tag("retrieval")
                    }.labelsHidden().fixedSize().frame(maxWidth: 380, alignment: .trailing)
                }
                SettingRow(id: "sfm.mapper") {
                    Picker("", selection: $mapper) { Text("Automatic").tag("auto"); Text("GLOMAP").tag("glomap"); Text("COLMAP").tag("colmap") }.labelsHidden().fixedSize().frame(maxWidth: 380, alignment: .trailing)
                }
                SettingRow(id: "sfm.cameraModel") {
                    Picker("", selection: $options.sfmCameraModel) {
                        Text("Automatic").tag("auto"); Text("Pinhole").tag("pinhole"); Text("OpenCV").tag("opencv"); Text("Fisheye").tag("fisheye")
                    }.labelsHidden().fixedSize().frame(maxWidth: 380, alignment: .trailing)
                }
            }
            if solverKind == .objectCapture || solverKind == .automatic {
                if effectiveMode == .advanced { Toggle("High feature sensitivity (slower, for low-texture scenes)", isOn: $options.objectCaptureHighSensitivity) }
            }
            if source?.hasPoses == true && solverKind == .fromCapture {
                SettingRow(id: "sfm.refineCapturePoses") {
                    Toggle("", isOn: $options.refineCapturePoses).labelsHidden()
                }
            }
        }
        PanelSection(title: "Masking", systemImage: "person.crop.rectangle") {
            SettingRow(id: "mask.model") {
                Picker("", selection: $maskKind) {
                    Text("None").tag(MaskKind.none)
                    Text("Remove people & pets — quick (Apple Vision, built-in)").tag(MaskKind.moving)
                    Text("Remove moving things — best (Models pack: detection + SAM 2 tracking)").tag(MaskKind.movingAI)
                    Text("Keep main subject (Apple Vision)").tag(MaskKind.subject)
                    Text("Remove people (Apple Vision)").tag(MaskKind.people)
                    Text("Remove sky (Models pack)").tag(MaskKind.sky)
                    Text("Isolate one object — SAM 2 (Models pack; objects, not rooms)").tag(MaskKind.sam2)
                }.labelsHidden().fixedSize().frame(maxWidth: 380, alignment: .trailing)
            }
            Label("Training also ignores moving objects automatically (Ignore moving objects, on by default); masks are for removing them reliably from the start.", systemImage: "info.circle")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            if maskKind == .sam2 {
                Label("Keeps only the object in the middle of the first frame and masks out everything else — for scanning a single item. For rooms and scenes use “Remove moving things”.", systemImage: "exclamationmark.triangle")
                    .font(.caption).foregroundStyle(.orange).fixedSize(horizontal: false, vertical: true)
            }
            if maskKind == .movingAI {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Objects to remove").font(.callout)
                    TextField("person, dog, cat, car…", text: $dynamicClasses).textFieldStyle(.roundedBorder)
                    Text("Comma-separated, plain English — the detector is open-vocabulary (e.g. “person, cat, robot vacuum, shopping cart”). Static things like bags are better removed in the mask editor.")
                        .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
            }
            if maskKind != .none { SettingRow(id: "mask.invert") { Toggle("", isOn: $options.masking.invert).labelsHidden() } }
            SettingRow(id: "mask.staticRegions") { Toggle("", isOn: $options.maskStaticRegions).labelsHidden() }
            SettingRow(id: "mask.sky") { Toggle("", isOn: $options.masking.sky).labelsHidden() }
        }
    }

    private var featureChoices: [(String, String)] {
        [("superpoint_lightglue", "SuperPoint + LightGlue"), ("aliked_lightglue", "ALIKED + LightGlue"), ("disk_lightglue", "DISK + LightGlue"),
         ("sift_lightglue", "SIFT + LightGlue"), ("colmap_sift", "COLMAP SIFT (classic)"), ("loftr", "LoFTR (dense)"),
         ("mast3r", "MASt3R (few images)"), ("vggt", "VGGT (feed-forward)")]
    }

    // MARK: page 3 — training

    /// splatprep models this project will ask the Linux worker for (masks + learned camera solving).
    private var neededWorkerModels: [(id: String, name: String)] {
        var m: [(String, String)] = []
        switch maskKind {
        case .movingAI: m.append(("dynamic", "moving-object masks"))
        case .sam2: m.append(("sam2_auto", "SAM 2"))
        case .sky: m.append(("sky", "sky masks"))
        default: break
        }
        if solverKind == .learned { m.append((featureModel, featureChoices.first { $0.0 == featureModel }?.1 ?? featureModel)) }
        return m
    }

    /// When the PC will do the preparation but lacks a model it needs, offer to install it from here.
    @ViewBuilder private var missingWorkerModelsNotice: some View {
        let browser = WorkerBrowser.shared
        if model.training.placement != .thisMac,
           let id = model.training.workerId ?? browser.workers.first?.id,
           let info = browser.health[id]?.info, (info.capabilities ?? []).contains("process") {
            let have = Set(info.models ?? [])
            let missing = neededWorkerModels.filter { !have.contains($0.id) }
            if !missing.isEmpty {
                WorkerModelInstallView(workerId: id,
                    reason: "\(info.name) doesn't have \(missing.map(\.name).joined(separator: " or ")) yet, so that step would run on this Mac instead.")
            }
        }
    }

    private var isVideoSource: Bool {
        switch source { case .video, .video360: return true; default: return false }
    }

    @ViewBuilder private var trainingPage: some View {
        Text("Training").font(.title2.bold())
        Text("You can change all of this later and retrain from the same dataset.").foregroundStyle(.secondary)
        Toggle("Start training as soon as the dataset is ready", isOn: $trainAfter)
        if trainAfter {
            TrainingPresetPicker(session: model.training)
            ComputePicker(session: model.training, prepPlacement: $options.compute)
            missingWorkerModelsNotice
            if options.compute != .thisMac, isVideoSource {
                SettingRow(id: "net.videoOnWorker") { Toggle("", isOn: $options.videoOnWorker).labelsHidden() }
            }
            DisclosureGroup(isExpanded: $showTrainingDetails) {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Changing any value switches the preset to Custom. Every setting has an ⓘ explanation.")
                        .font(.caption).foregroundStyle(.secondary)
                    AdvancedTrainingSettings(session: model.training)
                }
                .padding(.top, 6)
                .environment(\.experienceMode, .advanced)
            } label: {
                Label("All training settings", systemImage: "slider.horizontal.3").font(.headline)
            }
            .onAppear { if effectiveMode == .advanced { showTrainingDetails = true } }
        }
        GroupBox("Summary") {
            VStack(alignment: .leading, spacing: 4) {
                Text("• Source: \(source.map(kindName) ?? "–") — \(source?.displayName ?? "")")
                Text("• Cameras: \(solverDescription)")
                Text("• Masks: \(maskKind == .none ? "none" : maskKind.rawValue)")
                if trainAfter { Text("• Training: \(model.training.preset.title), \(model.training.config.iterations) steps, up to \(formatCount(model.training.config.maxGaussians)) splats on \(placementName)") }
                Text("• Project folder: \(projectParent.appendingPathComponent(projectName).path)").lineLimit(1).truncationMode(.middle)
            }
            .font(.callout)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var solverDescription: String {
        switch solverKind {
        case .automatic: return "automatic"
        case .objectCapture: return "Apple Object Capture"
        case .learned: return "\(featureChoices.first { $0.0 == featureModel }?.1 ?? featureModel), \(mapper)"
        case .fromCapture: return "poses from source"
        }
    }
    private var placementName: String {
        switch model.training.placement { case .thisMac: return "this Mac"; case .remote: return "the Linux worker"; case .both: return "this Mac + worker" }
    }

    // MARK: progress

    private var progressPage: some View {
        VStack(alignment: .leading, spacing: 14) {
            if let e = prep.error {
                Label("Preparation failed", systemImage: "xmark.octagon.fill").font(.title2.bold()).foregroundStyle(.red)
                Text(e).textSelection(.enabled)
                Text("Common fixes: use more overlapping images, try a different camera solver, or enable masking for moving objects.").foregroundStyle(.secondary)
            } else {
                // Redraw once a second so elapsed time / ETA / heartbeat stay live even during long silent steps.
                TimelineView(.periodic(from: .now, by: 1)) { _ in
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Preparing \(projectName)").font(.title2.bold())
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text("Overall").font(.headline)
                                Spacer()
                                Text("\(Int(prep.overall * 100)) %").font(.headline.monospacedDigit())
                            }
                            ProgressView(value: prep.overall)
                            HStack(spacing: 14) {
                                Label("Elapsed \(formatDuration(prep.elapsed))", systemImage: "clock")
                                if let eta = prep.eta { Label("About \(formatDuration(eta)) left", systemImage: "hourglass") }
                                else { Label("Estimating time left…", systemImage: "hourglass") }
                                Spacer()
                                heartbeat
                            }
                            .font(.callout.monospacedDigit()).foregroundStyle(.secondary)
                        }
                        stageList
                        if prep.thenTrain {
                            Label("Training starts automatically when the dataset is ready — follow it in the Train panel and the banner at the bottom of the viewer.", systemImage: "arrow.turn.down.right")
                                .font(.callout).foregroundStyle(.secondary)
                        }
                    }
                }
            }
            GroupBox("Log") {
                ScrollViewReader { proxy in
                    ScrollView {
                        VStack(alignment: .leading, spacing: 1) {
                            ForEach(Array(prep.log.enumerated()), id: \.offset) { i, l in Text(l).id(i) }
                        }
                        .font(.caption.monospaced()).frame(maxWidth: .infinity, alignment: .leading).textSelection(.enabled)
                    }
                    .onChange(of: prep.log.count) { _, n in if n > 0 { proxy.scrollTo(n - 1, anchor: .bottom) } }
                }
                .frame(maxHeight: .infinity)
            }
        }
        .padding(20)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    private var heartbeat: some View {
        let s = prep.sinceLastUpdate
        let color: Color = s < 15 ? .green : s < 90 ? .orange : .red
        return HStack(spacing: 5) {
            Circle().fill(color).frame(width: 7, height: 7)
            Text(s < 3 ? "working" : "last update \(formatDuration(s)) ago")
        }
        .help("Time since the processing step last reported progress. Some steps (like image alignment) report rarely; orange/red only means it has been quiet, not that it has failed.")
    }

    private var stageList: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(PrepSession.Phase.allCases, id: \.self) { ph in
                let current = prep.phase
                let done = ph < current
                let active = ph == current
                HStack(spacing: 10) {
                    Group {
                        if done { Image(systemName: "checkmark.circle.fill").foregroundStyle(.green) }
                        else if active { ProgressView().controlSize(.small) }
                        else { Image(systemName: "circle").foregroundStyle(.tertiary) }
                    }
                    .frame(width: 18)
                    VStack(alignment: .leading, spacing: 1) {
                        Text(ph.title).font(active ? .callout.bold() : .callout).foregroundStyle(done || active ? .primary : .secondary)
                        if active {
                            Text([prep.progress.stage, prep.progress.message].filter { !$0.isEmpty }.joined(separator: " — "))
                                .font(.caption).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
                        }
                    }
                    Spacer()
                    if let d = prep.stageDuration(ph) { Text(formatDuration(d)).font(.caption.monospacedDigit()).foregroundStyle(.secondary) }
                }
            }
        }
        .padding(10)
        .background(RoundedRectangle(cornerRadius: 8).fill(.quaternary.opacity(0.4)))
    }

    // MARK: actions

    private func start() {
        guard let source else { return }
        var o = options
        // Easy: every 25th photo (4 %), enough for a trustworthy trend at almost no cost; Advanced: every 8th
        // (the research-paper convention, a more precise number).
        o.holdoutInterval = holdout ? (mode == .advanced ? 8 : 25) : nil
        switch solverKind {
        case .automatic: o.poseSolver = .automatic
        case .objectCapture: o.poseSolver = .objectCapture
        case .learned: o.poseSolver = .splatprep(featureModelId: featureModel, matching: matching, mapper: mapper)
        case .fromCapture: o.poseSolver = .fromCapture
        }
        switch maskKind {
        case .none: o.masking.model = .none
        case .moving: o.masking.model = .appleVisionDynamic
        case .movingAI:
            o.masking.model = .splatprep(maskModelId: "dynamic")
            let cls = dynamicClasses.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
            if !cls.isEmpty { o.masking.dynamicClasses = cls }
        case .subject: o.masking.model = .appleVisionForeground
        case .people: o.masking.model = .appleVisionPerson
        case .sam2: o.masking.model = .splatprep(maskModelId: "sam2_auto")
        case .sky: o.masking.model = .splatprep(maskModelId: "sky")
        }
        if model.training.placement != .thisMac { o.remoteWorkerId = model.training.workerId ?? WorkerBrowser.shared.workers.first?.id }
        let dir = projectParent.appendingPathComponent(projectName, isDirectory: true)
        if case .captureBundle(let capture) = source { CaptureProjects.record(capture: capture, project: dir) }
        if let offer = ReuseOffer.find(projectDir: dir) {
            pendingRun = (o, dir); reuseFrames = offer.frames != nil; reuseMasks = offer.masks != nil; reusePoses = offer.poses != nil
            reuseOffer = offer
            return
        }
        prep.run(source: source, options: o, projectDir: dir, thenTrain: trainAfter)
    }

    private func reuseSheet(_ offer: ReuseOffer) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Reuse earlier results?").font(.title2.bold())
            Text("This project has results from an earlier run. Reusing them skips that work (and the upload). Untick anything you changed on purpose — a replaced video, different frame or mask settings.")
                .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            if let f = offer.frames {
                Toggle(isOn: $reuseFrames) { VStack(alignment: .leading) { Text("Frames"); Text(f).font(.caption).foregroundStyle(.secondary) } }
            }
            if let m = offer.masks {
                Toggle(isOn: $reuseMasks) { VStack(alignment: .leading) { Text("Masks"); Text(m).font(.caption).foregroundStyle(.secondary) } }
                    .disabled(!reuseFrames)
            }
            if let p = offer.poses {
                Toggle(isOn: $reusePoses) { VStack(alignment: .leading) { Text("Camera positions"); Text(p).font(.caption).foregroundStyle(.secondary) } }
                    .disabled(!reuseFrames)
            }
            if !reuseFrames && (offer.masks != nil || offer.poses != nil) {
                Text("Masks and camera positions belong to the earlier frames, so they are redone when the frames are.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            HStack {
                Button("Cancel") { reuseOffer = nil; pendingRun = nil }.keyboardShortcut(.cancelAction)
                Spacer()
                Button("Start fresh") { startAfterReuseChoice(fresh: true) }
                Button("Start") { startAfterReuseChoice(fresh: false) }.keyboardShortcut(.defaultAction)
            }
        }
        .padding(20).frame(width: 460)
    }

    private func startAfterReuseChoice(fresh: Bool) {
        guard let offer = reuseOffer, let (o, dir) = pendingRun else { return }
        if fresh { offer.apply(frames: false, masks: false, poses: false) }
        else { offer.apply(frames: reuseFrames, masks: reuseMasks, poses: reusePoses) }
        reuseOffer = nil; pendingRun = nil
        prep.run(source: source!, options: o, projectDir: dir, thenTrain: trainAfter)
    }

    private func choose(dir: Bool, files: Bool = false, make: @escaping (URL) -> SourceDescription) {
        let p = NSOpenPanel()
        p.canChooseDirectories = dir; p.canChooseFiles = !dir || files
        if p.runModal() == .OK, let u = p.url {
            source = make(u)
            // outdoor-capable sources: find the sky by default (skybox); iPhone object captures don't need it
            switch source! { case .video, .video360, .photos, .photoFiles, .photos360: options.masking.sky = true; default: options.masking.sky = false }
            if projectName.isEmpty { projectName = defaultName(source!) }
        }
    }

    private func defaultName(_ s: SourceDescription) -> String {
        (s.displayName as NSString).deletingPathExtension.replacingOccurrences(of: "/", with: "-")
    }

    private func kindName(_ s: SourceDescription) -> String {
        switch s {
        case .photos, .photoFiles: return "Photos"
        case .video: return "Video"
        case .video360: return "360° video"
        case .photos360: return "360° photos"
        case .stereoVideo: return "Stereo footage"
        case .captureBundle: return "iPhone / headset capture"
        case .colmap: return "COLMAP project"
        case .nerfstudio: return "Nerfstudio dataset"
        }
    }

    private var sourceIcon: String {
        switch source {
        case .photos, .photoFiles: return "photo.stack"
        case .video: return "video"
        case .video360, .photos360: return "globe"
        case .stereoVideo: return "eyeglasses"
        case .captureBundle: return "iphone.gen3"
        case .colmap, .nerfstudio: return "camera.metering.matrix"
        case nil: return "questionmark.square.dashed"
        }
    }
}

/// Preset cards + (advanced) a link to the full parameter list.
struct TrainingPresetPicker: View {
    @Bindable var session: TrainingSession
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack { Text("Quality").font(.headline); HelpButton(id: "train.preset") }
            // Adaptive grid: 4 across in the wide New Project sheet, 2×2 in the narrow inspector.
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 138, maximum: 200), spacing: 8)], alignment: .leading, spacing: 8) {
                ForEach(TrainingConfig.Preset.allCases) { p in
                    Button { session.customised = false; session.preset = p } label: {
                        VStack(alignment: .leading, spacing: 3) {
                            Text(p.title).font(.callout.bold())
                            Text(p.subtitle).font(.caption2).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                        }
                        .padding(8).frame(maxWidth: .infinity, minHeight: 74, alignment: .topLeading)
                        .background(RoundedRectangle(cornerRadius: 8).fill(session.preset == p && !session.customised ? AnyShapeStyle(.tint.opacity(0.18)) : AnyShapeStyle(.quaternary.opacity(0.5))))
                        .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(session.preset == p && !session.customised ? AnyShapeStyle(.tint) : AnyShapeStyle(.clear)))
                    }
                    .buttonStyle(.plain)
                }
            }
            if session.customised { Text("Custom settings (edited in the Train panel)").font(.caption).foregroundStyle(.orange) }
        }
    }
}

struct ComputePicker: View {
    @Bindable var session: TrainingSession
    @Binding var prepPlacement: ComputePlacement
    var dataset: PreparedDataset? = nil
    private var browser = WorkerBrowser.shared
    init(session: TrainingSession, prepPlacement: Binding<ComputePlacement>, dataset: PreparedDataset? = nil) {
        self.session = session; self._prepPlacement = prepPlacement; self.dataset = dataset
    }

    /// Ticked devices: "local" and worker ids (from the session's placement when nothing was ticked yet).
    private var selected: Set<String> {
        if let s = session.deviceIds { return s }
        switch session.placement {
        case .thisMac: return ["local"]
        case .remote: return [session.workerId ?? browser.workers.first?.id ?? "local"]
        case .both: return Set(["local"] + browser.workers.map(\.id))
        }
    }
    private func online(_ id: String) -> Bool { if case .online = browser.health[id]?.status { return true }; return false }
    private func tick(_ id: String) -> Binding<Bool> {
        Binding(get: { selected.contains(id) }, set: { on in
            var s = selected
            if on { s.insert(id) } else if s.count > 1 { s.remove(id) }       // keep at least one device
            apply(s)
        })
    }
    /// Maps the ticked devices onto the existing modes: this Mac only / one worker / several (block-parallel).
    private func apply(_ s: Set<String>) {
        session.deviceIds = s
        let workers = s.subtracting(["local"])
        if workers.isEmpty { session.placement = .thisMac }
        else if s.count == 1, let w = workers.first { session.placement = .remote; session.workerId = w }
        else { session.placement = .both; session.workerId = workers.sorted().first }
        prepPlacement = session.placement
    }

    var body: some View {
        SettingRow(id: "train.device") {
            VStack(alignment: .leading, spacing: 4) {
                Toggle(isOn: tick("local")) {
                    Text("This Mac").font(.callout) + Text("  \(MetalContext.shared.device.name)").font(.caption).foregroundColor(.secondary)
                }
                ForEach(browser.workers) { w in
                    Toggle(isOn: tick(w.id)) {
                        Text(w.name).font(.callout)
                            + Text("  \(w.gpuName ?? w.host)\(online(w.id) ? "" : " · offline")").font(.caption).foregroundColor(online(w.id) ? .secondary : .orange)
                    }
                }
                if browser.workers.isEmpty {
                    Text("No workers found — start Splat Worker on a PC or Compute on an iPhone (or add one under Compute).")
                        .font(.caption).foregroundStyle(.secondary)
                } else if selected.count > 1 {
                    Text("Several devices: big scenes are split into blocks shared between them; small ones train on the fastest.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            .toggleStyle(.checkbox)
        }
        if let ds = dataset, session.placement != .remote, let problem = MemoryForecast.macProblem(dataset: ds, config: session.config) {
            Label(problem + " If you start anyway, it is trained in blocks that each fit, then merged.", systemImage: "memorychip")
                .font(.caption).foregroundStyle(.orange).fixedSize(horizontal: false, vertical: true)
        }
        if let ds = dataset, session.placement != .thisMac,
           let wid = session.workerId ?? browser.workers.first?.id, let info = browser.health[wid]?.info,
           let problem = MemoryForecast.workerProblem(info: info, dataset: ds, config: session.config) {
            Label(problem + " If you start anyway, it is trained in blocks that each fit, then merged.", systemImage: "memorychip")
                .font(.caption).foregroundStyle(.orange).fixedSize(horizontal: false, vertical: true)
        }
        if let ds = dataset {
            let est = TrainingTimeEstimate.entries(dataset: ds, config: session.config, workers: browser.workers)
            if est.isEmpty || (est.count == 1 && !browser.workers.isEmpty) {
                Text("Run Compute → Benchmark to see how long this takes on each machine.").font(.caption).foregroundStyle(.secondary)
            } else {
                let fastest = est.min { $0.seconds < $1.seconds }?.deviceId
                Text("Estimated: " + est.map { e in
                    "\(e.name) ≈ \(formatDuration(e.seconds))" + (e.uploadSeconds >= 30 ? " (incl. \(formatDuration(e.uploadSeconds)) upload)" : "")
                        + (e.deviceId == fastest && est.count > 1 ? " — fastest" : "")
                }.joined(separator: " · ")).font(.caption).foregroundStyle(.secondary)
            }
        }
    }
}
