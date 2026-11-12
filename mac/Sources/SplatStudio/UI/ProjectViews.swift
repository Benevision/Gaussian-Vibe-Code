import SwiftUI
import AppKit

// MARK: - Status

/// Where a project stands, in one line, with what to do about it.
struct ProjectStatus {
    enum Kind { case processing, processingStopped, processingFailed, training, trainingStopped, trained, ready, empty, missing }
    var kind: Kind
    var title: String
    var detail: String? = nil

    var icon: String {
        switch kind {
        case .processing, .training: return "gearshape.2.fill"
        case .processingStopped, .trainingStopped: return "pause.circle.fill"
        case .processingFailed: return "exclamationmark.triangle.fill"
        case .trained: return "checkmark.circle.fill"
        case .ready: return "tray.full.fill"
        case .empty: return "circle.dashed"
        case .missing: return "questionmark.folder"
        }
    }
    var color: Color {
        switch kind {
        case .processing, .training: return .accentColor
        case .processingStopped, .trainingStopped: return .orange
        case .processingFailed: return .red
        case .trained: return .green
        case .ready: return .teal
        case .empty, .missing: return .secondary
        }
    }
    var needsAttention: Bool { [.processingStopped, .processingFailed, .trainingStopped].contains(kind) }

    @MainActor
    static func of(_ p: ProjectSummary, app: AppModel) -> ProjectStatus {
        guard p.exists else { return ProjectStatus(kind: .missing, title: "Folder not found", detail: p.url.path) }
        let prepLive = app.prep.running && app.prep.projectDir?.standardizedFileURL == p.url.standardizedFileURL
        if prepLive {
            let stage = app.prep.progress.stage.isEmpty ? "Starting" : app.prep.progress.stage
            return ProjectStatus(kind: .processing, title: "Processing · \(Int(app.prep.overall * 100)) %", detail: stage)
        }
        if let pr = p.record?.prep, pr.state != .finished {
            let at = pr.stage.map { " at \($0.title.lowercased())" } ?? ""
            switch pr.state {
            case .failed: return ProjectStatus(kind: .processingFailed, title: "Processing failed\(at)", detail: pr.message)
            case .running: return ProjectStatus(kind: .processingStopped, title: "Processing interrupted\(at)",
                                                detail: "The app quit or the Mac restarted while it ran. Resume continues from the last finished stage.")
            default: return ProjectStatus(kind: .processingStopped, title: "Processing stopped\(at)",
                                          detail: "Resume continues from the last finished stage.")
            }
        }
        let trainLive = app.training.isActive && app.training.projectDir?.standardizedFileURL == p.url.standardizedFileURL
        if trainLive {
            let pct = app.training.status.totalSteps > 0 ? " · \(Int(app.training.fraction * 100)) %" : ""
            return ProjectStatus(kind: .training, title: (app.training.state == .paused ? "Training paused" : "Training") + pct,
                                 detail: app.training.status.phase.isEmpty ? nil : app.training.status.phase)
        }
        if let t = p.record?.training, t.state != .finished {
            var d = t.totalSteps > 0 ? "Reached step \(t.step.formatted()) of \(t.totalSteps.formatted())" : ""
            if let b = t.blockCount { d += (d.isEmpty ? "" : "; ") + "\(t.blocksDone ?? 0) of \(b) blocks finished and kept" }
            else if let c = t.checkpointStep { d += (d.isEmpty ? "" : "; ") + "checkpoint at step \(c.formatted()) on this Mac" }
            switch t.state {
            case .failed: return ProjectStatus(kind: .trainingStopped, title: "Training failed", detail: [t.message, d].compactMap { $0 }.joined(separator: ". "))
            case .running: return ProjectStatus(kind: .trainingStopped, title: "Training interrupted", detail: d)
            default: return ProjectStatus(kind: .trainingStopped, title: "Training stopped", detail: d)
            }
        }
        if let r = p.results.first {
            let date = (try? r.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate)
            return ProjectStatus(kind: .trained, title: "Trained", detail: date.map { "Latest result \($0.formatted(.relative(presentation: .named)))" })
        }
        if p.datasetReady { return ProjectStatus(kind: .ready, title: "Ready to train", detail: p.cameras > 0 ? "\(p.cameras) cameras" : nil) }
        return ProjectStatus(kind: .empty, title: p.checkpoints.isEmpty ? "Not processed yet" : "Processing not finished")
    }
}

// MARK: - Sidebar rows

struct ProjectSidebarRow: View {
    let model: AppModel
    let url: URL
    var body: some View {
        let _ = ProjectLibrary.shared.revision
        let p = ProjectSummary(url, full: false)
        let st = ProjectStatus.of(p, app: model)
        Label {
            VStack(alignment: .leading, spacing: 1) {
                Text(p.name).lineLimit(1)
                Text(st.title).font(.caption).foregroundStyle(st.needsAttention ? AnyShapeStyle(st.color) : AnyShapeStyle(.secondary)).lineLimit(1)
            }
        } icon: {
            Image(systemName: st.icon).foregroundStyle(st.color)
        }
        .help(url.path)
    }
}

// MARK: - Project page

/// One project: what it was made from, how far each stage got, and buttons to resume, redo from any stage,
/// train, or open a result.
struct ProjectPage: View {
    let model: AppModel
    let url: URL
    @State private var options = PrepOptions()
    @State private var confirmRemove = false
    @State private var redoStage: PrepStage? = nil

    var body: some View {
        let _ = ProjectLibrary.shared.revision
        let p = ProjectSummary(url)
        let st = ProjectStatus.of(p, app: model)
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                header(p)
                statusCard(p, st)
                if p.canResumeProcessing { stagesSection(p) }
                else if p.datasetReady {
                    PanelSection(title: "Dataset", systemImage: "tray.full") {
                        Text("\(p.cameras) cameras. This project was made before SplatWork kept its settings, so its processing can't be redone from here — make a new project from the same source for that.")
                            .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                    }
                }
                resultsSection(p)
                footer(p)
            }
            .padding(24)
            .frame(maxWidth: 680, alignment: .leading)
            .frame(maxWidth: .infinity)
        }
        .background(.background)
        .onAppear { loadOptions() }
        .onChange(of: url) { loadOptions() }
        .confirmationDialog("Remove “\(p.name)” from the list?", isPresented: $confirmRemove) {
            Button("Remove from List") { ProjectLibrary.shared.remove(url.path); model.sidebar = .viewer }
        } message: { Text("The project folder and its files stay where they are. Open it again to bring it back.") }
        .confirmationDialog(redoStage.map { "Redo from \($0.title.lowercased())?" } ?? "", isPresented: Binding(get: { redoStage != nil }, set: { if !$0 { redoStage = nil } })) {
            Button("Redo") { if let s = redoStage { resumeProcessing(redo: s) }; redoStage = nil }
        } message: {
            Text("That stage and every stage after it run again with the project's settings. Earlier stages are kept.")
        }
    }

    private func loadOptions() {
        options = ProjectRecord.load(url)?.options ?? PrepOptions()
        ProcessingDevicePicker.applyRemembered(to: &options)
    }

    // MARK: sections

    private func header(_ p: ProjectSummary) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(p.name).font(.largeTitle.bold())
            HStack(spacing: 8) {
                Text(p.url.path).lineLimit(1).truncationMode(.middle).foregroundStyle(.secondary)
                Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([p.url]) }.controlSize(.small)
            }
            .font(.callout)
            if let src = p.record?.source {
                Label("Made from \(src.displayName)", systemImage: "doc").font(.callout).foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder private func statusCard(_ p: ProjectSummary, _ st: ProjectStatus) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: st.icon).font(.title).foregroundStyle(st.color)
                VStack(alignment: .leading, spacing: 3) {
                    Text(st.title).font(.title2.bold())
                    if let d = st.detail { Text(d).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true).textSelection(.enabled) }
                }
            }
            if [.processingStopped, .processingFailed, .empty].contains(st.kind), p.canResumeProcessing {
                ProcessingDevicePicker(options: $options).disabled(model.prep.running)
            }
            HStack(spacing: 10) { actions(p, st) }
            if model.prep.running, st.kind != .processing, [.processingStopped, .processingFailed, .empty].contains(st.kind) {
                Text("Another project is being processed; this one can resume when it finishes.").font(.caption).foregroundStyle(.secondary)
            }
            if model.training.isActive, st.kind == .trainingStopped || st.kind == .ready || st.kind == .trained {
                Text("Another project is training; stop it to train this one.").font(.caption).foregroundStyle(.secondary)
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(st.color.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(st.color.opacity(0.25)))
    }

    @ViewBuilder private func actions(_ p: ProjectSummary, _ st: ProjectStatus) -> some View {
        switch st.kind {
        case .processing:
            Button("Show Progress") { model.showNewProject = true }.buttonStyle(.borderedProminent)
            Button("Stop", role: .destructive) { model.prep.cancel() }
        case .processingStopped, .processingFailed, .empty:
            if p.canResumeProcessing {
                Button(st.kind == .empty && p.checkpoints.isEmpty ? "Start Processing" : "Resume Processing") { resumeProcessing(redo: nil) }
                    .buttonStyle(.borderedProminent).disabled(model.prep.running)
                if st.kind == .processingFailed {
                    Button("Change Settings…") { changeSettings() }.disabled(model.prep.running)
                }
            }
            if p.datasetReady || FileManager.default.fileExists(atPath: p.datasetDir.appendingPathComponent(PreparedDataset.fileName).path),
               st.kind != .empty, !p.canResumeProcessing {
                Button("Open Dataset") { openDataset(p) }
            }
        case .training:
            Button("Show Live View") { model.showLiveTraining() }.buttonStyle(.borderedProminent)
        case .trainingStopped:
            if let t = p.record?.training {
                Button(resumeLabel(t, p)) { resumeTraining(p, t) }.buttonStyle(.borderedProminent).disabled(model.training.isActive)
                Button("Start Over") { openDataset(p, train: true) }.disabled(model.training.isActive)
                    .help("Open the dataset in the Train panel to train from scratch, with any settings")
            }
            if let r = p.results.first { Button("Open Latest Result") { open(r) } }
        case .trained:
            Button("Open Latest Result") { if let r = p.results.first { open(r) } }.buttonStyle(.borderedProminent)
            Button("Train Again…") { openDataset(p, train: true) }.disabled(!p.datasetReady || model.training.isActive)
        case .ready:
            Button("Train…") { openDataset(p, train: true) }.buttonStyle(.borderedProminent).disabled(model.training.isActive)
            Button("Open Dataset") { openDataset(p) }
        case .missing:
            Button("Remove from List") { confirmRemove = true }
        }
    }

    private func resumeLabel(_ t: ProjectRecord.Training, _ p: ProjectSummary) -> String {
        let out = p.url.appendingPathComponent(t.output)
        if FileManager.default.fileExists(atPath: TrainingSession.PendingRemoteJob.file(for: out).path) { return "Reconnect to the Worker Job" }
        if let b = t.blockCount, let d = t.blocksDone, d > 0 { return "Resume (\(d) of \(b) blocks done)" }
        if let c = t.checkpointStep, c > 0, t.placement == .thisMac { return "Resume from Step \(c.formatted())" }
        if t.placement == .remote { return "Run Again on the Worker" }
        return "Resume Training"
    }

    private func stagesSection(_ p: ProjectSummary) -> some View {
        let fps = p.record.flatMap { r in r.source.flatMap { s in r.options.map { PrepCheckpoints.fingerprints(s, $0) } } } ?? [:]
        let live = model.prep.running && model.prep.projectDir?.standardizedFileURL == p.url.standardizedFileURL
        let busy = model.prep.running || (model.training.isActive && model.training.projectDir?.standardizedFileURL == p.url.standardizedFileURL)
        return PanelSection(title: "Processing", systemImage: "list.bullet.rectangle") {
            VStack(alignment: .leading, spacing: 0) {
                ForEach(PrepStage.allCases, id: \.self) { s in
                    let e = p.checkpoints.first { $0.stage == s }?.entry
                    let current = e != nil && e?.fingerprint == fps[s]
                    let running = live && PrepSessionStage.current(model.prep) == s
                    HStack(spacing: 10) {
                        Group {
                            if running { ProgressView().controlSize(.small) }
                            else if current { Image(systemName: "checkmark.circle.fill").foregroundStyle(.green) }
                            else if e != nil { Image(systemName: "exclamationmark.circle").foregroundStyle(.orange) }
                            else { Image(systemName: "circle").foregroundStyle(.tertiary) }
                        }
                        .frame(width: 18)
                        VStack(alignment: .leading, spacing: 1) {
                            Text(s.title).font(.callout.weight(.medium))
                            Group {
                                if running { Text(model.prep.progress.message.isEmpty ? model.prep.progress.stage : model.prep.progress.message) }
                                else if let e { Text(e.summary + " · " + e.savedAt.formatted(date: .abbreviated, time: .shortened) + (current ? "" : " · settings or source changed since")) }
                                else { Text("Not done yet") }
                            }
                            .font(.caption).foregroundStyle(.secondary).lineLimit(1)
                        }
                        Spacer()
                        if e != nil && !busy {
                            Button("Redo from Here") { redoStage = s }.controlSize(.small)
                        }
                    }
                    .padding(.vertical, 6)
                    if s != PrepStage.allCases.last { Divider().padding(.leading, 28) }
                }
            }
            HStack {
                Button("Change Settings…") { changeSettings() }.disabled(model.prep.running)
                Text("Opens the project's settings; stages whose settings you don't change are reused.").font(.caption).foregroundStyle(.secondary)
            }
            .padding(.top, 4)
        }
    }

    @ViewBuilder private func resultsSection(_ p: ProjectSummary) -> some View {
        if !p.results.isEmpty || p.record?.training != nil {
            PanelSection(title: "Training", systemImage: "cube.transparent") {
                if let t = p.record?.training {
                    Text("Last run: \(t.config.iterations.formatted()) steps on \(placementName(t)), started \(t.started.formatted(date: .abbreviated, time: .shortened))")
                        .font(.caption).foregroundStyle(.secondary)
                }
                ForEach(p.results.prefix(8), id: \.self) { r in
                    HStack {
                        Image(systemName: "circle.hexagongrid").foregroundStyle(.tint)
                        VStack(alignment: .leading, spacing: 1) {
                            Text(r.lastPathComponent).lineLimit(1).truncationMode(.middle)
                            if let d = try? r.resourceValues(forKeys: [.contentModificationDateKey, .fileSizeKey]) {
                                Text([d.contentModificationDate?.formatted(date: .abbreviated, time: .shortened),
                                      d.fileSize.map { ByteCountFormatter.string(fromByteCount: Int64($0), countStyle: .file) }].compactMap { $0 }.joined(separator: " · "))
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                        }
                        Spacer()
                        Button("Open") { open(r) }.controlSize(.small)
                    }
                }
                if p.results.count > 8 { Text("…and \(p.results.count - 8) older results in outputs/").font(.caption).foregroundStyle(.secondary) }
            }
        }
    }

    private func footer(_ p: ProjectSummary) -> some View {
        HStack {
            Spacer()
            Button("Remove from List…") { confirmRemove = true }.controlSize(.small)
        }
    }

    private func placementName(_ t: ProjectRecord.Training) -> String {
        switch t.placement {
        case .thisMac: return "this Mac"
        case .remote: return t.workerId.flatMap { id in WorkerBrowser.shared.workers.first { $0.id == id }?.name } ?? "a worker"
        case .both: return "\(t.deviceIds?.count ?? 2) devices"
        }
    }

    // MARK: actions

    private func resumeProcessing(redo: PrepStage?) {
        model.prep.resume(project: url, redo: redo, workerId: options.compute == .thisMac ? nil : options.remoteWorkerId)
        model.showNewProject = true
    }

    private func changeSettings() {
        model.newProjectSource = ProjectRecord.load(url)?.source
        model.newProjectPrefill = url
        model.showNewProject = true
    }

    private func openDataset(_ p: ProjectSummary, train: Bool = false) {
        model.openDataset(p.datasetDir)
        if train { model.inspectorTab = .train; model.showInspector = true }
        model.sidebar = .viewer
    }

    private func resumeTraining(_ p: ProjectSummary, _ t: ProjectRecord.Training) {
        model.openDataset(p.datasetDir)          // re-attaches a worker job that is still running, if any
        if !model.training.isActive, let ds = model.dataset {
            model.training.start(dataset: ds, resume: t)
        }
        model.inspectorTab = .train; model.showInspector = true
        model.sidebar = .viewer
    }

    private func open(_ r: URL) { model.open(r); model.sidebar = .viewer }
}

/// Which `PrepStage` the live preparation is in.
enum PrepSessionStage {
    @MainActor static func current(_ prep: PrepSession) -> PrepStage {
        switch prep.phase { case .images: return .frames; case .masks: return .masks; case .poses: return .poses; case .finalising: return .done }
    }
}

// MARK: - Start page: pick up where you left off

/// Projects that stopped partway (or are running), on the start page.
struct ContinueProjectsView: View {
    let model: AppModel
    var body: some View {
        let _ = ProjectLibrary.shared.revision
        let items = ProjectLibrary.shared.projects.prefix(12).map { ProjectSummary(URL(fileURLWithPath: $0.path), full: false) }
            .filter { $0.exists }.map { ($0, ProjectStatus.of($0, app: model)) }
        let open = items.filter { $0.1.needsAttention || $0.1.kind == .processing || $0.1.kind == .training }.prefix(3)
        let recent = items.prefix(4)
        let shown = open.isEmpty ? Array(recent) : Array(open)
        if !shown.isEmpty {
            VStack(alignment: .leading, spacing: 10) {
                Text(open.isEmpty ? "Recent projects" : "Pick up where you left off").font(.headline).foregroundStyle(.secondary)
                ForEach(shown, id: \.0.id) { p, st in
                    Button { model.sidebar = .project(p.url) } label: {
                        HStack(spacing: 12) {
                            Image(systemName: st.icon).font(.title2).foregroundStyle(st.color).frame(width: 28)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(p.name).font(.headline)
                                Text(st.title).font(.callout).foregroundStyle(.secondary)
                            }
                            Spacer()
                            Image(systemName: "chevron.right").foregroundStyle(.tertiary)
                        }
                        .padding(12)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 10))
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }
            }
            .frame(width: 624, alignment: .leading)
        }
    }
}
