import SwiftUI
import UniformTypeIdentifiers

// MARK: - iPhone inbox

struct InboxView: View {
    let model: AppModel
    var onOpen: () -> Void
    @State private var receiver = CaptureReceiverHolder.shared.receiver

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("iPhone Inbox").font(.largeTitle.bold())
                    Text("Captures sent from Splat Capture on your iPhone over Wi-Fi land here.").foregroundStyle(.secondary)
                }
                Spacer()
                HelpButton(id: "net.receive")
            }
            GroupBox {
                HStack(alignment: .top, spacing: 24) {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Circle().fill(isListening ? .green : .secondary).frame(width: 9, height: 9)
                            Text(statusText).font(.headline)
                        }
                        if let port = receiver.listeningPort {
                            Text("Manual address: " + (receiver.lanAddresses.isEmpty ? "–" : receiver.lanAddresses.map { "\($0):\(port)" }.joined(separator: ", ")))
                                .font(.callout.monospaced()).textSelection(.enabled)
                        }
                        Toggle("Receive from iPhone", isOn: Binding(get: { isListening }, set: { $0 ? receiver.start() : receiver.stop() }))
                        HStack { Toggle("Require pairing code", isOn: Binding(get: { receiver.requirePairingCode }, set: { receiver.requirePairingCode = $0 })); HelpButton(id: "net.pairing") }
                    }
                    Spacer()
                    if receiver.requirePairingCode {
                        VStack(spacing: 4) {
                            Text("Pairing code").font(.caption).foregroundStyle(.secondary)
                            Text(receiver.pairingCode.chunked).font(.system(size: 34, weight: .semibold, design: .monospaced))
                            Button("New code") { receiver.regeneratePairingCode() }.controlSize(.small)
                        }
                    }
                }
                .padding(8)
            }
            ForEach(Array(receiver.transfers.values)) { t in
                HStack { Text("Receiving \(t.name)…"); if let f = t.fraction { ProgressView(value: f) } else { ProgressView().controlSize(.small) } }
            }
            if receiver.received.isEmpty {
                ContentUnavailableView("No captures yet", systemImage: "iphone.radiowaves.left.and.right",
                                       description: Text("On your iPhone open Splat Capture → Library → Send to Mac."))
            } else {
                List(receiver.received) { c in
                    InboxRow(capture: c, model: model, receiver: receiver, onOpen: onOpen)
                }
            }
            Spacer()
        }
        .padding(24)
        .onAppear { receiver.loadExisting(); if receiver.state == .stopped { receiver.start() } }
    }

    private var isListening: Bool { if case .listening = receiver.state { return true }; return false }
    private var statusText: String {
        switch receiver.state {
        case .stopped: return "Not receiving"
        case .starting: return "Starting…"
        case .listening: return "Ready — visible as “\(receiver.advertisedName)”"
        case .failed(let e): return "Failed: \(e)"
        }
    }
}

private extension String {
    var chunked: String { count == 6 ? "\(prefix(3)) \(suffix(3))" : self }
}

// MARK: - Workers

/// "Act as a worker" switch + the jobs other devices sent to this Mac.
struct MacWorkerSection: View {
    @State private var worker = MacWorker.shared

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Divider()
            Toggle(isOn: Binding(get: { worker.isEnabled }, set: { worker.setEnabled($0) })) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Act as a worker")
                    Text(status).font(.caption).foregroundStyle(.secondary)
                }
            }
            ForEach(worker.jobs.reversed().prefix(5)) { j in
                HStack {
                    Text(j.name).lineLimit(1)
                    Spacer()
                    if j.isRunning { ProgressView(value: j.fraction).frame(width: 120) }
                    Text(j.isTerminal ? j.state : "\(Int(j.fraction * 100)) %").font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    if !j.isTerminal { Button("Stop") { worker.cancel(j.jobId) }.controlSize(.small) }
                }.font(.callout)
            }
        }.padding(.horizontal, 6).padding(.bottom, 6)
    }

    private var status: String {
        switch worker.serverState {
        case .stopped: return "Lets Splat Capture on your iPhone (and other Macs) send training here. Training only — learned masks and camera solving need a Linux worker."
        case .starting: return "Starting…"
        case .ready(let port): return "Visible as “\(worker.advertisedName)” on port \(port)" + (worker.isTraining ? " · training a job" : "")
        case .failed(let e): return "Couldn't start: \(e)"
        }
    }
}

struct WorkersView: View {
    let model: AppModel
    @State private var browser = WorkerBrowser.shared
    @State private var host = ""
    @State private var port = 7878
    @State private var token = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Compute").font(.largeTitle.bold())
                    Text("Offload camera solving and training to a Linux PC with an NVIDIA GPU running Splat Worker — or let this Mac train jobs for your iPhone and other Macs.").foregroundStyle(.secondary)
                }
                Spacer()
                HelpButton(id: "net.workers")
            }
            GroupBox("This Mac") {
                let d = MetalContext.shared.device
                HStack {
                    Image(systemName: "desktopcomputer").font(.title2)
                    VStack(alignment: .leading) {
                        Text(d.name).font(.headline)
                        Text("Metal 3 · \(formatBytes(Int(d.recommendedMaxWorkingSetSize))) GPU working set").font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Text("Ready").foregroundStyle(.green)
                }.padding(6)
                MacWorkerSection()
            }
            GroupBox("Linux workers") {
                if browser.workers.isEmpty {
                    Text(browser.isBrowsing ? "Searching the local network… Start Splat Worker on your PC and it will appear here." : "Discovery is off.")
                        .foregroundStyle(.secondary).padding(6).frame(maxWidth: .infinity, alignment: .leading)
                }
                ForEach(browser.workers) { w in
                    WorkerRow(endpoint: w, browser: browser)
                    WorkerJobsList(endpoint: w, model: model)
                }
                if !browser.hidden.isEmpty {
                    HStack {
                        Text("\(browser.hidden.count) discovered worker\(browser.hidden.count == 1 ? "" : "s") hidden").font(.caption).foregroundStyle(.secondary)
                        Button("Show again") { browser.unhideAll() }.buttonStyle(.link).font(.caption)
                    }.padding(.horizontal, 6)
                }
            }
            BenchmarkBox()
            GroupBox("Add manually") {
                HStack {
                    TextField("Host or IP", text: $host).textFieldStyle(.roundedBorder)
                    TextField("Port", value: $port, format: .number.grouping(.never)).textFieldStyle(.roundedBorder).frame(width: 70)
                    SecureField("Token (optional)", text: $token).textFieldStyle(.roundedBorder).frame(width: 180)
                    Button("Add") {
                        let ep = browser.addManual(host: host, port: port, token: token.isEmpty ? nil : token)
                        host = ""; token = ""
                        Task { await browser.checkHealth(ep.id) }
                    }.disabled(host.isEmpty)
                }.padding(6)
            }
            Text("Set up a worker: see linux/README.md — `./install.sh`, then `splatworker serve --token <token>`. The worker advertises itself with Bonjour; tokens are stored in your Keychain.")
                .font(.callout).foregroundStyle(.secondary)
            Spacer()
        }
        .padding(24)
        .onAppear { browser.start() }
    }
}

/// Jobs currently on a worker, with "Watch" to stream a training job's preview into the viewer.
private struct WorkerJobsList: View {
    let endpoint: WorkerEndpoint
    let model: AppModel
    @State private var jobs: [JobStatus] = []
    @State private var downloading: Set<String> = []
    @State private var downloadError: String? = nil
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(jobs) { j in
                HStack(spacing: 8) {
                    Image(systemName: j.kind == "process" ? "gearshape.2" : "cube.transparent").foregroundStyle(.secondary)
                    VStack(alignment: .leading, spacing: 1) {
                        Text(j.name ?? j.jobId).font(.callout).lineLimit(1)
                        Text(subtitle(j)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    }
                    Spacer()
                    if (j.kind ?? "train") == "train" && j.state == .completed {
                        Button(downloading.contains(j.jobId) ? "Downloading…" : "Open result") { openResult(j) }
                            .disabled(downloading.contains(j.jobId))
                            .help("Save this job's trained splat on this Mac and open it")
                    } else if (j.kind ?? "train") == "train" && !j.state.isTerminal {
                        let watching = model.remoteWatch.watched?.jobId == j.jobId
                        Button(watching ? "Watching" : "Watch") {
                            model.remoteWatch.watch(workerId: endpoint.id, jobId: j.jobId, name: j.name ?? j.jobId)
                            model.sidebar = .viewer
                        }.disabled(watching)
                    }
                }
            }
        }
        .overlay(alignment: .bottomLeading) { if let e = downloadError { Text(e).font(.caption).foregroundStyle(.red) } }
        .padding(.leading, 44).padding(.bottom, jobs.isEmpty ? 0 : 6)
        .task(id: endpoint.id) {
            while !Task.isCancelled {
                if let c = WorkerBrowser.shared.client(for: endpoint.id), let all = try? await c.jobs() {
                    // running / queued, then the most recent finished training jobs (results stay on the worker)
                    let done = all.filter { $0.state == .completed && ($0.kind ?? "train") == "train" }.suffix(5)
                    jobs = all.filter { !$0.state.isTerminal } + done.reversed()
                }
                try? await Task.sleep(nanoseconds: 5_000_000_000)
            }
        }
    }
    private func openResult(_ j: JobStatus) {
        guard let c = WorkerBrowser.shared.client(for: endpoint.id) else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [UTType(filenameExtension: "ply")!]
        panel.nameFieldStringValue = (j.name ?? j.jobId).replacingOccurrences(of: "/", with: "-") + ".ply"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        downloading.insert(j.jobId); downloadError = nil
        Task {
            do {
                try await c.downloadResultRetrying(j.jobId, to: url)
                model.open(url)
            } catch { downloadError = "Couldn't download \(j.name ?? j.jobId): \(error.localizedDescription)" }
            downloading.remove(j.jobId)
        }
    }

    private func subtitle(_ j: JobStatus) -> String {
        var p = [j.state.rawValue]
        if let s = j.step, let t = j.totalSteps, t > 0 { p.append("\(s.formatted())/\(t.formatted())") }
        if let ps = j.psnr { p.append(String(format: "%.2f dB", ps)) }
        return p.joined(separator: " · ")
    }
}

private struct WorkerRow: View {
    let endpoint: WorkerEndpoint
    let browser: WorkerBrowser
    @State private var token = ""
    var body: some View {
        let h = browser.health[endpoint.id]
        HStack(alignment: .top) {
            Image(systemName: "server.rack").font(.title2).foregroundStyle(color(h?.status))
            VStack(alignment: .leading, spacing: 2) {
                Text(endpoint.name).font(.headline)
                Text("\(endpoint.host):\(endpoint.port)\(endpoint.linkKind.map { " · \($0 == "wifi" ? "Wi-Fi" : $0)" } ?? "")\(endpoint.isManual ? " · manual" : " · Bonjour")").font(.caption.monospaced()).foregroundStyle(.secondary)
                if let info = h?.info {
                    ForEach(info.gpus ?? [], id: \.self) { g in Text("\(g.name) · \(g.memoryMB.map { "\($0 / 1024) GB" } ?? "")").font(.caption) }
                    Text("\(info.backend ?? "") · \((info.capabilities ?? []).joined(separator: ", "))\(info.busy == true ? " · busy" : "")").font(.caption).foregroundStyle(.secondary)
                    if (info.capabilities ?? []).contains("process") {
                        let m = Set(info.models ?? [])
                        if m.contains("dynamic") && m.contains("sam2_track") && m.contains("superpoint_lightglue") {
                            Text("Processing models: installed (\(m.count) models)").font(.caption).foregroundStyle(.secondary)
                        } else {
                            WorkerModelInstallView(workerId: endpoint.id, reason: "Processing models (camera solving, moving-object masks) aren't installed on this worker yet.")
                                .padding(.top, 4)
                        }
                    }
                }
                if case .unauthorized = h?.status {
                    HStack {
                        SecureField("Token", text: $token).textFieldStyle(.roundedBorder).frame(width: 200)
                        Button("Save") { browser.setToken(token, for: endpoint.id); Task { await browser.checkHealth(endpoint.id) } }
                    }
                }
            }
            Spacer()
            Text(statusText(h?.status)).font(.callout).foregroundStyle(color(h?.status))
            Button { Task { await browser.checkHealth(endpoint.id) } } label: { Image(systemName: "arrow.clockwise") }.buttonStyle(.borderless)
            if endpoint.isManual {
                Button { browser.removeManual(endpoint.id) } label: { Image(systemName: "trash") }.buttonStyle(.borderless)
                    .help("Remove this worker")
            } else {
                Button { browser.hide(endpoint.id) } label: { Image(systemName: "eye.slash") }.buttonStyle(.borderless)
                    .help("Remove this discovered worker from the list (e.g. to add it manually by its ethernet IP). “Show again” brings it back.")
            }
        }
        .padding(6)
    }
    private func statusText(_ s: WorkerHealth.Status?) -> String {
        switch s {
        case .online: return "Online"
        case .checking: return "Checking…"
        case .unauthorized: return "Needs token"
        case .offline(let e): return "Offline (\(e))"
        default: return "Unknown"
        }
    }
    private func color(_ s: WorkerHealth.Status?) -> Color {
        switch s { case .online: return .green; case .unauthorized: return .orange; case .offline: return .red; default: return .secondary }
    }
}

struct InboxRow: View {
    let capture: ReceivedCapture
    let model: AppModel
    let receiver: CaptureReceiver
    var onOpen: () -> Void
    @State private var projects: [(project: URL, splats: [URL])] = []
    @State private var confirmTrash = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                thumbnail
                VStack(alignment: .leading) {
                    Text(capture.name).font(.headline)
                    Text("\(capture.receivedAt.formatted(date: .abbreviated, time: .shortened)) · \(formatBytes(Int(capture.bytes)))\(capture.sender.map { " · from \($0)" } ?? "")").font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                Button("Show") { NSWorkspace.shared.activateFileViewerSelecting([capture.url]) }
                Button(projects.isEmpty ? "Create splat…" : "New splat…") {
                    model.newProjectSource = .captureBundle(capture.url); model.showNewProject = true; onOpen()
                }.buttonStyle(.borderedProminent)
                Menu {
                    Button("Move to Trash…", role: .destructive) { confirmTrash = true }
                } label: { Image(systemName: "ellipsis.circle") }
                .menuStyle(.borderlessButton).fixedSize()
            }
            ForEach(projects, id: \.project) { p in
                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 6) {
                        Image(systemName: "folder").foregroundStyle(.secondary)
                        Text(p.project.lastPathComponent).font(.callout)
                        Spacer()
                        let ds = p.project.appendingPathComponent("dataset", isDirectory: true)
                        if FileManager.default.fileExists(atPath: ds.appendingPathComponent(PreparedDataset.fileName).path) {
                            Button("Open dataset") { model.openDataset(ds); onOpen() }.controlSize(.small)
                        } else {
                            Text("not finished").font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    ForEach(p.splats, id: \.self) { s in
                        HStack(spacing: 6) {
                            Image(systemName: "cube.transparent").foregroundStyle(.tint)
                            Text(s.lastPathComponent).font(.caption).lineLimit(1).truncationMode(.middle)
                            Spacer()
                            Button("Open") { model.open(s); onOpen() }.controlSize(.small)
                        }
                    }
                }
                .padding(.leading, 44)
            }
        }
        .padding(.vertical, 4)
        .task(id: capture.url) { reload() }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in reload() }
        .confirmationDialog("Move “\(capture.name)” to the Trash?", isPresented: $confirmTrash) {
            Button("Move to Trash", role: .destructive) { receiver.trash(capture) }
        } message: { Text("Projects and splats made from it are kept.") }
    }

    @ViewBuilder private var thumbnail: some View {
        if let img = NSImage(contentsOf: capture.url.appendingPathComponent("thumbnail.jpg")) {
            Image(nsImage: img).resizable().aspectRatio(contentMode: .fill).frame(width: 36, height: 36).clipShape(RoundedRectangle(cornerRadius: 6))
        } else {
            Image(systemName: "iphone.gen3").foregroundStyle(.tint).frame(width: 36, height: 36)
        }
    }

    private func reload() {
        projects = CaptureProjects.projects(for: capture).map { ($0, CaptureProjects.splats(in: $0)) }
    }
}

// MARK: - Models & benchmarks

struct ModelsView: View {
    @State private var cards: [ModelCard] = []
    @State private var report: BenchmarkReport?
    @State private var reportURL: URL?
    @State private var python = PythonEnvironment.shared
    @State private var selected: String?
    @State private var installing = false

    var body: some View {
        HStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 12) {
                Text("Models & Benchmarks").font(.largeTitle.bold())
                Text("Feature matchers, 3D foundation models and segmentation models available for camera solving and masking. Numbers come from benchmarks run on this machine (or bundled reference runs).")
                    .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                envBox
                Table(rows, selection: $selected) {
                    TableColumn("Model") { r in
                        HStack { Text(r.name).lineLimit(1).help(r.id); if r.available { Image(systemName: "checkmark.circle.fill").foregroundStyle(.green).help("Installed") } }
                    }
                    .width(min: 190, ideal: 230)
                    TableColumn("Registered") { r in Text(r.summary?.meanRegisteredPct.map { String(format: "%.0f %%", $0) } ?? "–").monospacedDigit() }
                    TableColumn("Pose AUC@10") { r in Text(r.summary?.meanAucAt10.map { String(format: "%.2f", $0) } ?? r.summary?.meanPairPoseAucAt10.map { String(format: "%.2f*", $0) } ?? "–").monospacedDigit() }
                    TableColumn("Reproj. err") { r in Text(r.summary?.meanReprojError.map { String(format: "%.2f px", $0) } ?? "–").monospacedDigit() }
                    TableColumn("Time") { r in Text(r.summary?.meanWallTimeSec.map { formatDuration($0) } ?? "–").monospacedDigit() }
                    TableColumn("Memory") { r in Text(r.summary?.maxPeakMemoryMB.map { formatBytes(Int($0 * 1e6)) } ?? "–").monospacedDigit() }
                    TableColumn("Licence") { r in Text(r.card?.license ?? "–").lineLimit(1) }
                }
                .frame(minHeight: 240, maxHeight: .infinity)
                Text("* matching-only metric (no mapper installed). Higher is better for Registered and AUC; lower for error, time and memory.").font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                if let u = reportURL { Text("Benchmark: \(u.path)").font(.caption2).foregroundStyle(.secondary).textSelection(.enabled) }
            }
            .padding(20)
            .frame(minWidth: 480, maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            Divider()
            detailPane.frame(width: 320).frame(maxHeight: .infinity, alignment: .top)
        }
        .onAppear(perform: reload)
    }

    private var envBox: some View {
        GroupBox {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    HStack { Text("Models pack").font(.headline); HelpButton(id: "net.python") }
                    Text(envStatus).font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                if case .ready = python.state {
                    Button("Refresh") { Task { await python.refresh(); reload() } }
                } else {
                    Menu("Install…") {
                        Button("Core (SIFT + LightGlue family, ~0.6 GB)") { install(["sfm", "lightglue"]) }
                        Button("Core + masks + LoFTR (~0.9 GB)") { install(["sfm", "lightglue", "loftr", "masks", "sam2"]) }
                        Button("Everything incl. VGGT (~6 GB of weights on first use)") { install(["sfm", "lightglue", "loftr", "masks", "sam2", "vggt"]) }
                    }
                    .fixedSize().disabled(installing)
                }
            }
            if installing || !python.log.isEmpty {
                ScrollView { Text(python.log.suffix(200).joined(separator: "\n")).font(.caption2.monospaced()).frame(maxWidth: .infinity, alignment: .leading) }.frame(height: 90)
            }
        }
    }

    private var envStatus: String {
        switch python.state {
        case .unknown: return "Checking…"
        case .missing: return "Not installed. Built-in options (Object Capture, Apple Vision) work without it."
        case .installing: return "Installing…"
        case .ready(let v): return "Python \(v) · \(python.availableModels.count) models available"
        case .failed(let e): return "Install failed: \(e)"
        }
    }

    private func install(_ extras: [String]) {
        installing = true
        Task { do { try await python.install(extras: extras) } catch {}; installing = false; reload() }
    }

    struct Row: Identifiable { let id: String; let name: String; let card: ModelCard?; let summary: BenchmarkSummary?; let available: Bool }
    private var rows: [Row] {
        let ids = Set(cards.map(\.id)).union(report?.summary?.keys.map { $0 } ?? [])
        return ids.sorted().map { id in
            let c = cards.first { $0.id == id }
            return Row(id: id, name: c?.displayName ?? id, card: c, summary: report?.summary?[id], available: python.availableModels.contains(id))
        }
    }

    @ViewBuilder private var detailPane: some View {
        if let id = selected, let r = rows.first(where: { $0.id == id }) {
            ScrollView {
                VStack(alignment: .leading, spacing: 10) {
                    Text(r.name).font(.title2.bold())
                    if let s = r.card?.summary ?? r.card?.description { Text(s) }
                    if let b = r.card?.bestFor, !b.isEmpty { section("Best for", b, "star") }
                    if let s = r.card?.strengths, !s.isEmpty { section("Strengths", s, "plus.circle") }
                    if let w = r.card?.weaknesses, !w.isEmpty { section("Weaknesses", w, "minus.circle") }
                    Grid(alignment: .leading, horizontalSpacing: 10, verticalSpacing: 4) {
                        if let v = r.card?.speed { GridRow { Text("Speed").foregroundStyle(.secondary); Text(v) } }
                        if let v = r.card?.quality { GridRow { Text("Quality").foregroundStyle(.secondary); Text(v) } }
                        if let v = r.card?.license { GridRow { Text("Licence").foregroundStyle(.secondary); Text(v) } }
                        if let v = r.card?.weightsMB { GridRow { Text("Weights").foregroundStyle(.secondary); Text(formatBytes(Int(v * 1e6))) } }
                    }.font(.callout)
                    if let u = r.card?.url, let url = URL(string: u) { Link("Project page", destination: url) }
                }
                .padding(20)
            }
        } else {
            ContentUnavailableView("Select a model", systemImage: "cpu", description: Text("See strengths, weaknesses and licence."))
        }
    }

    private func section(_ title: String, _ items: [String], _ icon: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Label(title, systemImage: icon).font(.headline)
            ForEach(items, id: \.self) { Text("• \($0)").fixedSize(horizontal: false, vertical: true) }
        }
    }

    private func reload() {
        cards = ModelCard.loadDefault()
        if let l = BenchmarkReport.latest(in: BenchmarkReport.defaultDirectory) {
            report = l.report; reportURL = l.url
        } else if let t = PythonEnvironment.locateToolsDirectory(), let l = BenchmarkReport.latest(in: t) {
            report = l.report; reportURL = l.url
        }
        Task { await python.refresh() }
    }
}

// MARK: - Settings reference

struct SettingsReferenceView: View {
    @State private var query = ""
    @State private var selection: String?
    var body: some View {
        HStack(spacing: 0) {
            List(selection: $selection) {
                ForEach(SettingDoc.Group.allCases, id: \.self) { g in
                    let items = filtered.filter { $0.group == g }
                    if !items.isEmpty {
                        Section(g.rawValue) {
                            ForEach(items) { d in
                                VStack(alignment: .leading, spacing: 2) {
                                    HStack { Text(d.title); if d.advanced { Text("Advanced").font(.caption2).foregroundStyle(.secondary) } }
                                    Text(d.summary).font(.caption).foregroundStyle(.secondary).lineLimit(2)
                                }.tag(d.id)
                            }
                        }
                    }
                }
            }
            .searchable(text: $query, placement: .sidebar, prompt: "Search settings")
            .frame(width: 320)
            Divider()
            Group {
                if let id = selection { ScrollView { SettingDocView(id: id).padding(24).frame(maxWidth: 640, alignment: .leading) } }
                else { ScrollView { overview.padding(24).frame(maxWidth: 760, alignment: .leading) } }
            }
            .frame(minWidth: 380, maxWidth: .infinity, maxHeight: .infinity)
        }
    }
    /// All settings at a glance: one table per group; click a row to open it.
    private var overview: some View {
        VStack(alignment: .leading, spacing: 18) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Settings Reference").font(.largeTitle.bold())
                Text("Every option in Splat Studio, explained — \(SettingsCatalog.all.count) settings. Click one for details, or search on the left. Easy settings are always visible; Advanced ones appear in Advanced mode.")
                    .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            ForEach(SettingDoc.Group.allCases, id: \.self) { g in
                let items = filtered.filter { $0.group == g }
                if !items.isEmpty {
                    VStack(alignment: .leading, spacing: 6) {
                        Text(g.rawValue).font(.title3.bold())
                        Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 0) {
                            GridRow {
                                Text("Setting"); Text("Default"); Text("Range"); Text("Mode")
                            }.font(.caption.weight(.semibold)).foregroundStyle(.secondary).padding(.vertical, 4)
                            ForEach(items) { d in
                                Divider()
                                GridRow {
                                    Button(d.title) { selection = d.id }.buttonStyle(.link).multilineTextAlignment(.leading)
                                    Text(d.defaultValue).lineLimit(2)
                                    Text(d.range.isEmpty ? "–" : d.range).foregroundStyle(.secondary).lineLimit(2)
                                    Text(d.advanced ? "Advanced" : "Easy").font(.caption).foregroundStyle(.secondary)
                                }.font(.callout).padding(.vertical, 5)
                            }
                        }
                        .padding(.horizontal, 10)
                        .background(.quinary, in: RoundedRectangle(cornerRadius: 8))
                    }
                }
            }
        }
    }

    private var filtered: [SettingDoc] {
        guard !query.isEmpty else { return SettingsCatalog.all }
        let q = query.lowercased()
        return SettingsCatalog.all.filter { "\($0.title) \($0.summary) \($0.detail)".lowercased().contains(q) }
    }
}


/// Compute → Benchmark: the same training test on this Mac and every online worker, plus transfer speed.
/// Results drive the time estimates next to the training device picker and which machine Both mode picks.
private struct BenchmarkBox: View {
    @State private var results = BenchmarkStore.shared.all
    @State private var running = false
    @State private var status = ""
    @State private var fraction = 0.0
    @State private var error: String?
    @State private var includeMac = true
    private let browser = WorkerBrowser.shared

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Trains the same synthetic scene (1 M splats, 1600 × 1200, 2,000 steps) on each machine and times the transfer to and from each worker. Takes a few minutes per machine; results set the time estimates and which machine gets the work.")
                        .font(.callout).foregroundStyle(.secondary)
                    Spacer()
                    HelpButton(id: "net.benchmark")
                }
                if !results.isEmpty {
                    Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 4) {
                        GridRow {
                            Text("Machine").bold(); Text("GPU").bold(); Text("Training").bold(); Text("Relative").bold()
                            Text("Upload").bold(); Text("Download").bold(); Text("Measured").bold()
                        }.font(.caption)
                        let slowest = results.map(\.trainStepsPerSecond).min() ?? 1
                        ForEach(results) { r in
                            GridRow {
                                Text(r.deviceId == "local" ? "This Mac" : r.name)
                                Text(r.gpu ?? "–").lineLimit(1)
                                Text(String(format: "%.1f steps/s", r.trainStepsPerSecond))
                                Text(String(format: "%.1f×", r.trainStepsPerSecond / max(slowest, 0.01)))
                                Text(r.uploadMBps.map { String(format: "%.0f MB/s", $0) } ?? "–")
                                Text(r.downloadMBps.map { String(format: "%.0f MB/s", $0) } ?? "–")
                                Text(r.date.formatted(date: .abbreviated, time: .shortened))
                            }.font(.caption.monospacedDigit())
                        }
                    }
                }
                HStack {
                    Button(running ? "Running…" : "Run benchmark") { run() }.disabled(running)
                    Toggle("Include this Mac", isOn: $includeMac).toggleStyle(.checkbox)
                    if running { ProgressView(value: fraction).frame(width: 140) }
                    Text(status).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                }
                if let error { Text(error).font(.caption).foregroundStyle(.red) }
            }.padding(6)
        } label: { Text("Benchmark") }
    }

    private func run() {
        running = true; error = nil
        let targets = browser.workers.compactMap { w -> (WorkerEndpoint, WorkerClient)? in
            guard let h = browser.health[w.id], case .online = h.status, let c = browser.client(for: w.id) else { return nil }
            return (w, c)
        }
        let mac = includeMac
        Task {
            func report(_ m: String, _ f: Double) { Task { @MainActor in status = m; fraction = f } }
            do {
                for (w, c) in targets {
                    let r = try await SystemBenchmark.runRemote(endpoint: w, client: c) { m, f in report(m, f) }
                    BenchmarkStore.shared.save(r)
                    await MainActor.run { results = BenchmarkStore.shared.all }
                }
                if mac {
                    let r = try await SystemBenchmark.runLocal { m, f in report(m, f) }
                    BenchmarkStore.shared.save(r)
                }
                await MainActor.run { results = BenchmarkStore.shared.all; status = "Done"; running = false }
            } catch {
                await MainActor.run { self.error = error.localizedDescription; running = false }
            }
        }
    }
}
