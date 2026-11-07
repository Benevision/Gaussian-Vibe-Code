import SwiftUI
import AppKit
import ImageIO
import UniformTypeIdentifiers

/// Per-dataset mask editor: review masks frame by frame, paint / erase by hand, click an object to
/// segment it with SAM 2 (Models pack) and track it through every frame. Saving writes
/// `masks/<stem>.png` (255 = keep) and links them in dataset.json.
@MainActor
@Observable
final class MaskEditorModel: Identifiable {
    let id = UUID()
    let dataset: PreparedDataset
    var index = 0
    var tool: Tool = .click
    var brush: CGFloat = 40
    var showOverlay = true
    var status: String = ""
    var busy = false
    var dirty: Set<Int> = []
    /// SAM 2 click prompts on the current frame: (x, y, label) in image pixels.
    var points: [(CGPoint, Int)] = []
    /// Objects (prompts per frame) queued for tracking through the sequence.
    var trackedObjects: [[String: Any]] = []

    enum Tool: String, CaseIterable, Identifiable { case click, paint, erase; var id: String { rawValue } }

    private var masks: [Int: [UInt8]] = [:]
    private var images: [Int: CGImage] = [:]
    private var undo: [(Int, [UInt8])] = []
    private var server: MaskServerClient?

    init(dataset: PreparedDataset) { self.dataset = dataset }

    var cameras: [DatasetCamera] { dataset.manifest.cameras }
    var current: DatasetCamera { cameras[index] }

    func image(_ i: Int) -> CGImage? {
        if let im = images[i] { return im }
        let url = dataset.url(cameras[i].image)
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil), let im = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return nil }
        images[i] = im
        if images.count > 24 { images.removeValue(forKey: images.keys.first { $0 != i }!) }
        return im
    }

    func thumbnail(_ i: Int) -> CGImage? {
        let url = dataset.url(cameras[i].image)
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        return CGImageSourceCreateThumbnailAtIndex(src, 0, [kCGImageSourceCreateThumbnailFromImageAlways: true, kCGImageSourceThumbnailMaxPixelSize: 160] as CFDictionary)
    }

    func mask(_ i: Int) -> [UInt8] {
        if let m = masks[i] { return m }
        let c = cameras[i]
        var m = [UInt8](repeating: 255, count: c.width * c.height)
        if let rel = c.mask, let (g, _, _) = try? ImageUtil.readGray8(dataset.url(rel), width: c.width, height: c.height) { m = g }
        masks[i] = m
        return m
    }

    func hasMask(_ i: Int) -> Bool { cameras[i].mask != nil || dirty.contains(i) }

    /// Mask overlay image (red where ignored).
    func overlay(_ i: Int) -> CGImage? {
        let c = cameras[i], m = mask(i)
        var px = [UInt8](repeating: 0, count: c.width * c.height * 4)
        for j in 0..<m.count where m[j] < 128 { px[j * 4] = 255; px[j * 4 + 1] = 40; px[j * 4 + 2] = 40; px[j * 4 + 3] = 140 }
        guard let ctx = CGContext(data: &px, width: c.width, height: c.height, bitsPerComponent: 8, bytesPerRow: c.width * 4,
                                  space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        return ctx.makeImage()
    }

    func paint(at p: CGPoint, keep: Bool, begin: Bool) {
        let c = current
        var m = mask(index)
        if begin { undo.append((index, m)); if undo.count > 40 { undo.removeFirst() } }
        let r = Int(brush), r2 = r * r
        let cx = Int(p.x), cy = Int(p.y)
        for y in max(0, cy - r)..<min(c.height, cy + r + 1) {
            for x in max(0, cx - r)..<min(c.width, cx + r + 1) where (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r2 {
                m[y * c.width + x] = keep ? 255 : 0
            }
        }
        masks[index] = m
        dirty.insert(index)
    }

    func undoLast() {
        guard let (i, m) = undo.popLast() else { return }
        masks[i] = m; dirty.insert(i); index = i
    }

    func clearFrame() { undo.append((index, mask(index))); masks[index] = [UInt8](repeating: 255, count: current.width * current.height); dirty.insert(index) }

    // MARK: SAM 2 (Models pack)

    var samAvailable: Bool { if case .ready = PythonEnvironment.shared.state { return true }; return false }

    func click(at p: CGPoint, positive: Bool) {
        points.append((p, positive ? 1 : 0))
        Task { await previewSAM() }
    }

    func previewSAM() async {
        guard !points.isEmpty else { return }
        busy = true; status = "Segmenting…"
        defer { busy = false }
        do {
            if server == nil { server = try MaskServerClient() }
            let url = dataset.url(current.image)
            let pngURL = try await server!.segment(image: url, points: points.map { [Double($0.0.x), Double($0.0.y), Double($0.1)] })
            let c = current
            let (obj, _, _) = try ImageUtil.readGray8(pngURL, width: c.width, height: c.height)
            // SAM returns the object; remove it from the keep-mask (on top of the mask before this click session).
            var m = baseBeforeClicks ?? mask(index)
            if baseBeforeClicks == nil { baseBeforeClicks = m; undo.append((index, m)) }
            for j in m.indices where obj[j] >= 128 { m[j] = 0 }
            masks[index] = m
            dirty.insert(index)
            status = "Object selected. Click more (+ object, ⌥ background) or Track through all frames."
        } catch {
            status = "SAM 2 unavailable: \(error.localizedDescription)"
        }
    }
    private var baseBeforeClicks: [UInt8]?

    func commitObject() {
        guard !points.isEmpty else { return }
        trackedObjects.append(["image": (current.image as NSString).lastPathComponent,
                               "points": points.map { [Double($0.0.x), Double($0.0.y), Double($0.1)] },
                               "objectId": trackedObjects.count + 1])
        points.removeAll(); baseBeforeClicks = nil
    }

    func resetClicks() {
        if let b = baseBeforeClicks { masks[index] = b }
        points.removeAll(); baseBeforeClicks = nil
    }

    /// Tracks all queued objects through the sequence with the SAM 2 video predictor.
    func trackAll() async {
        commitObject()
        guard !trackedObjects.isEmpty else { return }
        busy = true; status = "Tracking \(trackedObjects.count) object(s) through \(cameras.count) frames…"
        defer { busy = false }
        do {
            try await save()   // so the tracker combines with existing masks
            var r = ProcessingRequest(task: .masks)
            r.imagesDir = "images"; r.outputDir = "."; r.maskModel = "sam2_auto"
            r.maskPrompts = trackedObjects.map { JSONValue(any: $0) }
            try await SplatPrepRunner().run(r, workDir: dataset.root, progress: { p in
                Task { @MainActor in self.status = "\(p.stage) \(Int(max(p.fraction, 0) * 100)) %" }
            })
            masks.removeAll()
            try relinkMasks()
            trackedObjects.removeAll()
            status = "Tracked. Review frames, then Save."
        } catch {
            status = "Tracking failed: \(error.localizedDescription)"
        }
    }

    // MARK: save

    func save() async throws {
        var ds = dataset
        for i in dirty {
            let c = cameras[i]
            let rel = "masks/" + ((c.image as NSString).lastPathComponent as NSString).deletingPathExtension + ".png"
            try FileManager.default.createDirectory(at: ds.url("masks"), withIntermediateDirectories: true)
            try ImageUtil.writeGray8(mask(i), width: c.width, height: c.height, to: ds.url(rel))
            ds.manifest.cameras[i].mask = rel
        }
        try ds.save()
        dirty.removeAll()
        status = "Saved masks. Retrain to apply them."
    }

    /// Links any masks/<stem>.png present on disk into dataset.json.
    private func relinkMasks() throws {
        var ds = dataset
        for i in ds.manifest.cameras.indices {
            let rel = "masks/" + ((ds.manifest.cameras[i].image as NSString).lastPathComponent as NSString).deletingPathExtension + ".png"
            if FileManager.default.fileExists(atPath: ds.url(rel).path) { ds.manifest.cameras[i].mask = rel }
        }
        try ds.save()
    }
}

/// Talks to `python -m splatprep maskserver` (keeps SAM 2 loaded for instant click feedback).
/// Protocol: one JSON request per line on stdin, one JSON reply per line on stdout.
final class MaskServerClient: @unchecked Sendable {
    private let process = Process()
    private let input = Pipe(), output = Pipe()
    private var buffer = Data()
    private let lock = NSLock()

    init() throws {
        let py = PythonEnvironment.defaultPython
        guard FileManager.default.isExecutableFile(atPath: py.path) else { throw PipelineError.unsupported("Models pack not installed") }
        process.executableURL = py
        process.arguments = ["-m", "splatprep", "maskserver"]
        var env = ProcessInfo.processInfo.environment
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        if let t = PythonEnvironment.locateToolsDirectory() { env["PYTHONPATH"] = t.path }
        process.environment = env
        process.standardInput = input
        process.standardOutput = output
        process.standardError = FileHandle.nullDevice
        try process.run()
    }

    deinit { process.terminate() }

    func segment(image: URL, points: [[Double]]) async throws -> URL {
        let req: [String: Any] = ["image": image.path, "points": points]
        let line = try JSONSerialization.data(withJSONObject: req) + Data("\n".utf8)
        input.fileHandleForWriting.write(line)
        let reply: [String: Any] = try await Task.detached { [self] in
            while true {
                if let r = self.nextLine() {
                    if let obj = try? JSONSerialization.jsonObject(with: r) as? [String: Any] { return obj }
                    continue
                }
                let d = self.output.fileHandleForReading.availableData
                if d.isEmpty { throw PipelineError.processFailed("mask server exited") }
                self.lock.lock(); self.buffer.append(d); self.lock.unlock()
            }
        }.value
        if let e = reply["error"] as? String { throw PipelineError.processFailed(e) }
        guard let m = reply["mask"] as? String else { throw PipelineError.processFailed("no mask in reply") }
        return URL(fileURLWithPath: m)
    }

    private func nextLine() -> Data? {
        lock.lock(); defer { lock.unlock() }
        guard let nl = buffer.firstIndex(of: 0x0a) else { return nil }
        let l = buffer.prefix(upTo: nl)
        buffer.removeSubrange(...nl)
        return l.isEmpty ? nil : Data(l)
    }
}

struct MaskEditorView: View {
    @State var model: MaskEditorModel
    @Environment(\.dismiss) private var dismiss
    @State private var dragging = false

    var body: some View {
        VStack(spacing: 0) {
            toolbar
            Divider()
            HStack(spacing: 0) {
                frameList.frame(width: 190)
                Divider()
                canvas.frame(maxWidth: .infinity, maxHeight: .infinity).background(Color.black.opacity(0.85))
            }
            Divider()
            HStack {
                Text(model.status).font(.callout).foregroundStyle(.secondary).lineLimit(2)
                Spacer()
                Text("\(model.dirty.count) unsaved").font(.callout).foregroundStyle(model.dirty.isEmpty ? Color.secondary : Color.orange)
                Button("Close") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Save masks") { Task { try? await model.save() } }.buttonStyle(.borderedProminent).disabled(model.dirty.isEmpty)
            }
            .padding(10)
        }
        .frame(minWidth: 1000, minHeight: 680)
    }

    private var toolbar: some View {
        HStack(spacing: 14) {
            Picker("Tool", selection: $model.tool) {
                Label("Select object", systemImage: "cursorarrow.click.2").tag(MaskEditorModel.Tool.click)
                Label("Paint out", systemImage: "paintbrush").tag(MaskEditorModel.Tool.paint)
                Label("Paint back", systemImage: "eraser").tag(MaskEditorModel.Tool.erase)
            }
            .pickerStyle(.segmented).frame(width: 360)
            if model.tool != .click {
                HStack { Text("Size"); Slider(value: $model.brush, in: 4...200).frame(width: 140); Text("\(Int(model.brush)) px").monospacedDigit() }
            } else {
                Button("Track through all frames") { Task { await model.trackAll() } }
                    .disabled(model.busy || (model.points.isEmpty && model.trackedObjects.isEmpty) || !model.samAvailable)
                Button("Next object") { model.commitObject() }.disabled(model.points.isEmpty)
                Button("Reset clicks") { model.resetClicks() }.disabled(model.points.isEmpty)
                if !model.samAvailable { Text("Install the Models pack for click-to-select").font(.caption).foregroundStyle(.orange) }
            }
            Spacer()
            Toggle("Overlay", isOn: $model.showOverlay)
            Button { model.undoLast() } label: { Image(systemName: "arrow.uturn.backward") }.keyboardShortcut("z").help("Undo")
            Button("Clear frame") { model.clearFrame() }
            if model.busy { ProgressView().controlSize(.small) }
        }
        .padding(10)
    }

    private var frameList: some View {
        ScrollViewReader { proxy in
            List(model.cameras.indices, id: \.self, selection: Binding(get: { model.index }, set: { if let v = $0 { model.index = v } })) { i in
                HStack {
                    if let t = model.thumbnail(i) { Image(decorative: t, scale: 1).resizable().aspectRatio(contentMode: .fit).frame(width: 64, height: 48) }
                    VStack(alignment: .leading) {
                        Text((model.cameras[i].image as NSString).lastPathComponent).font(.caption).lineLimit(1)
                        if model.dirty.contains(i) { Text("edited").font(.caption2).foregroundStyle(.orange) }
                        else if model.hasMask(i) { Text("masked").font(.caption2).foregroundStyle(.secondary) }
                    }
                }
                .tag(i).id(i)
            }
            .onChange(of: model.index) { _, i in proxy.scrollTo(i) }
        }
    }

    private var canvas: some View {
        GeometryReader { geo in
            let c = model.current
            let scale = min(geo.size.width / CGFloat(c.width), geo.size.height / CGFloat(c.height))
            let w = CGFloat(c.width) * scale, h = CGFloat(c.height) * scale
            let origin = CGPoint(x: (geo.size.width - w) / 2, y: (geo.size.height - h) / 2)
            ZStack(alignment: .topLeading) {
                if let im = model.image(model.index) { Image(decorative: im, scale: 1).resizable().frame(width: w, height: h) }
                if model.showOverlay, let ov = model.overlay(model.index) { Image(decorative: ov, scale: 1).resizable().frame(width: w, height: h) }
                ForEach(Array(model.points.enumerated()), id: \.offset) { _, p in
                    Circle().fill(p.1 == 1 ? Color.green : Color.red).frame(width: 12, height: 12)
                        .overlay(Circle().stroke(.white, lineWidth: 2))
                        .position(x: p.0.x * scale, y: p.0.y * scale)
                }
            }
            .frame(width: w, height: h)
            .offset(x: origin.x, y: origin.y)
            .contentShape(Rectangle())
            .gesture(DragGesture(minimumDistance: 0)
                .onChanged { v in
                    let p = CGPoint(x: v.location.x / scale, y: v.location.y / scale)
                    switch model.tool {
                    case .paint: model.paint(at: p, keep: false, begin: !dragging)
                    case .erase: model.paint(at: p, keep: true, begin: !dragging)
                    case .click: break
                    }
                    dragging = true
                }
                .onEnded { v in
                    if model.tool == .click {
                        let p = CGPoint(x: v.location.x / scale, y: v.location.y / scale)
                        model.click(at: p, positive: !NSEvent.modifierFlags.contains(.option))
                    }
                    dragging = false
                })
        }
        .overlay(alignment: .bottom) {
            HStack {
                Button { model.index = max(0, model.index - 1) } label: { Image(systemName: "chevron.left") }.keyboardShortcut(.leftArrow, modifiers: [])
                Text("Frame \(model.index + 1) of \(model.cameras.count)").monospacedDigit()
                Button { model.index = min(model.cameras.count - 1, model.index + 1) } label: { Image(systemName: "chevron.right") }.keyboardShortcut(.rightArrow, modifiers: [])
                Text("Click an object to select it · ⌥-click marks background · red = ignored in training").font(.caption).foregroundStyle(.secondary)
            }
            .padding(8).background(.regularMaterial, in: Capsule()).padding(10)
        }
    }
}
