import Foundation
import SwiftUI
import Observation
import simd
import UniformTypeIdentifiers

/// Central app state. One per window.
@MainActor
@Observable
final class AppModel {
    // MARK: mode & panels
    var mode: ExperienceMode = ExperienceMode(rawValue: UserDefaults.standard.string(forKey: "experienceMode") ?? "easy") ?? .easy {
        didSet { UserDefaults.standard.set(mode.rawValue, forKey: "experienceMode") }
    }
    enum InspectorTab: String, CaseIterable, Identifiable { case view = "View", edit = "Edit", train = "Train", export = "Export"; var id: String { rawValue } }
    var inspectorTab: InspectorTab = .view
    var sidebar: ContentView.SidebarItem? = .viewer
    /// Live view of a job running on a Linux worker (Compute → Watch).
    let remoteWatch = RemoteJobWatcher()
    var showInspector = true
    var showNewProject = false
    var newProjectSource: SourceDescription? = nil
    var showShortcuts = false
    var alert: AlertInfo? = nil
    struct AlertInfo: Identifiable { let id = UUID(); var title: String; var message: String }

    // MARK: rendering
    let renderer: SplatRenderer
    var viewer = ViewerSettings() { didSet { renderer.settings = viewer } }
    var cameraMode: ViewerCamera.Mode = .orbit {
        didSet {
            var c = renderer.camera
            c.mode = cameraMode
            // Fly → orbit: pivot on the surface in the middle of the view (as double-click does), not on a point
            // floating a fixed distance ahead. Orbit → fly keeps the distance, so fly speed carries over.
            if oldValue == .fly, cameraMode == .orbit, let p = surfaceAtViewCentre(c) {
                let e = c.eye
                c.target = p
                c.distance = max(simd_distance(e, p), 0.01)
                let d = simd_normalize(e - p), up = simd_normalize(c.up)
                let ref: SIMD3<Float> = abs(up.z) < 0.9 ? SIMD3(0, 0, 1) : SIMD3(1, 0, 0)
                let right = simd_normalize(simd_cross(ref, up)), fwd = simd_cross(up, right)
                c.pitch = asin(min(max(simd_dot(d, up), -1), 1))
                c.yaw = atan2(simd_dot(d, right), simd_dot(d, fwd))
            }
            renderer.camera = c
        }
    }

    /// Nearest reasonably opaque splat within ~0.6° of the view's centre line.
    func surfaceAtViewCentre(_ cam: ViewerCamera) -> SIMD3<Float>? {
        guard let sc = scene else { return nil }
        let m = cam.worldToCamera
        let pos = sc.positionsPointer, fl = sc.flagsPointer
        var bestZ: Float = .infinity
        var best: SIMD3<Float>?
        let tol: Float = 0.01 * 0.01
        for i in stride(from: 0, to: sc.count, by: max(1, sc.count / 600_000)) where fl[i] & 1 == 0 && pos[i].w > 0.3 {
            let t = m * SIMD4(pos[i].x, pos[i].y, pos[i].z, 1)
            guard t.z > cam.near else { continue }
            let u = t.x / t.z, v = t.y / t.z
            if u * u + v * v < tol && t.z < bestZ { bestZ = t.z; best = SIMD3(pos[i].x, pos[i].y, pos[i].z) }
        }
        return best
    }
    var navigation = NavigationSettings.load() {
        didSet { navigation.save(); renderer.smoothing = navigation.smoothing ? 0.08 : 0 }
    }
    /// World "up" used by the viewer. 3DGS / COLMAP convention is −Y; Splat Studio datasets are
    /// normalised to it. Changed with the Orientation controls and remembered per file.
    var worldUp = SIMD3<Float>(0, -1, 0)
    var recents: [URL] = NSDocumentController.shared.recentDocumentURLs
    var confirmHome = false
    /// Open mask editor (sheet), if any.
    var maskEditor: MaskEditorModel? = nil
    /// Step the New Project sheet opens on (snapshot harness / deep links).
    var newProjectStartStep = 0
    var stats = RenderStats()

    // MARK: document
    var cloud: GaussianCloud? = nil
    var scene: SplatScene? = nil
    var documentURL: URL? = nil
    /// Project name when a dataset is open (output files carry long timestamped names), else the file name.
    var documentName: String {
        if let ds = dataset { return ds.root.deletingLastPathComponent().lastPathComponent }
        return documentURL?.deletingPathExtension().lastPathComponent ?? (training.isActive ? "Training…" : "Splat Studio")
    }
    var isLoading = false
    var dataset: PreparedDataset? = nil
    var selectedCount = 0
    var deletedCount = 0
    private var undoStack: [Data] = []
    private var redoStack: [Data] = []
    var canUndo: Bool { !undoStack.isEmpty }
    var canRedo: Bool { !redoStack.isEmpty }

    // MARK: sessions
    let training = TrainingSession()
    let prep = PrepSession()

    init() {
        do { renderer = try SplatRenderer() } catch { fatalError("Renderer failed: \(error)") }
        renderer.onStats = { [weak self] s in self?.stats = s }
        renderer.settings = viewer
        renderer.smoothing = navigation.smoothing ? 0.08 : 0
        training.app = self
        remoteWatch.app = self
        prep.app = self
    }

    // MARK: open / save

    static let openableTypes: [UTType] = [UTType(filenameExtension: "ply")!, UTType(filenameExtension: "splat") ?? .data, UTType(filenameExtension: "spz") ?? .data]

    func open(_ url: URL) {
        let ext = url.pathExtension.lowercased()
        if ["ply", "splat", "spz"].contains(ext), ext != "ply" || PLYCodec.isGaussianPLY(url: url) {
            loadSplat(url)
        } else if url.lastPathComponent == PreparedDataset.fileName || FileManager.default.fileExists(atPath: url.appendingPathComponent(PreparedDataset.fileName).path) {
            let root = url.lastPathComponent == PreparedDataset.fileName ? url.deletingLastPathComponent() : url
            openDataset(root)
        } else if let src = SourceDescription.guess(url) {
            newProjectSource = src
            showNewProject = true
        } else {
            alert = AlertInfo(title: "Can't open \(url.lastPathComponent)", message: "Drop a splat (.ply, .spz, .splat), a folder of photos, a video, an iPhone capture, a COLMAP project or a Nerfstudio transforms.json.")
        }
        NSDocumentController.shared.noteNewRecentDocumentURL(url)
        recents = NSDocumentController.shared.recentDocumentURLs
    }

    func loadSplat(_ url: URL) {
        isLoading = true
        Task.detached(priority: .userInitiated) {
            do {
                let g = try SplatFileFormat.load(url: url)
                let sc = SplatScene(cloud: g)
                await MainActor.run {
                    self.documentURL = url
                    self.worldUp = Orientation.savedUp(for: url) ?? Orientation.upFromPLYComment(url)
                        ?? Orientation.upFromProject(of: url) ?? Orientation.standardUp
                    self.setScene(cloud: g, scene: sc, frame: true)
                    self.homeCamera = self.renderer.camera
                    self.isLoading = false
                }
            } catch {
                await MainActor.run {
                    self.isLoading = false
                    self.alert = AlertInfo(title: "Couldn't open \(url.lastPathComponent)", message: error.localizedDescription)
                }
            }
        }
    }

    func setScene(cloud g: GaussianCloud?, scene sc: SplatScene, frame: Bool) {
        cloud = g
        scene = sc
        renderer.scene = sc
        undoStack.removeAll(); redoStack.removeAll()
        refreshCounts()
        if frame { frameScene() }
        if viewer.cropMin == ViewerSettings().cropMin {
            viewer.cropMin = sc.boundsMin; viewer.cropMax = sc.boundsMax
        }
    }

    func openDataset(_ root: URL) {
        do {
            let ds = try PreparedDataset.load(root)
            dataset = ds
            // A worker job that was still running when this project was last open: pick it back up.
            training.resumePendingRemote(dataset: ds, outputsDir: ds.root.deletingLastPathComponent().appendingPathComponent("outputs", isDirectory: true))
            updateCameraLines()
            inspectorTab = .train
            if scene == nil {
                if let p = ds.pointCloudURL, let pc = try? PLYCodec.readPointCloud(url: p) {
                    var cfg = TrainingConfig(); cfg.maxGaussians = 2_000_000
                    let g = GaussianInitializer.fromPoints(pc, config: cfg)
                    setScene(cloud: g, scene: SplatScene(cloud: g), frame: false)
                }
                frameDataset()
            }
        } catch {
            alert = AlertInfo(title: "Couldn't open dataset", message: error.localizedDescription)
        }
    }

    /// Exports the current (edited) scene.
    /// - bakeOrientation: rotates the scene so the viewer's current up becomes the standard −Y
    ///   (exact, including view-dependent colour), so other viewers show it upright too.
    func export(to url: URL, format: SplatFileFormat, shDegree: Int?, onlyKept: Bool, bakeOrientation: Bool) async throws {
        guard var g = editedCloud(onlyKept: onlyKept) else { throw FormatError.invalid("nothing to export") }
        let up = worldUp
        let bake = bakeOrientation && simd_distance(up, Orientation.standardUp) > 1e-4
        try await Task.detached {
            if bake { g.rotate(by: Orientation.rotation(from: up, to: Orientation.standardUp)) }
            if format == .ply { try PLYCodec.writeGaussians(g, to: url, shDegree: shDegree, up: bake ? Orientation.standardUp : up) }
            else { try format.save(g, to: url, shDegree: shDegree) }
        }.value
    }

    /// Current cloud with deletions and crop applied.
    func editedCloud(onlyKept: Bool = true) -> GaussianCloud? {
        guard var g = cloud ?? training.trainerSnapshot() else { return nil }
        guard onlyKept, let sc = scene, sc.count == g.count else { return g }
        let f = sc.flagsPointer
        var keep = [Bool](repeating: true, count: g.count)
        for i in 0..<g.count {
            if f[i] & 1 != 0 { keep[i] = false; continue }
            if viewer.cropEnabled {
                let p = g.position(i)
                if any(p .< viewer.cropMin) || any(p .> viewer.cropMax) { keep[i] = false }
            }
        }
        if keep.contains(false) { g = g.filtered(keep) }
        return g
    }

    // MARK: camera

    /// Camera to return to with Reset (R). Set whenever a scene or dataset is first shown.
    var homeCamera: ViewerCamera? = nil

    func frameScene() {
        guard let sc = scene else { return }
        var c = renderer.camera
        c.up = worldUp
        c.yaw = 0; c.pitch = 0.25
        c.frame(center: sc.center, radius: sc.radius)
        c.mode = cameraMode
        renderer.camera = c
    }

    /// Starts at a real training viewpoint: splats are only well-defined from where the scene was
    /// captured, so this is a far better first impression than framing the bounding box from outside.
    func frameDataset() {
        guard let ds = dataset else { return }
        let u = ds.manifest.upVector
        worldUp = SIMD3(u[0], u[1], u[2])
        if let cam = ds.manifest.cameras.first(where: { $0.holdout != true }) ?? ds.manifest.cameras.first {
            lookThrough(cam)
        } else {
            var c = renderer.camera
            let ctr = ds.manifest.sceneCenter
            c.up = worldUp
            c.frame(center: SIMD3(ctr[0], ctr[1], ctr[2]), radius: ds.manifest.sceneExtent)
            renderer.camera = c
        }
        homeCamera = renderer.camera
    }

    func lookThrough(_ cam: DatasetCamera) {
        var c = renderer.camera
        let ctr = dataset.map { SIMD3($0.manifest.sceneCenter[0], $0.manifest.sceneCenter[1], $0.manifest.sceneCenter[2]) } ?? .zero
        let focus = max(simd_distance(cam.center, ctr), 0.1)
        c.adopt(cameraToWorld: cam.c2w, focusDistance: focus, fovY: 2 * atan(Float(cam.height) / (2 * cam.fy)), worldUp: worldUp)
        c.mode = cameraMode
        renderer.camera = c
    }

    /// Back to the starting view (R / toolbar). Keeps the current orientation.
    func resetCamera() {
        if var h = homeCamera {
            if h.up != worldUp { h.up = worldUp }
            h.mode = cameraMode
            renderer.camera = h
        } else if dataset != nil { frameDataset() } else { frameScene() }
    }

    // MARK: orientation

    func setWorldUp(_ up: SIMD3<Float>) {
        worldUp = simd_normalize(up)
        if let u = documentURL { Orientation.saveUp(worldUp, for: u) }
        var c = renderer.goalCamera
        c.up = worldUp
        renderer.camera = c
        homeCamera?.up = worldUp
    }
    func flipUpsideDown() { setWorldUp(-worldUp) }
    /// Makes whatever is currently "up" on screen the scene's up direction.
    func levelToCurrentView() { setWorldUp(renderer.camera.screenUp) }

    // MARK: home

    /// Returns to the start page and frees the scene's GPU and CPU memory.
    func goHome(force: Bool = false) {
        if training.isActive && !force { confirmHome = true; return }
        if training.isActive { training.cancel() }
        renderer.releaseBuffers()
        scene = nil
        cloud = nil
        dataset = nil
        documentURL = nil
        homeCamera = nil
        undoStack.removeAll(); redoStack.removeAll()
        selectedCount = 0; deletedCount = 0
        viewer.cropEnabled = false
        viewer.showCropBox = false
        viewer.cropMin = ViewerSettings().cropMin; viewer.cropMax = ViewerSettings().cropMax
        worldUp = SIMD3(0, -1, 0)
        inspectorTab = .view
        sidebar = .viewer
        showShortcuts = false
        stats = RenderStats()
        if !training.isActive { training.history.removeAll(); training.evalHistory.removeAll(); training.state = .idle }
        renderer.camera = ViewerCamera()
    }

    func setView(_ preset: String) {
        var c = renderer.camera
        switch preset {
        case "front": c.yaw = 0; c.pitch = 0
        case "back": c.yaw = .pi; c.pitch = 0
        case "left": c.yaw = -.pi / 2; c.pitch = 0
        case "right": c.yaw = .pi / 2; c.pitch = 0
        case "top": c.pitch = 1.5
        default: break
        }
        renderer.camera = c
    }

    static func extent(of cams: [DatasetCamera]) -> Float {
        guard !cams.isEmpty else { return 1 }
        let c = cams.map(\.center), m = c.reduce(SIMD3<Float>(repeating: 0), +) / Float(c.count)
        return max(1e-3, 1.1 * (c.map { simd_length($0 - m) }.max() ?? 1))
    }

    func updateCameraLines() {
        var lines: [LineVertex] = []
        // Watching a worker job: its own cameras (or none), never the open project's — they're another scene.
        let watching = remoteWatch.watched != nil
        let cams: [DatasetCamera] = watching ? (remoteWatch.cameras ?? []) : (dataset?.manifest.cameras ?? [])
        let extent: Float = watching ? (remoteWatch.cameraExtent ?? Self.extent(of: cams)) : (dataset?.manifest.sceneExtent ?? 1)
        if viewer.showCameras, !cams.isEmpty {
            let size = extent * 0.04
            for cam in cams {
                let m = cam.c2w
                let o = cam.center
                let col: SIMD4<Float> = (cam.holdout ?? false) ? SIMD4(1, 0.6, 0.1, 0.9) : SIMD4(0.3, 0.7, 1, 0.8)
                let hw = Float(cam.width) / (2 * cam.fx), hh = Float(cam.height) / (2 * cam.fy)
                let corners = [SIMD3<Float>(-hw, -hh, 1), SIMD3(hw, -hh, 1), SIMD3(hw, hh, 1), SIMD3(-hw, hh, 1)].map { c -> SIMD3<Float> in
                    let w = m * SIMD4(c * size, 1); return SIMD3(w.x, w.y, w.z)
                }
                for c in corners { lines.append(.init(pos: o, color: col)); lines.append(.init(pos: c, color: col)) }
                for i in 0..<4 { lines.append(.init(pos: corners[i], color: col)); lines.append(.init(pos: corners[(i + 1) % 4], color: col)) }
            }
        }
        if viewer.showCropBox || viewer.cropEnabled {
            let a = viewer.cropMin, b = viewer.cropMax
            let col = SIMD4<Float>(1, 0.8, 0.2, 1)
            let v = (0..<8).map { i in SIMD3<Float>(i & 1 == 0 ? a.x : b.x, i & 2 == 0 ? a.y : b.y, i & 4 == 0 ? a.z : b.z) }
            for (i, j) in [(0,1),(2,3),(4,5),(6,7),(0,2),(1,3),(4,6),(5,7),(0,4),(1,5),(2,6),(3,7)] {
                lines.append(.init(pos: v[i], color: col)); lines.append(.init(pos: v[j], color: col))
            }
        }
        renderer.lines = lines
    }

    // MARK: editing (flags live in GPU-shared memory)

    func refreshCounts() {
        guard let sc = scene else { selectedCount = 0; deletedCount = 0; return }
        selectedCount = sc.selectedCount
        deletedCount = sc.deletedCount
    }

    private func pushUndo() {
        guard let sc = scene else { return }
        undoStack.append(Data(bytes: sc.flagsPointer, count: sc.count))
        if undoStack.count > 30 { undoStack.removeFirst() }
        redoStack.removeAll()
    }

    func undo() {
        guard let sc = scene, let d = undoStack.popLast() else { return }
        redoStack.append(Data(bytes: sc.flagsPointer, count: sc.count))
        d.withUnsafeBytes { memcpy(sc.flagsPointer, $0.baseAddress!, min(d.count, sc.count)) }
        renderer.invalidate(); refreshCounts()
    }

    func redo() {
        guard let sc = scene, let d = redoStack.popLast() else { return }
        undoStack.append(Data(bytes: sc.flagsPointer, count: sc.count))
        d.withUnsafeBytes { memcpy(sc.flagsPointer, $0.baseAddress!, min(d.count, sc.count)) }
        renderer.invalidate(); refreshCounts()
    }

    private func modifyFlags(_ body: (UnsafeMutablePointer<UInt8>, UnsafeMutablePointer<SIMD4<Float>>, Int) -> Void) {
        guard let sc = scene else { return }
        pushUndo()
        body(sc.flagsPointer, sc.positionsPointer, sc.count)
        renderer.invalidate()
        refreshCounts()
    }

    func deleteBelowOpacity(_ t: Float) {
        modifyFlags { f, p, n in for i in 0..<n where p[i].w < t { f[i] |= 1 } }
    }

    func deleteLargerThan(fraction: Float) {
        guard let g = cloud, let sc = scene, g.count == sc.count else { return }
        let limit = log(fraction * sc.radius * 2)
        modifyFlags { f, _, n in
            for i in 0..<n where max(g.scales[i*3], max(g.scales[i*3+1], g.scales[i*3+2])) > limit { f[i] |= 1 }
        }
    }

    func removeOutliers(minNeighbours: Int, radiusFraction: Float) {
        guard let sc = scene else { return }
        let r = radiusFraction * sc.radius * 2
        let n = sc.count
        let pts = (0..<n).map { i -> SIMD3<Float> in let v = sc.positionsPointer[i]; return SIMD3(v.x, v.y, v.z) }
        let flags = sc.flagsPointer
        // Uniform grid neighbour count.
        struct K: Hashable { var x, y, z: Int32 }
        var grid: [K: Int32] = [:]
        func key(_ p: SIMD3<Float>) -> K { K(x: Int32(floor(p.x / r)), y: Int32(floor(p.y / r)), z: Int32(floor(p.z / r))) }
        for i in 0..<n where flags[i] & 1 == 0 { grid[key(pts[i]), default: 0] += 1 }
        modifyFlags { f, _, n in
            for i in 0..<n where f[i] & 1 == 0 {
                let k = key(pts[i]); var c: Int32 = -1
                for dx: Int32 in -1...1 { for dy: Int32 in -1...1 { for dz: Int32 in -1...1 { c += grid[K(x: k.x+dx, y: k.y+dy, z: k.z+dz)] ?? 0 } } }
                if c < minNeighbours { f[i] |= 1 }
            }
        }
    }

    func applyCrop() {
        let a = viewer.cropMin, b = viewer.cropMax
        modifyFlags { f, p, n in
            for i in 0..<n {
                let q = SIMD3(p[i].x, p[i].y, p[i].z)
                if any(q .< a) || any(q .> b) { f[i] |= 1 }
            }
        }
        viewer.cropEnabled = false
        updateCameraLines()
    }

    func deleteSelected() { modifyFlags { f, _, n in for i in 0..<n where f[i] & 2 != 0 { f[i] = (f[i] | 1) & ~2 } } }
    func clearSelection() { modifyFlags { f, _, n in for i in 0..<n { f[i] &= ~2 } } }
    func invertSelection() { modifyFlags { f, _, n in for i in 0..<n where f[i] & 1 == 0 { f[i] ^= 2 } } }
    func restoreAll() { modifyFlags { f, _, n in for i in 0..<n { f[i] = 0 } } }

    /// Brush selection: everything projecting within `radius` pixels of `point` (view coords, top-left origin).
    func brushSelect(at point: CGPoint, radius: CGFloat, viewSize: CGSize, deselect: Bool, beginStroke: Bool) {
        guard let sc = scene else { return }
        if beginStroke { pushUndo() }
        let cam = renderer.camera
        let w = Float(viewSize.width), h = Float(viewSize.height)
        let k = cam.intrinsics(width: w, height: h)
        let m = cam.worldToCamera
        let px = Float(point.x), py = Float(point.y), r2 = Float(radius * radius)
        let p = sc.positionsPointer, f = sc.flagsPointer
        let n = sc.count
        let chunks = 32
        DispatchQueue.concurrentPerform(iterations: chunks) { c in
            for i in (n * c / chunks)..<(n * (c + 1) / chunks) {
                if f[i] & 1 != 0 { continue }
                let t = m * SIMD4(p[i].x, p[i].y, p[i].z, 1)
                if t.z <= cam.near { continue }
                let u = k.x * t.x / t.z + k.z - px, v = k.y * t.y / t.z + k.w - py
                if u * u + v * v < r2 { if deselect { f[i] &= ~2 } else { f[i] |= 2 } }
            }
        }
        renderer.invalidate()
        refreshCounts()
    }

    // MARK: screenshots

    func screenshot(to url: URL, width: Int, height: Int) throws {
        guard let tex = renderer.renderOffscreen(width: width, height: height, camera: renderer.camera, settings: viewer) else { throw FormatError.invalid("nothing to render") }
        try ImageWriter.writePNG(texture: tex, to: url)
    }
}
