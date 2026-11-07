import SwiftUI
import MetalKit
import simd

/// Interaction tool active in the viewer.
enum ViewerTool: String, CaseIterable, Identifiable {
    case navigate, select
    var id: String { rawValue }
}

/// User navigation preferences (persisted). Documented in SettingsCatalog (`nav.*`).
struct NavigationSettings: Codable, Equatable {
    enum TwoFinger: String, Codable, CaseIterable { case orbit, pan }
    var rotateSpeed: Float = 1
    var panSpeed: Float = 1
    var zoomSpeed: Float = 1
    var flySpeed: Float = 1
    var invertVertical = false
    var trackpadTwoFinger: TwoFinger = .orbit
    var smoothing = true

    static let key = "navigationSettings"
    static func load() -> NavigationSettings {
        guard let d = UserDefaults.standard.data(forKey: key), let n = try? JSONDecoder().decode(NavigationSettings.self, from: d) else { return .init() }
        return n
    }
    func save() { if let d = try? JSONEncoder().encode(self) { UserDefaults.standard.set(d, forKey: Self.key) } }
}

/// MTKView with orbit / fly navigation, trackpad gestures and the selection brush.
///
/// Mouse: drag = orbit (grab the scene), right-drag or ⇧-drag = pan, wheel = zoom.
/// Trackpad: two-finger drag = orbit (or pan, per preference; ⇧ swaps), pinch = zoom,
/// two-finger rotate = turn. Input moves a goal camera; the renderer eases towards it.
final class SplatMTKView: MTKView {
    var renderer: SplatRenderer?
    weak var model: AppModel?
    var tool: ViewerTool = .navigate
    var brushRadius: CGFloat = 20
    private var keysDown = Set<UInt16>()
    private var lastDrag: CGPoint = .zero
    private var modifiers: NSEvent.ModifierFlags = []

    override var acceptsFirstResponder: Bool { true }
    override func becomeFirstResponder() -> Bool { true }

    private var nav: NavigationSettings { model?.navigation ?? .init() }

    func configure() {
        colorPixelFormat = .bgra8Unorm
        framebufferOnly = false
        autoResizeDrawable = false
        preferredFramesPerSecond = NSScreen.main?.maximumFramesPerSecond ?? 120
        renderer?.onBeforeFrame = { [weak self] dt in self?.applyFlyKeys(dt) }
        addGestureRecognizer(NSMagnificationGestureRecognizer(target: self, action: #selector(onPinch(_:))))
        addGestureRecognizer(NSRotationGestureRecognizer(target: self, action: #selector(onRotate(_:))))
    }

    private func viewPoint(_ e: NSEvent) -> CGPoint {
        let p = convert(e.locationInWindow, from: nil)
        return CGPoint(x: p.x, y: bounds.height - p.y)
    }

    private func goal(_ f: (inout ViewerCamera) -> Void) { renderer?.mutateGoal(f) }
    private var vSign: Float { nav.invertVertical ? -1 : 1 }

    // MARK: mouse

    override func mouseDown(with e: NSEvent) {
        window?.makeFirstResponder(self)
        lastDrag = viewPoint(e)
        if e.clickCount == 2 { focus(at: lastDrag); return }
        if tool == .select { model?.brushSelect(at: lastDrag, radius: brushRadius, viewSize: bounds.size, deselect: e.modifierFlags.contains(.option), beginStroke: true) }
    }

    override func mouseDragged(with e: NSEvent) {
        let p = viewPoint(e)
        let dx = Float(p.x - lastDrag.x), dy = Float(p.y - lastDrag.y)
        lastDrag = p
        if tool == .select {
            model?.brushSelect(at: p, radius: brushRadius, viewSize: bounds.size, deselect: e.modifierFlags.contains(.option), beginStroke: false)
            return
        }
        let n = nav, r = sceneRadius
        goal { c in
            if e.modifierFlags.contains(.shift) { c.pan(dx: dx, dy: dy, viewHeight: Float(bounds.height), speed: n.panSpeed, sceneRadius: r) }
            else if c.mode == .fly { c.look(dx: dx, dy: dy * vSign, speed: n.rotateSpeed) }
            else { c.orbit(dx: dx, dy: dy * vSign, speed: n.rotateSpeed) }
        }
    }

    override func rightMouseDown(with e: NSEvent) { lastDrag = viewPoint(e) }
    override func rightMouseDragged(with e: NSEvent) {
        let p = viewPoint(e)
        let dx = Float(p.x - lastDrag.x), dy = Float(p.y - lastDrag.y); lastDrag = p
        let n = nav, r = sceneRadius
        goal { $0.pan(dx: dx, dy: dy, viewHeight: Float(bounds.height), speed: n.panSpeed, sceneRadius: r) }
    }
    override func otherMouseDown(with e: NSEvent) { lastDrag = viewPoint(e) }
    override func otherMouseDragged(with e: NSEvent) { rightMouseDragged(with: e) }

    override func scrollWheel(with e: NSEvent) {
        let n = nav
        if e.hasPreciseScrollingDeltas {
            // Trackpad / Magic Mouse: express deltas as finger motion regardless of "natural scrolling".
            let flip: CGFloat = e.isDirectionInvertedFromDevice ? 1 : -1
            let fx = Float(e.scrollingDeltaX * flip), fy = Float(e.scrollingDeltaY * flip)
            var mode = n.trackpadTwoFinger
            if e.modifierFlags.contains(.shift) { mode = mode == .orbit ? .pan : .orbit }
            let r = sceneRadius
            goal { c in
                if e.modifierFlags.contains(.command) {
                    c.zoom(exp(fy * 0.006 * n.zoomSpeed))
                } else if c.mode == .fly {
                    c.look(dx: fx * 0.8, dy: fy * 0.8 * vSign, speed: n.rotateSpeed)
                } else if mode == .pan {
                    c.pan(dx: fx, dy: fy, viewHeight: Float(bounds.height), speed: n.panSpeed, sceneRadius: r)
                } else {
                    c.orbit(dx: fx * 0.8, dy: fy * 0.8 * vSign, speed: n.rotateSpeed)
                }
            }
        } else {
            // Mouse wheel: one notch ≈ 8 % zoom (fly mode: move forward/back).
            let notches = Float(e.scrollingDeltaY)
            probeFlyDistance()
            let step = flyStep  // read before goal{}: flyStep takes the renderer lock that goal{} holds
            goal { c in
                if c.mode == .fly { c.fly(SIMD3(0, 0, notches), speed: step * 0.15) }
                else { c.zoom(exp(-notches * 0.08 * n.zoomSpeed)) }
            }
        }
    }

    @objc func onPinch(_ g: NSMagnificationGestureRecognizer) {
        let m = Float(g.magnification); g.magnification = 0
        let n = nav
        probeFlyDistance()
        let step = flyStep  // read before goal{} (see scrollWheel)
        goal { c in
            if c.mode == .fly { c.fly(SIMD3(0, 0, m * 4), speed: step) } else { c.zoom(exp(-m * n.zoomSpeed)) }
        }
    }
    @objc func onRotate(_ g: NSRotationGestureRecognizer) {
        let r = Float(g.rotation); g.rotation = 0
        let n = nav
        goal { $0.orbit(dx: -r * 57, dy: 0, speed: n.rotateSpeed) }
    }

    private var sceneRadius: Float? { renderer?.scene?.radius }
    private var lastDepthProbe = Date.distantPast

    /// Fly mode: re-measure the working distance (to the surface straight ahead) a few times a second, so
    /// speed drops as you approach something and rises again in open space.
    private func probeFlyDistance() {
        guard Date().timeIntervalSince(lastDepthProbe) > 0.25, let model, let r = renderer else { return }
        lastDepthProbe = Date()
        let cam = r.goalCamera
        guard cam.mode == .fly, let p = model.surfaceAtViewCentre(cam) else { return }
        let d = max(simd_distance(cam.eye, p), 0.005)
        goal { c in
            let e = c.eye
            c.distance = d
            c.target = e - c.offsetDir * d     // same eye, same view direction
        }
    }

    /// Distance travelled per second at fly speed 1: a quarter of the working distance (how far away what
    /// you're looking at is), so flying is slow up close and fast across a big scene — the same feel as orbit
    /// zoom, which is relative too. From the framed overview this equals the old quarter-scene-per-second.
    private var flyStep: Float {
        let cam = renderer?.goalCamera ?? ViewerCamera()
        return cam.workingDistance(sceneRadius: sceneRadius) * 0.25 * nav.flySpeed
    }

    /// Double-click: orbit around the splat under the cursor.
    private func focus(at p: CGPoint) {
        guard let sc = renderer?.scene, let r = renderer else { return }
        let cam = r.goalCamera
        let w = Float(bounds.width), h = Float(bounds.height)
        let k = cam.intrinsics(width: w, height: h)
        let m = cam.worldToCamera
        var bestZ: Float = .infinity
        var best: SIMD3<Float>?
        let pos = sc.positionsPointer, fl = sc.flagsPointer
        let px = Float(p.x), py = Float(p.y)
        for i in stride(from: 0, to: sc.count, by: max(1, sc.count / 400_000)) where fl[i] & 1 == 0 && pos[i].w > 0.3 {
            let t = m * SIMD4(pos[i].x, pos[i].y, pos[i].z, 1)
            guard t.z > cam.near else { continue }
            let u = k.x * t.x / t.z + k.z - px, v = k.y * t.y / t.z + k.w - py
            if u * u + v * v < 100 && t.z < bestZ { bestZ = t.z; best = SIMD3(pos[i].x, pos[i].y, pos[i].z) }
        }
        guard let b = best else { return }
        goal { c in
            // Keep the eye where it is and re-aim the orbit at the picked point.
            let e = c.eye
            c.target = b
            c.distance = max(simd_distance(e, b), 0.01)
            let d = simd_normalize(e - b)
            let up = simd_normalize(c.up)
            let ref: SIMD3<Float> = abs(up.z) < 0.9 ? SIMD3(0, 0, 1) : SIMD3(1, 0, 0)
            let right = simd_normalize(simd_cross(ref, up)), fwd = simd_cross(up, right)
            c.pitch = asin(min(max(simd_dot(d, up), -1), 1))
            c.yaw = atan2(simd_dot(d, right), simd_dot(d, fwd))
        }
    }

    // MARK: keyboard

    override func keyDown(with e: NSEvent) {
        keysDown.insert(e.keyCode)
        modifiers = e.modifierFlags
        guard let ch = e.charactersIgnoringModifiers?.lowercased() else { return }
        switch ch {
        case "f": model?.frameScene()
        case "r" where !e.modifierFlags.contains(.command): model?.resetCamera()
        case "1": model?.viewer.displayMode = .color
        case "2": model?.viewer.displayMode = .depth
        case "3": model?.viewer.displayMode = .centers
        case "4": model?.viewer.displayMode = .rings
        case "5": model?.viewer.displayMode = .opacity
        case "6": model?.viewer.displayMode = .coverage
        case "?", "/": model?.showShortcuts.toggle()
        case "\t": model?.cameraMode = (model?.cameraMode == .orbit) ? .fly : .orbit
        default:
            if e.keyCode == 51 || e.keyCode == 117 { model?.deleteSelected() } // delete / forward delete
        }
    }
    override func keyUp(with e: NSEvent) { keysDown.remove(e.keyCode); modifiers = e.modifierFlags }
    override func flagsChanged(with e: NSEvent) { modifiers = e.modifierFlags }
    override func resignFirstResponder() -> Bool { keysDown.removeAll(); return true }

    private func applyFlyKeys(_ dt: Double) {
        guard !keysDown.isEmpty, renderer != nil else { return }
        var v = SIMD3<Float>(0, 0, 0)
        // ANSI key codes: W 13, A 0, S 1, D 2, Q 12, E 14, arrows 123–126
        if keysDown.contains(13) || keysDown.contains(126) { v.z += 1 }
        if keysDown.contains(1) || keysDown.contains(125) { v.z -= 1 }
        if keysDown.contains(0) || keysDown.contains(123) { v.x -= 1 }
        if keysDown.contains(2) || keysDown.contains(124) { v.x += 1 }
        if keysDown.contains(14) { v.y += 1 }
        if keysDown.contains(12) { v.y -= 1 }
        guard simd_length(v) > 0 else { return }
        var mult: Float = 1
        if modifiers.contains(.shift) { mult *= 4 }
        if modifiers.contains(.option) { mult *= 0.25 }
        probeFlyDistance()
        let step = flyStep * Float(min(dt, 0.1)) * mult
        goal { $0.fly(simd_normalize(v), speed: step) }
    }
}

struct ViewerView: NSViewRepresentable {
    let model: AppModel
    var tool: ViewerTool
    var brushRadius: CGFloat

    func makeNSView(context: Context) -> SplatMTKView {
        let v = SplatMTKView(frame: .zero, device: model.renderer.ctx.device)
        v.renderer = model.renderer
        v.model = model
        v.delegate = model.renderer
        v.configure()
        return v
    }

    func updateNSView(_ v: SplatMTKView, context: Context) {
        v.tool = tool
        v.brushRadius = brushRadius
    }
}
