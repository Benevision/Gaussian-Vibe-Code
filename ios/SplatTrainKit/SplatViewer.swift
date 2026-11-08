import Foundation
import MetalKit
import simd
import SwiftUI
import UIKit

/// Views a trained splat (INRIA 3DGS PLY) with the Mac's Metal renderer. One finger orbits, two fingers pan,
/// pinch zooms, double tap goes back to the starting view (a real capture viewpoint when the dataset is known).
public struct SplatViewer: UIViewRepresentable {
    public let plyURL: URL
    /// Directory with the dataset.json the splat was trained from (for the starting viewpoint and world up).
    public var datasetURL: URL?
    public var onLoaded: ((Result<Int, Error>) -> Void)?

    public init(plyURL: URL, datasetURL: URL? = nil, onLoaded: ((Result<Int, Error>) -> Void)? = nil) {
        self.plyURL = plyURL; self.datasetURL = datasetURL; self.onLoaded = onLoaded
    }

    public func makeUIView(context: Context) -> UIView {
        guard let r = try? SplatRenderer() else {
            let l = UILabel(); l.text = "Metal is not available"; l.textAlignment = .center; l.textColor = .secondaryLabel
            return l
        }
        let v = SplatTouchView(renderer: r)
        v.load(plyURL, dataset: datasetURL, onLoaded: onLoaded)
        return v
    }

    public func updateUIView(_ v: UIView, context: Context) {
        guard let v = v as? SplatTouchView, v.loadedURL != plyURL else { return }
        v.load(plyURL, dataset: datasetURL, onLoaded: onLoaded)
    }

    public static func dismantleUIView(_ v: UIView, coordinator: ()) {
        (v as? SplatTouchView)?.renderer.releaseBuffers()
    }
}

final class SplatTouchView: MTKView {
    let renderer: SplatRenderer
    private(set) var loadedURL: URL?
    private var home = ViewerCamera()
    private var radius: Float = 1
    private var lastPan = CGPoint.zero
    private var lastPan2 = CGPoint.zero
    private var lastPinch: CGFloat = 1

    init(renderer r: SplatRenderer) {
        renderer = r
        super.init(frame: .zero, device: r.ctx.device)
        colorPixelFormat = .bgra8Unorm
        framebufferOnly = true    // never read back on the phone
        autoResizeDrawable = false
        preferredFramesPerSecond = 60
        delegate = renderer
        var st = renderer.settings
        st.showCameras = false
        st.background = SIMD3(0.05, 0.05, 0.06)
        // full resolution: a drawable smaller than bounds × scale isn't composited (Simulator, at least)
        st.renderScale = 1
        renderer.settings = st
        let one = UIPanGestureRecognizer(target: self, action: #selector(onPan(_:)))
        one.maximumNumberOfTouches = 1
        let two = UIPanGestureRecognizer(target: self, action: #selector(onPan2(_:)))
        two.minimumNumberOfTouches = 2
        let pinch = UIPinchGestureRecognizer(target: self, action: #selector(onPinch(_:)))
        let dbl = UITapGestureRecognizer(target: self, action: #selector(onDoubleTap))
        dbl.numberOfTapsRequired = 2
        for g in [one, two, pinch, dbl] as [UIGestureRecognizer] { addGestureRecognizer(g) }
        two.delegate = self; pinch.delegate = self
    }

    required init(coder: NSCoder) { fatalError("not used") }

    /// The renderer sizes the drawable inside draw(in:), but with autoResizeDrawable off an iOS MTKView whose
    /// drawable is still 0×0 never calls draw — so give it its first size here.
    override func layoutSubviews() {
        super.layoutSubviews()
        let s = contentScaleFactor * CGFloat(renderer.settings.renderScale)
        let target = CGSize(width: max(1, bounds.width * s), height: max(1, bounds.height * s))
        if drawableSize != target { drawableSize = target }
    }

    func load(_ url: URL, dataset: URL?, onLoaded: ((Result<Int, Error>) -> Void)?) {
        loadedURL = url
        let r = renderer
        Task.detached(priority: .userInitiated) {
            do {
                let cloud = try PLYCodec.readGaussians(url: url)
                let scene = SplatScene(cloud: cloud)
                let ds = dataset.flatMap { try? PreparedDataset.load($0) }
                await MainActor.run {
                    r.scene = scene
                    self.frame(scene: scene, dataset: ds)
                    onLoaded?(.success(cloud.count))
                }
            } catch {
                await MainActor.run { onLoaded?(.failure(error)) }
            }
        }
    }

    private func frame(scene: SplatScene, dataset: PreparedDataset?) {
        radius = scene.radius
        home = Self.startCamera(scene: scene, dataset: dataset)
        renderer.camera = home
    }

    /// A real capture viewpoint (the camera nearest the middle of the capture path), level with the scene;
    /// without a dataset, the bounding sphere from outside.
    static func startCamera(scene: SplatScene, dataset: PreparedDataset?) -> ViewerCamera {
        var c = ViewerCamera()
        if let ds = dataset, ds.manifest.upVector.count == 3 {
            let up = ds.upVectorSIMD
            let cams = ds.manifest.cameras.filter { $0.holdout != true }
            let mid = cams.isEmpty ? SIMD3<Float>.zero : cams.map(\.center).reduce(.zero, +) / Float(cams.count)
            let ctr = SIMD3(ds.manifest.sceneCenter[0], ds.manifest.sceneCenter[1], ds.manifest.sceneCenter[2])
            if let cam = cams.min(by: { simd_distance($0.center, mid) < simd_distance($1.center, mid) }) {
                c.adopt(cameraToWorld: cam.c2w, focusDistance: max(simd_distance(cam.center, ctr), 0.1),
                        fovY: 2 * atan(Float(cam.height) / (2 * cam.fy)), worldUp: up)
            } else {
                c.up = up; c.frame(center: scene.center, radius: scene.radius)
            }
        } else {
            c.up = Orientation.standardUp
            c.pitch = 0.25
            c.frame(center: scene.center, radius: scene.radius)
        }
        return c
    }

    @objc private func onPan(_ g: UIPanGestureRecognizer) {
        let p = g.translation(in: self)
        if g.state == .began { lastPan = .zero }
        let dx = Float(p.x - lastPan.x), dy = Float(p.y - lastPan.y)
        lastPan = p
        renderer.mutateGoal { $0.orbit(dx: dx, dy: dy, speed: 1.2) }
    }

    @objc private func onPan2(_ g: UIPanGestureRecognizer) {
        let p = g.translation(in: self)
        if g.state == .began { lastPan2 = .zero }
        let dx = Float(p.x - lastPan2.x), dy = Float(p.y - lastPan2.y)
        lastPan2 = p
        let h = Float(bounds.height), r = radius
        renderer.mutateGoal { $0.pan(dx: dx, dy: dy, viewHeight: h, sceneRadius: r) }
    }

    @objc private func onPinch(_ g: UIPinchGestureRecognizer) {
        if g.state == .began { lastPinch = 1 }
        let f = Float(lastPinch / max(g.scale, 0.01))
        lastPinch = g.scale
        renderer.mutateGoal { $0.zoom(f) }
    }

    @objc private func onDoubleTap() { renderer.camera = home }
}

extension SplatTouchView: UIGestureRecognizerDelegate {
    func gestureRecognizer(_ g: UIGestureRecognizer, shouldRecognizeSimultaneouslyWith other: UIGestureRecognizer) -> Bool { true }
}

#if DEBUG
/// Debug: renders a splat offscreen from one of its dataset cameras (sixteen- and 32-bit sort) to PNGs,
/// to tell renderer problems from view problems (SplatSelfTest).
public enum SplatDebugRender {
    public static func render(ply: URL, dataset: URL, view: Int, to dir: URL) throws {
        let cloud = try PLYCodec.readGaussians(url: ply)
        let ds = try PreparedDataset.load(dataset)
        let r = try SplatRenderer()
        r.scene = SplatScene(cloud: cloud)
        let c = ds.manifest.cameras[min(max(0, view), ds.manifest.cameras.count - 1)]
        var cam = ViewerCamera()
        cam.adopt(cameraToWorld: c.c2w, focusDistance: 1, fovY: 2 * atan(Float(c.height) / (2 * c.fy)), worldUp: ds.upVectorSIMD)
        let start = SplatTouchView.startCamera(scene: r.scene!, dataset: ds)
        for (name, cm, bits16) in [("16", cam, true), ("32", cam, false), ("start", start, true)] {
            var st = ViewerSettings(); st.sixteenBitSort = bits16
            guard let tex = r.renderOffscreen(width: c.width, height: c.height, camera: cm, settings: st) else { continue }
            var bytes = [UInt8](repeating: 0, count: c.width * c.height * 4)
            tex.getBytes(&bytes, bytesPerRow: c.width * 4, from: MTLRegionMake2D(0, 0, c.width, c.height), mipmapLevel: 0)
            for i in stride(from: 0, to: bytes.count, by: 4) { bytes.swapAt(i, i + 2) }   // BGRA → RGBA
            try SystemBenchmark.writeJPEG(bytes, width: c.width, height: c.height, to: dir.appendingPathComponent("render-\(name).jpg"))
        }
    }
}
#endif
