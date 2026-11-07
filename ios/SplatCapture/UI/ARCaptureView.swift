import ARKit
import SceneKit
import SwiftUI

/// Camera feed (ARSCNView) with the fused LiDAR point cloud drawn as dots.
struct ARCaptureView: UIViewRepresentable {
    let model: CaptureViewModel
    let settings: CaptureSettings

    func makeUIView(context: Context) -> ARSCNView {
        let view = ARSCNView(frame: .zero)
        view.session = model.engine.session
        view.automaticallyUpdatesLighting = false
        view.rendersCameraGrain = false
        view.rendersMotionBlur = false
        view.antialiasingMode = .none
        view.preferredFramesPerSecond = 60
        view.scene = SCNScene()
        let renderer = PointCloudRenderer(root: view.scene.rootNode)
        renderer.apply(settings: settings)
        context.coordinator.renderer = renderer
        model.renderer = renderer
        return view
    }

    func updateUIView(_ view: ARSCNView, context: Context) {
        context.coordinator.renderer?.apply(settings: settings)
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator {
        var renderer: PointCloudRenderer?
    }
}

/// Owns the SceneKit node that displays the accumulated point cloud.
@MainActor
final class PointCloudRenderer {
    private let node = SCNNode()
    private var lastSnapshot: PointSnapshot?
    private var style: OverlayStyle = .color
    private var pointSize: Float = 3
    private var visible = true

    init(root: SCNNode) {
        node.renderingOrder = 10
        root.addChildNode(node)
    }

    func apply(settings: CaptureSettings) {
        let changed = settings.overlayStyle != style || settings.overlayPointSize != pointSize
        style = settings.overlayStyle
        pointSize = settings.overlayPointSize
        visible = settings.showOverlay
        node.isHidden = !visible
        if changed, let s = lastSnapshot { update(s) }
    }

    func clear() {
        lastSnapshot = nil
        node.geometry = nil
    }

    func update(_ snapshot: PointSnapshot) {
        lastSnapshot = snapshot
        guard snapshot.count > 0 else { node.geometry = nil; return }
        let vertices = SCNGeometrySource(
            data: snapshot.positions, semantic: .vertex, vectorCount: snapshot.count,
            usesFloatComponents: true, componentsPerVector: 3, bytesPerComponent: 4,
            dataOffset: 0, dataStride: 12)
        var indices = Data(count: snapshot.count * 4)
        indices.withUnsafeMutableBytes { raw in
            let p = raw.bindMemory(to: UInt32.self)
            for i in 0..<snapshot.count { p[i] = UInt32(i) }
        }
        let element = SCNGeometryElement(data: indices, primitiveType: .point,
                                         primitiveCount: snapshot.count, bytesPerIndex: 4)
        element.pointSize = CGFloat(pointSize)
        element.minimumPointScreenSpaceRadius = CGFloat(max(1, pointSize * 0.5))
        element.maximumPointScreenSpaceRadius = CGFloat(pointSize * 2)

        var sources = [vertices]
        let material = SCNMaterial()
        material.lightingModel = .constant
        material.isDoubleSided = true
        material.writesToDepthBuffer = true
        switch style {
        case .color:
            sources.append(SCNGeometrySource(
                data: snapshot.colors, semantic: .color, vectorCount: snapshot.count,
                usesFloatComponents: true, componentsPerVector: 3, bytesPerComponent: 4,
                dataOffset: 0, dataStride: 12))
            material.diffuse.contents = UIColor.white
        case .highlight:
            material.diffuse.contents = UIColor(red: 0.25, green: 0.95, blue: 0.75, alpha: 0.85)
        }
        let geometry = SCNGeometry(sources: sources, elements: [element])
        geometry.materials = [material]
        node.geometry = geometry
    }
}
