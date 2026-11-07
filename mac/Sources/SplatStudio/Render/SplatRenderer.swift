import Foundation
import Metal
import MetalKit
import QuartzCore
import simd

/// Mirrors `CameraUniforms` in CommonShaders (176 bytes).
struct CameraUniforms {
    var worldToCam = matrix_identity_float4x4
    var intr = SIMD4<Float>()
    var viewport = SIMD4<Float>()
    var camPos = SIMD4<Float>()
    var count: UInt32 = 0
    var shDegree: UInt32 = 0
    var splatScale: Float = 1
    var flags: UInt32 = 0
    var cropMin = SIMD4<Float>()
    var cropMax = SIMD4<Float>()
    var misc = SIMD4<Float>()

    static func make(worldToCam: simd_float4x4, intr: SIMD4<Float>, width: Int, height: Int, count: Int, shDegree: Int) -> CameraUniforms {
        var u = CameraUniforms()
        u.worldToCam = worldToCam
        u.intr = intr
        let w = Float(width), h = Float(height)
        u.viewport = SIMD4(w, h, 1.3 * (0.5 * w / intr.x), 1.3 * (0.5 * h / intr.y))
        let inv = worldToCam.inverse
        u.camPos = SIMD4(inv[3][0], inv[3][1], inv[3][2], 1)
        u.count = UInt32(count)
        u.shDegree = UInt32(shDegree)
        u.misc = SIMD4(1.0 / 255.0, 1, 0.3, 0)
        return u
    }
}

struct LineVertex { var pos: SIMD3<Float>; var color: SIMD4<Float> }

/// User-facing viewer options. Every field is documented in SettingsCatalog.
struct ViewerSettings: Equatable, Codable {
    enum DisplayMode: Int, CaseIterable, Identifiable, Codable {
        case color = 0, depth = 1, centers = 2, rings = 3, opacity = 4
        var id: Int { rawValue }
        var title: String {
            switch self {
            case .color: return "Colour"
            case .depth: return "Depth"
            case .centers: return "Centres"
            case .rings: return "Ellipses"
            case .opacity: return "Opacity"
            }
        }
    }
    var displayMode: DisplayMode = .color
    var maxSHDegree = 3
    var splatScale: Float = 1
    var background = SIMD3<Float>(0.06, 0.06, 0.07)
    var sixteenBitSort = true
    var alphaCutoff: Float = 1.0 / 255.0
    var renderScale: Float = 1
    var showCameras = true
    var showGrid = false
    var showCropBox = false
    var cropEnabled = false
    var cropMin = SIMD3<Float>(-1, -1, -1)
    var cropMax = SIMD3<Float>(1, 1, 1)
    var highlightSelection = true
    var lazySort = true
    var antialiasOverride: Bool? = nil
}

struct RenderStats: Equatable {
    var fps: Double = 0
    var gpuMs: Double = 0
    var visible: Int = 0
    var total: Int = 0
}

/// Draws a `SplatScene` into an MTKView: project → sort → instanced quads (+ line overlays).
final class SplatRenderer: NSObject, MTKViewDelegate, @unchecked Sendable {
    let ctx: MetalContext
    private let sorter: GPURadixSort
    private let splatPSO: MTLRenderPipelineState
    private let linePSO: MTLRenderPipelineState
    private var verts: MTLBuffer?
    private var keys: MTLBuffer?
    private var vals: MTLBuffer?
    private let args: MTLBuffer
    private var capacity = 0
    private let lock = NSLock()

    // state guarded by lock
    private var _scene: SplatScene?
    private var _camera = ViewerCamera()
    /// Where interactive input wants the camera to be; the drawn camera eases towards it.
    private var _goal = ViewerCamera()
    /// Ease-in time constant for navigation (0 = off).
    var smoothing: Float = 0.08
    private var _settings = ViewerSettings()
    private var _lines: [LineVertex] = []
    private var lastKey: (ViewerCamera, ViewerSettings, ObjectIdentifier?, CGSize)? = nil
    private var editGeneration = 0
    private var lastEditGeneration = -1
    private var lastFrameTime = CACurrentMediaTime()
    private var fpsAccum: (Double, Int) = (0, 0)

    var onStats: ((RenderStats) -> Void)?
    /// Called on the render thread before each frame (camera animation / fly keys).
    var onBeforeFrame: ((Double) -> Void)?

    init(ctx: MetalContext = .shared) throws {
        self.ctx = ctx
        sorter = GPURadixSort(ctx: ctx)
        args = ctx.buffer(16, label: "drawArgs")
        let d = ctx.device
        let pd = MTLRenderPipelineDescriptor()
        pd.vertexFunction = ctx.library.makeFunction(name: "splat_vertex")
        pd.fragmentFunction = ctx.library.makeFunction(name: "splat_fragment")
        pd.colorAttachments[0].pixelFormat = .bgra8Unorm
        let ca = pd.colorAttachments[0]!
        ca.isBlendingEnabled = true
        ca.rgbBlendOperation = .add; ca.alphaBlendOperation = .add
        ca.sourceRGBBlendFactor = .one; ca.sourceAlphaBlendFactor = .one
        ca.destinationRGBBlendFactor = .oneMinusSourceAlpha; ca.destinationAlphaBlendFactor = .oneMinusSourceAlpha
        splatPSO = try d.makeRenderPipelineState(descriptor: pd)
        let ld = MTLRenderPipelineDescriptor()
        ld.vertexFunction = ctx.library.makeFunction(name: "line_vertex")
        ld.fragmentFunction = ctx.library.makeFunction(name: "line_fragment")
        ld.colorAttachments[0].pixelFormat = .bgra8Unorm
        let la = ld.colorAttachments[0]!
        la.isBlendingEnabled = true
        la.sourceRGBBlendFactor = .one; la.destinationRGBBlendFactor = .oneMinusSourceAlpha
        la.sourceAlphaBlendFactor = .one; la.destinationAlphaBlendFactor = .oneMinusSourceAlpha
        linePSO = try d.makeRenderPipelineState(descriptor: ld)
        super.init()
    }

    var scene: SplatScene? {
        get { lock.lock(); defer { lock.unlock() }; return _scene }
        set { lock.lock(); _scene = newValue; lock.unlock() }
    }
    /// Setting jumps immediately (no easing) — used for framing, reset and "look through camera".
    var camera: ViewerCamera {
        get { lock.lock(); defer { lock.unlock() }; return _camera }
        set { lock.lock(); _camera = newValue; _goal = newValue; lock.unlock() }
    }
    /// Interactive navigation target (eased).
    var goalCamera: ViewerCamera {
        get { lock.lock(); defer { lock.unlock() }; return _goal }
        set { lock.lock(); _goal = newValue; lock.unlock() }
    }
    func mutateGoal(_ f: (inout ViewerCamera) -> Void) {
        lock.lock(); f(&_goal); if smoothing <= 0 { _camera = _goal }; lock.unlock()
    }

    /// Frees the per-frame GPU buffers (Home / unload).
    func releaseBuffers() {
        lock.lock(); _scene = nil; _lines = []; lock.unlock()
        verts = nil; keys = nil; vals = nil; capacity = 0; lastKey = nil
    }
    var settings: ViewerSettings {
        get { lock.lock(); defer { lock.unlock() }; return _settings }
        set { lock.lock(); _settings = newValue; lock.unlock() }
    }
    var lines: [LineVertex] {
        get { lock.lock(); defer { lock.unlock() }; return _lines }
        set { lock.lock(); _lines = newValue; lock.unlock() }
    }
    /// Call after changing per-splat flags so the next frame re-projects.
    func invalidate() { lock.lock(); editGeneration += 1; lock.unlock() }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        let now = CACurrentMediaTime()
        let dt = now - lastFrameTime
        lastFrameTime = now
        onBeforeFrame?(dt)
        lock.lock()
        if smoothing > 0 && !_camera.isClose(to: _goal) {
            let t = Float(1 - exp(-dt / Double(smoothing)))
            _camera = _camera.interpolated(to: _goal, t: min(max(t, 0), 1))
        } else {
            _camera = _goal
        }
        let scene = _scene, cam = _camera, st = _settings, lineVerts = _lines, gen = editGeneration
        lock.unlock()

        // Render resolution scaling.
        let scale = CGFloat(st.renderScale) * (view.window?.backingScaleFactor ?? 2)
        let target = CGSize(width: max(1, view.bounds.width * scale), height: max(1, view.bounds.height * scale))
        if view.drawableSize != target { view.drawableSize = target }

        guard let drawable = view.currentDrawable, let rpd = view.currentRenderPassDescriptor,
              let cb = ctx.queue.makeCommandBuffer() else { return }
        rpd.colorAttachments[0].clearColor = MTLClearColor(red: Double(st.background.x), green: Double(st.background.y), blue: Double(st.background.z), alpha: 1)
        rpd.colorAttachments[0].loadAction = .clear
        let size = view.drawableSize
        let w = Int(size.width), h = Int(size.height)
        var uniforms = CameraUniforms()
        var visibleDraw = false
        if let scene, scene.count > 0 {
            ensure(scene.count)
            uniforms = CameraUniforms.make(worldToCam: cam.worldToCamera, intr: cam.intrinsics(width: Float(w), height: Float(h)),
                                           width: w, height: h, count: scene.count, shDegree: min(st.maxSHDegree, scene.shDegree))
            uniforms.splatScale = st.splatScale
            let aa = st.antialiasOverride ?? scene.antialiased
            uniforms.flags = (aa ? 1 : 0) | (st.cropEnabled ? 2 : 0) | (UInt32(st.displayMode.rawValue) << 8)
            uniforms.cropMin = SIMD4(st.cropMin, cam.near)
            uniforms.cropMax = SIMD4(st.cropMax, cam.far)
            uniforms.misc = SIMD4(st.alphaCutoff, st.highlightSelection ? 1 : 0, 0.3, 0)
            let key = (cam, st, ObjectIdentifier(scene), size)
            let unchanged = st.lazySort && lastKey.map { $0.0 == key.0 && $0.1 == key.1 && $0.2 == key.2 && $0.3 == key.3 } == true && gen == lastEditGeneration
            if !unchanged {
                let enc = cb.makeComputeCommandEncoder()!
                enc.setBuffer(args, offset: 0, index: 0)
                enc.dispatch1D(ctx.pipeline("reset_draw_args"), count: 1)
                enc.memoryBarrier(scope: .buffers)
                enc.setBuffer(scene.gpos, offset: 0, index: 0)
                enc.setBuffer(scene.gcov, offset: 0, index: 1)
                enc.setBuffer(scene.gsh, offset: 0, index: 2)
                enc.setBuffer(scene.gflags, offset: 0, index: 3)
                enc.setBytes(&uniforms, length: MemoryLayout<CameraUniforms>.stride, index: 4)
                enc.setBuffer(verts!, offset: 0, index: 5)
                enc.setBuffer(keys!, offset: 0, index: 6)
                enc.setBuffer(vals!, offset: 0, index: 7)
                enc.setBuffer(args, offset: 0, index: 8)
                enc.set(UInt32(st.sixteenBitSort ? 1 : 0), 9)
                enc.dispatch1D(ctx.pipeline("project_splats"), count: scene.count)
                enc.memoryBarrier(scope: .buffers)
                sorter.encode(enc, keys: keys!, vals: vals!, count: scene.count, bits: st.sixteenBitSort ? 16 : 32)
                enc.endEncoding()
                lastKey = key
                lastEditGeneration = gen
            }
            visibleDraw = true
        }

        let renc = cb.makeRenderCommandEncoder(descriptor: rpd)!
        if visibleDraw {
            renc.setRenderPipelineState(splatPSO)
            renc.setVertexBuffer(verts!, offset: 0, index: 0)
            renc.setVertexBuffer(vals!, offset: 0, index: 1)
            renc.setVertexBytes(&uniforms, length: MemoryLayout<CameraUniforms>.stride, index: 2)
            renc.setFragmentBytes(&uniforms, length: MemoryLayout<CameraUniforms>.stride, index: 0)
            renc.drawPrimitives(type: .triangleStrip, indirectBuffer: args, indirectBufferOffset: 0)
        }
        if !lineVerts.isEmpty {
            var vp = cam.viewProjection(width: Float(w), height: Float(h))
            renc.setRenderPipelineState(linePSO)
            lineVerts.withUnsafeBytes { raw in
                if raw.count <= 4096 {
                    renc.setVertexBytes(raw.baseAddress!, length: raw.count, index: 0)
                } else {
                    let b = ctx.device.makeBuffer(bytes: raw.baseAddress!, length: raw.count, options: .storageModeShared)!
                    renc.setVertexBuffer(b, offset: 0, index: 0)
                }
            }
            renc.setVertexBytes(&vp, length: 64, index: 1)
            renc.drawPrimitives(type: .line, vertexStart: 0, vertexCount: lineVerts.count)
        }
        renc.endEncoding()
        cb.present(drawable)
        let total = scene?.count ?? 0
        let argsBuf = args
        cb.addCompletedHandler { [weak self] cb in
            guard let self else { return }
            let gpu = (cb.gpuEndTime - cb.gpuStartTime) * 1000
            self.fpsAccum.0 += dt; self.fpsAccum.1 += 1
            if self.fpsAccum.0 > 0.5 {
                let fps = Double(self.fpsAccum.1) / self.fpsAccum.0
                self.fpsAccum = (0, 0)
                let vis = Int(argsBuf.contents().bindMemory(to: UInt32.self, capacity: 4)[1])
                let s = RenderStats(fps: fps, gpuMs: gpu, visible: vis, total: total)
                DispatchQueue.main.async { self.onStats?(s) }
            }
        }
        cb.commit()
    }

    private func ensure(_ n: Int) {
        guard n > capacity else { return }
        let cap = n + n / 8
        verts = ctx.buffer(cap * 32, label: "splatVerts")
        keys = ctx.buffer(cap * 4, label: "splatKeys")
        vals = ctx.buffer(cap * 4, label: "splatVals")
        capacity = cap
        lastKey = nil
    }

    // MARK: offscreen rendering (screenshots, turntable videos)

    /// Renders one frame into a new BGRA8 texture of the given size.
    func renderOffscreen(width: Int, height: Int, camera cam: ViewerCamera, settings st: ViewerSettings) -> MTLTexture? {
        guard let scene, scene.count > 0 else { return nil }
        let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false)
        td.usage = [.renderTarget, .shaderRead]
        td.storageMode = .shared
        guard let tex = ctx.device.makeTexture(descriptor: td) else { return nil }
        let localVerts = ctx.buffer(scene.count * 32), localKeys = ctx.buffer(scene.count * 4), localVals = ctx.buffer(scene.count * 4)
        let localArgs = ctx.buffer(16)
        let a = localArgs.contents().bindMemory(to: UInt32.self, capacity: 4)
        a[0] = 4; a[1] = 0; a[2] = 0; a[3] = 0
        var u = CameraUniforms.make(worldToCam: cam.worldToCamera, intr: cam.intrinsics(width: Float(width), height: Float(height)),
                                    width: width, height: height, count: scene.count, shDegree: min(st.maxSHDegree, scene.shDegree))
        u.splatScale = st.splatScale
        u.flags = ((st.antialiasOverride ?? scene.antialiased) ? 1 : 0) | (st.cropEnabled ? 2 : 0) | (UInt32(st.displayMode.rawValue) << 8)
        u.cropMin = SIMD4(st.cropMin, cam.near); u.cropMax = SIMD4(st.cropMax, cam.far)
        u.misc = SIMD4(st.alphaCutoff, 0, 0.3, 0)
        let cb = ctx.queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setBuffer(scene.gpos, offset: 0, index: 0); enc.setBuffer(scene.gcov, offset: 0, index: 1)
        enc.setBuffer(scene.gsh, offset: 0, index: 2); enc.setBuffer(scene.gflags, offset: 0, index: 3)
        enc.setBytes(&u, length: MemoryLayout<CameraUniforms>.stride, index: 4)
        enc.setBuffer(localVerts, offset: 0, index: 5); enc.setBuffer(localKeys, offset: 0, index: 6)
        enc.setBuffer(localVals, offset: 0, index: 7); enc.setBuffer(localArgs, offset: 0, index: 8)
        enc.set(UInt32(0), 9)
        enc.dispatch1D(ctx.pipeline("project_splats"), count: scene.count)
        enc.memoryBarrier(scope: .buffers)
        let s = GPURadixSort(ctx: ctx)
        s.encode(enc, keys: localKeys, vals: localVals, count: scene.count, bits: 32)
        enc.endEncoding()
        let rpd = MTLRenderPassDescriptor()
        rpd.colorAttachments[0].texture = tex
        rpd.colorAttachments[0].loadAction = .clear
        rpd.colorAttachments[0].storeAction = .store
        rpd.colorAttachments[0].clearColor = MTLClearColor(red: Double(st.background.x), green: Double(st.background.y), blue: Double(st.background.z), alpha: 1)
        let renc = cb.makeRenderCommandEncoder(descriptor: rpd)!
        renc.setRenderPipelineState(splatPSO)
        renc.setVertexBuffer(localVerts, offset: 0, index: 0)
        renc.setVertexBuffer(localVals, offset: 0, index: 1)
        renc.setVertexBytes(&u, length: MemoryLayout<CameraUniforms>.stride, index: 2)
        renc.setFragmentBytes(&u, length: MemoryLayout<CameraUniforms>.stride, index: 0)
        renc.drawPrimitives(type: .triangleStrip, indirectBuffer: localArgs, indirectBufferOffset: 0)
        renc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        return tex
    }
}
