import ARKit
import AVFoundation
import Foundation
import ImageIO
import simd

enum Guidance: Sendable, Equatable {
    case none
    case initializing
    case relocalizing
    case tooFast
    case tooDark
    case lowFeatures
    case tooClose
    case blurry
    case backlog
    case trackingLost

    var message: String? {
        switch self {
        case .none: nil
        case .initializing: "Initialising — move the phone slowly side to side"
        case .relocalizing: "Relocalising — return to where you started"
        case .tooFast: "Moving too fast — slow down"
        case .tooDark: "Too dark — add light"
        case .lowFeatures: "Not enough detail — aim at textured surfaces"
        case .tooClose: "Too close — step back a little"
        case .blurry: "Frames are blurry — slow down or add light"
        case .backlog: "Saving… slow down for a moment"
        case .trackingLost: "Tracking lost — point at the scene and hold still"
        }
    }

    /// Warnings that indicate frames are not being captured.
    var isSevere: Bool {
        switch self {
        case .tooFast, .trackingLost, .relocalizing, .initializing, .blurry: true
        default: false
        }
    }
}

enum TrackingSummary: Sendable, Equatable {
    case normal, limited, notAvailable

    init(_ s: ARCamera.TrackingState) {
        switch s {
        case .normal: self = .normal
        case .limited: self = .limited
        case .notAvailable: self = .notAvailable
        }
    }

    var manifestValue: String {
        switch self {
        case .normal: "normal"
        case .limited: "limited"
        case .notAvailable: "notAvailable"
        }
    }
}

struct LiveStats: Sendable {
    var tracking: TrackingSummary = .notAvailable
    var guidance: Guidance = .initializing
    var keyframes: Int = 0
    var bytes: Int64 = 0
    var rejectedBlurry: Int = 0
    var cameraPosition: SIMD3<Float> = .zero
    var cameraForward: SIMD3<Float> = SIMD3(0, 0, -1)
    var hasDepth: Bool = false
    var centerDepth: Float?
}

struct KeyframeEvent: Sendable {
    var position: SIMD3<Float>
    var lookAt: SIMD3<Float>
}

/// Owns the ARSession, runs keyframe selection on the AR delegate queue and hands
/// keyframes to a `CaptureRecorder`. All mutable state below the "delegate queue"
/// marker is touched only on `delegateQueue`.
final class CaptureEngine: NSObject, ARSessionDelegate, @unchecked Sendable {
    let session = ARSession()
    private let delegateQueue = DispatchQueue(label: "splatcapture.ar", qos: .userInteractive)

    // Callbacks (set once from the main actor before the session runs).
    var onStats: (@Sendable (LiveStats) -> Void)?
    var onKeyframe: (@Sendable (KeyframeEvent) -> Void)?
    var onPoints: (@Sendable (PointSnapshot) -> Void)?
    var onAutoStop: (@Sendable (String) -> Void)?
    var onError: (@Sendable (String) -> Void)?

    // MARK: delegate-queue state
    private var settings = CaptureSettings()
    private var depthEnabled = false
    private var recorder: CaptureRecorder?
    private var selector = KeyframeSelector(settings: CaptureSettings())
    private var gate = SharpnessGate()
    private var prevTransform: simd_float4x4?
    private var prevTime: TimeInterval = 0
    private var angularSpeed: Float = 0
    private var linearSpeed: Float = 0
    private var lastStatsTime: TimeInterval = 0
    private var lastSpaceCheck: TimeInterval = 0
    private var rejectedBlurry = 0
    private var lastBlurReject: TimeInterval = -10
    private var lastBacklog: TimeInterval = -10
    private var highResInFlight = false
    private var centerDepth: Float?
    private var autoStopSent = false

    override init() {
        super.init()
        session.delegate = self
        session.delegateQueue = delegateQueue
    }

    // MARK: - Main-actor API

    /// Runs (or re-runs) world tracking with `settings`.
    @MainActor
    func run(settings: CaptureSettings, reset: Bool) {
        let config = ARWorldTrackingConfiguration()
        if settings.highResolutionStills,
           let hf = ARWorldTrackingConfiguration.recommendedVideoFormatForHighResolutionFrameCapturing {
            config.videoFormat = hf
        } else if let f = VideoFormatOption.format(for: settings.videoFormatID) {
            config.videoFormat = f
        }
        var semantics: ARConfiguration.FrameSemantics = []
        var depth = false
        if settings.recordDepth || settings.showOverlay {
            if settings.smoothedDepth, DeviceInfo.supportsSmoothedDepth {
                semantics.insert(.smoothedSceneDepth)
                depth = true
            } else if DeviceInfo.hasLiDAR {
                semantics.insert(.sceneDepth)
                depth = true
            }
        }
        config.frameSemantics = semantics
        config.planeDetection = []
        config.environmentTexturing = .none
        config.isLightEstimationEnabled = true
        config.isAutoFocusEnabled = settings.autoFocus
        let useDepth = depth && settings.recordDepth
        delegateQueue.async { [self] in
            self.settings = settings
            self.depthEnabled = useDepth
            self.selector.settings = settings
        }
        session.run(config, options: reset ? [.resetTracking, .removeExistingAnchors] : [])
    }

    @MainActor
    func pause() { session.pause() }

    /// Creates a new capture bundle and starts saving keyframes into it.
    @MainActor
    func startRecording(settings: CaptureSettings) throws -> CaptureRecorder {
        let (url, name) = CaptureStore.uniqueURL(for: CaptureStore.defaultName())
        let format = session.configuration?.videoFormat
        let pixels = format.map { Int($0.imageResolution.width * $0.imageResolution.height) } ?? 0
        let large = settings.highResolutionStills || pixels > 4_000_000
        let hasDepth = DeviceInfo.hasLiDAR && settings.recordDepth
        let r = try CaptureRecorder(directory: url, name: name, settings: settings,
                                    hasLiDAR: hasDepth, largeFrames: large)
        r.onPoints = onPoints
        if settings.lockExposureWhileRecording { setExposureLocked(true) }
        delegateQueue.async { [self] in
            self.settings = settings
            self.selector = KeyframeSelector(settings: settings)
            self.gate.reset()
            self.rejectedBlurry = 0
            self.autoStopSent = false
            self.highResInFlight = false
            self.recorder = r
        }
        return r
    }

    @MainActor
    func stopRecording(completion: @escaping @Sendable (Result<CaptureSummary, any Error>) -> Void) {
        setExposureLocked(false)
        delegateQueue.async { [self] in
            guard let r = recorder else {
                completion(.failure(CaptureError.noFrames))
                return
            }
            recorder = nil
            r.finish(completion: completion)
        }
    }

    @MainActor
    func discardRecording() {
        setExposureLocked(false)
        delegateQueue.async { [self] in
            recorder?.discard()
            recorder = nil
        }
    }

    @MainActor
    private func setExposureLocked(_ locked: Bool) {
        guard let device = ARWorldTrackingConfiguration.configurableCaptureDeviceForPrimaryCamera else { return }
        do {
            try device.lockForConfiguration()
            defer { device.unlockForConfiguration() }
            if locked {
                // Cap the shutter (default 1/120 s) before locking: auto-exposure picks ~1/60 s indoors, which
                // blurs most frames while walking. Raise ISO to keep the same brightness (a bit more grain).
                let cap = settings.maxShutterSeconds
                let cur = device.exposureDuration.seconds
                if cap > 0, cur.isFinite, cur > cap * 1.05, device.isExposureModeSupported(.custom) {
                    let fmt = device.activeFormat
                    let iso = min(fmt.maxISO, max(fmt.minISO, device.iso * Float(cur / cap)))
                    let minD = fmt.minExposureDuration.seconds, maxD = fmt.maxExposureDuration.seconds
                    let d = min(max(cap, minD), maxD)
                    device.setExposureModeCustom(duration: CMTime(seconds: d, preferredTimescale: 1_000_000), iso: iso, completionHandler: nil)
                } else if device.isExposureModeSupported(.locked) { device.exposureMode = .locked }
                if device.isWhiteBalanceModeSupported(.locked) { device.whiteBalanceMode = .locked }
            } else {
                if device.isExposureModeSupported(.continuousAutoExposure) { device.exposureMode = .continuousAutoExposure }
                if device.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) { device.whiteBalanceMode = .continuousAutoWhiteBalance }
            }
        } catch {
            // Non-fatal: ARKit keeps auto exposure.
        }
    }

    // MARK: - ARSessionDelegate (delegate queue)

    func session(_ session: ARSession, didFailWithError error: any Error) {
        let message: String
        if let ar = error as? ARError, ar.code == .cameraUnauthorized {
            message = "Camera access is off. Enable it in Settings › Privacy & Security › Camera › SplatWork."
        } else {
            message = error.localizedDescription
        }
        onError?(message)
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let now = frame.timestamp
        let camera = frame.camera
        let transform = camera.transform
        updateMotion(transform, now)

        let depthData = frame.smoothedSceneDepth ?? frame.sceneDepth
        let tracking = TrackingSummary(camera.trackingState)

        // Center depth for "too close" guidance, cheap and only every few frames.
        if now - lastStatsTime >= 0.1, let d = depthData {
            centerDepth = Self.centerDepth(d.depthMap)
        }

        if let recorder { captureIfNeeded(frame, depthData: depthData, tracking: tracking, recorder: recorder) }

        if now - lastStatsTime >= 0.1 {
            lastStatsTime = now
            publishStats(frame, tracking: tracking, hasDepth: depthData != nil)
        }
    }

    // MARK: - Keyframe logic

    private func captureIfNeeded(_ frame: ARFrame, depthData: ARDepthData?, tracking: TrackingSummary,
                                 recorder: CaptureRecorder) {
        let now = frame.timestamp
        let camera = frame.camera
        let transform = camera.transform

        if recorder.submittedCount >= settings.maxFrames {
            sendAutoStop("Reached the \(settings.maxFrames)-frame limit set in Settings.")
            return
        }
        if now - lastSpaceCheck > 3 {
            lastSpaceCheck = now
            if CaptureStore.freeSpace < 300_000_000 {
                sendAutoStop("The iPhone is almost out of storage.")
                return
            }
        }
        guard !highResInFlight else { return }
        if settings.requireNormalTracking && tracking != .normal { return }
        guard selector.wantsKeyframe(transform: transform, time: now) else { return }
        guard recorder.canAccept else {
            lastBacklog = now
            return
        }

        // Quality gate.
        let exposure = camera.exposureDuration
        let fx = camera.intrinsics[0][0]
        let distance = max(0.3, centerDepth ?? 1.5)
        let blurPx = Float(exposure) * fx * (angularSpeed + linearSpeed / distance)
        let lv = PixelBufferTools.laplacianVariance(frame.capturedImage)
        let overdue = selector.overdue(transform: transform, time: now)
        let verdict = gate.evaluate(laplacianVariance: lv, motionBlurPixels: blurPx, settings: settings, overdue: overdue)
        guard verdict.accept else {
            rejectedBlurry += 1
            lastBlurReject = now
            return
        }

        let depth: DepthSample? = (depthEnabled ? depthData.flatMap(DepthSample.init) : nil)
        let iso = Self.iso(from: frame)
        let res = camera.imageResolution
        let nativeSize = SIMD2<Float>(Float(res.width), Float(res.height))
        let nativeIntrinsics = Self.intrinsics(camera, width: res.width, height: res.height)
        let trackingString = tracking.manifestValue

        if settings.highResolutionStills {
            highResInFlight = true
            selector.accept(transform: transform, time: now)
            session.captureHighResolutionFrame { [weak self] hiFrame, _ in
                guard let engine = self else { return }
                let box = UncheckedSendableBox(hiFrame)
                engine.delegateQueue.async { [engine] in
                    let `self` = engine
                    self.highResInFlight = false
                    guard let recorder = self.recorder else { return }
                    if let hf = box.value, let copy = PixelBufferTools.copy(hf.capturedImage) {
                        let hiDepth = (self.depthEnabled ? (hf.smoothedSceneDepth ?? hf.sceneDepth).flatMap(DepthSample.init) : nil)
                        let hc = hf.camera
                        let w = CVPixelBufferGetWidth(copy), h = CVPixelBufferGetHeight(copy)
                        let hiIntr = Self.intrinsics(hc, width: CGFloat(w), height: CGFloat(h))
                        let job = KeyframeJob(
                            timestamp: hf.timestamp, pixelBuffer: copy, intrinsics: hiIntr,
                            cameraToWorld: hc.transform, trackingState: TrackingSummary(hc.trackingState).manifestValue,
                            exposureDuration: hc.exposureDuration, iso: Self.iso(from: hf) ?? iso,
                            sharpness: verdict.score,
                            depth: hiDepth ?? depth,
                            depthCameraToWorld: hiDepth != nil ? hc.transform : transform,
                            depthIntrinsics: hiDepth != nil ? hiIntr : nativeIntrinsics,
                            depthImageSize: hiDepth != nil ? SIMD2(Float(w), Float(h)) : nativeSize)
                        self.selector.accept(transform: hc.transform, time: self.selector.lastTime)
                        recorder.submit(job)
                        self.emitKeyframe(transform: hc.transform)
                    }
                }
            }
            return
        }

        guard let copy = PixelBufferTools.copy(frame.capturedImage) else { return }
        let job = KeyframeJob(
            timestamp: now, pixelBuffer: copy, intrinsics: nativeIntrinsics,
            cameraToWorld: transform, trackingState: trackingString,
            exposureDuration: exposure, iso: iso, sharpness: verdict.score,
            depth: depth, depthCameraToWorld: transform,
            depthIntrinsics: nativeIntrinsics, depthImageSize: nativeSize)
        selector.accept(transform: transform, time: now)
        recorder.submit(job)
        emitKeyframe(transform: transform)
    }

    private func emitKeyframe(transform: simd_float4x4) {
        let p = transform.position
        let look = p + transform.forward * (centerDepth ?? 1.5)
        onKeyframe?(KeyframeEvent(position: p, lookAt: look))
    }

    private func sendAutoStop(_ reason: String) {
        guard !autoStopSent else { return }
        autoStopSent = true
        onAutoStop?(reason)
    }

    private func updateMotion(_ t: simd_float4x4, _ time: TimeInterval) {
        defer { prevTransform = t; prevTime = time }
        guard let prev = prevTransform else { return }
        let dt = Float(time - prevTime)
        guard dt > 0.001, dt < 0.5 else { return }
        let ang = CaptureMath.angle(prev.orientation, t.orientation) / dt
        let lin = simd_distance(prev.position, t.position) / dt
        // Light smoothing; ARKit poses jitter slightly frame to frame.
        angularSpeed = angularSpeed * 0.6 + ang * 0.4
        linearSpeed = linearSpeed * 0.6 + lin * 0.4
    }

    private func publishStats(_ frame: ARFrame, tracking: TrackingSummary, hasDepth: Bool) {
        let now = frame.timestamp
        var g: Guidance = .none
        let dark = (frame.lightEstimate?.ambientIntensity ?? 1000) < 200
        switch frame.camera.trackingState {
        case .notAvailable:
            g = .trackingLost
        case .limited(let reason):
            switch reason {
            case .initializing: g = .initializing
            case .excessiveMotion: g = .tooFast
            case .insufficientFeatures: g = dark ? .tooDark : .lowFeatures
            case .relocalizing: g = .relocalizing
            @unknown default: g = .lowFeatures
            }
        case .normal:
            if angularSpeed > 1.4 || linearSpeed > 0.9 {
                g = .tooFast
            } else if recorder != nil && now - lastBlurReject < 0.8 {
                g = .blurry
            } else if dark {
                g = .tooDark
            } else if let d = centerDepth, d < 0.25 {
                g = .tooClose
            } else if recorder != nil && now - lastBacklog < 0.8 {
                g = .backlog
            }
        }
        let t = frame.camera.transform
        let stats = LiveStats(
            tracking: tracking, guidance: g,
            keyframes: recorder?.framesSaved ?? 0,
            bytes: recorder?.bytesWritten ?? 0,
            rejectedBlurry: rejectedBlurry,
            cameraPosition: t.position, cameraForward: t.forward,
            hasDepth: hasDepth, centerDepth: centerDepth)
        onStats?(stats)
    }

    // MARK: - Helpers

    static func intrinsics(_ camera: ARCamera, width: CGFloat, height: CGFloat) -> SIMD4<Float> {
        let k = camera.intrinsics
        let res = camera.imageResolution
        let sx = Float(width / res.width), sy = Float(height / res.height)
        return SIMD4(k[0][0] * sx, k[1][1] * sy, k[2][0] * sx, k[2][1] * sy)
    }

    static func iso(from frame: ARFrame) -> Double? {
        let exif = frame.exifData
        if let arr = exif[kCGImagePropertyExifISOSpeedRatings as String] as? [NSNumber], let v = arr.first {
            return v.doubleValue
        }
        if let arr = exif[kCGImagePropertyExifISOSpeedRatings as String] as? [Double], let v = arr.first {
            return v
        }
        return nil
    }

    private static func centerDepth(_ map: CVPixelBuffer) -> Float? {
        CVPixelBufferLockBaseAddress(map, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(map, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(map) else { return nil }
        let w = CVPixelBufferGetWidth(map), h = CVPixelBufferGetHeight(map)
        let bpr = CVPixelBufferGetBytesPerRow(map)
        var vals: [Float] = []
        vals.reserveCapacity(100)
        for y in stride(from: h / 2 - 10, to: h / 2 + 10, by: 2) {
            let row = base.advanced(by: y * bpr).assumingMemoryBound(to: Float32.self)
            for x in stride(from: w / 2 - 10, to: w / 2 + 10, by: 2) {
                let v = row[x]
                if v.isFinite && v > 0 { vals.append(v) }
            }
        }
        guard !vals.isEmpty else { return nil }
        vals.sort()
        return vals[vals.count / 2]
    }
}
