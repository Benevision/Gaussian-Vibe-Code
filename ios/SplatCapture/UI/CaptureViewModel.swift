import Foundation
import Observation
import simd
import UIKit

@MainActor
@Observable
final class CaptureViewModel {
    enum Phase: Equatable {
        case idle
        case recording
        case finishing
    }

    let engine = CaptureEngine()

    private(set) var phase: Phase = .idle
    private(set) var stats = LiveStats()
    private(set) var keyframeCount = 0
    private(set) var keyframePositions: [SIMD3<Float>] = []
    private(set) var lookAtCentroid: SIMD3<Float>?
    /// 24 sectors around the subject; true = a keyframe looked at the subject from that direction.
    private(set) var orbitSectors = [Bool](repeating: false, count: 24)
    private(set) var recordingStart: Date?
    private(set) var lastSummary: CaptureSummary?
    var errorMessage: String?
    var autoStopMessage: String?
    var showFinished = false

    @ObservationIgnored weak var renderer: PointCloudRenderer?
    @ObservationIgnored private var lookAtSum = SIMD3<Float>.zero
    @ObservationIgnored private var lookAtCount = 0
    @ObservationIgnored private var runningSettings: CaptureSettings?
    @ObservationIgnored private var isSessionRunning = false

    init() {
        engine.onStats = { [weak self] s in
            Task { @MainActor in self?.stats = s }
        }
        engine.onKeyframe = { [weak self] e in
            Task { @MainActor in self?.handleKeyframe(e) }
        }
        engine.onPoints = { [weak self] snapshot in
            Task { @MainActor in self?.renderer?.update(snapshot) }
        }
        engine.onAutoStop = { [weak self] reason in
            Task { @MainActor in
                guard let self, self.phase == .recording else { return }
                self.autoStopMessage = "Recording stopped: \(reason)"
                self.stopRecording()
            }
        }
        engine.onError = { [weak self] message in
            Task { @MainActor in self?.errorMessage = message }
        }
    }

    var isRecording: Bool { phase == .recording }

    // MARK: Session

    func startSession(settings: CaptureSettings) {
        guard DeviceInfo.isARSupported else {
            errorMessage = "This device doesn't support ARKit world tracking."
            return
        }
        let needsReset = runningSettings.map { !$0.sameSessionConfiguration(as: settings) } ?? true
        // Only re-run when a session-relevant setting changed or we were paused.
        if !isSessionRunning || needsReset {
            engine.run(settings: settings, reset: needsReset)
            runningSettings = settings
            isSessionRunning = true
        }
        renderer?.apply(settings: settings)
        UIApplication.shared.isIdleTimerDisabled = true
    }

    func pauseSession() {
        guard phase == .idle else { return }
        engine.pause()
        isSessionRunning = false
        UIApplication.shared.isIdleTimerDisabled = false
    }

    // MARK: Recording

    func startRecording(settings: CaptureSettings) {
        guard phase == .idle else { return }
        if CaptureStore.freeSpace < 500_000_000 {
            errorMessage = "Less than 500 MB of storage is free. Delete old captures or free up space first."
            return
        }
        // Make sure the session runs the current settings before recording.
        startSession(settings: settings)
        do {
            _ = try engine.startRecording(settings: settings)
        } catch {
            errorMessage = error.localizedDescription
            return
        }
        keyframeCount = 0
        keyframePositions = []
        orbitSectors = [Bool](repeating: false, count: orbitSectors.count)
        lookAtSum = .zero
        lookAtCount = 0
        lookAtCentroid = nil
        recordingStart = Date()
        autoStopMessage = nil
        renderer?.clear()
        phase = .recording
        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
    }

    func stopRecording() {
        guard phase == .recording else { return }
        phase = .finishing
        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
        engine.stopRecording { [weak self] result in
            Task { @MainActor in self?.finished(result) }
        }
    }

    func discardLastCapture() {
        if let s = lastSummary {
            try? FileManager.default.removeItem(at: s.url)
        }
        lastSummary = nil
        showFinished = false
    }

    private func finished(_ result: Result<CaptureSummary, any Error>) {
        phase = .idle
        recordingStart = nil
        switch result {
        case .success(let s):
            lastSummary = s
            showFinished = true
            UINotificationFeedbackGenerator().notificationOccurred(.success)
        case .failure(let e):
            errorMessage = e.localizedDescription
            renderer?.clear()
        }
    }

    private func handleKeyframe(_ e: KeyframeEvent) {
        guard phase == .recording else { return }
        keyframeCount += 1
        keyframePositions.append(e.position)
        lookAtSum += e.lookAt
        lookAtCount += 1
        let center = lookAtSum / Float(lookAtCount)
        lookAtCentroid = center
        // Recompute the orbit coverage around the (moving) centroid.
        var sectors = [Bool](repeating: false, count: orbitSectors.count)
        for p in keyframePositions {
            let d = p - center
            guard d.x * d.x + d.z * d.z > 0.01 else { continue }
            let a = atan2(d.z, d.x)
            let idx = Int(((a + .pi) / (2 * .pi)) * Float(sectors.count)) % sectors.count
            sectors[idx] = true
        }
        orbitSectors = sectors
    }
}
