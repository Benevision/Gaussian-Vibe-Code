import Foundation
import simd

/// Decides when a frame is far enough from the previous keyframe to be worth saving.
/// Used only on the AR delegate queue.
struct KeyframeSelector {
    var settings: CaptureSettings
    private(set) var lastPose: simd_float4x4?
    private(set) var lastTime: TimeInterval = -.infinity

    init(settings: CaptureSettings) { self.settings = settings }

    mutating func reset() {
        lastPose = nil
        lastTime = -.infinity
    }

    /// True if the motion / interval criterion is met (quality gates are separate).
    func wantsKeyframe(transform: simd_float4x4, time: TimeInterval) -> Bool {
        let minInterval = 1.0 / max(0.1, settings.maxKeyframeRate)
        guard time - lastTime >= minInterval else { return false }
        guard let last = lastPose else { return true }
        switch settings.mode {
        case .timed:
            return time - lastTime >= settings.timedInterval
        case .motion:
            let moved = simd_distance(last.position, transform.position)
            if moved >= settings.minTranslation { return true }
            let turned = CaptureMath.degrees(CaptureMath.angle(last.orientation, transform.orientation))
            return turned >= settings.minRotationDegrees
        }
    }

    mutating func accept(transform: simd_float4x4, time: TimeInterval) {
        lastPose = transform
        lastTime = time
    }
}

/// Rejects blurry frames using two cheap signals:
///  * predicted motion blur = (angular speed × exposure × focal length) + translational term
///  * Laplacian variance relative to the recent sharp frames (scene-independent)
struct SharpnessGate {
    struct Verdict {
        var accept: Bool
        var relativeSharpness: Float
        var motionBlurPixels: Float
        /// Combined 0..1 score stored in the manifest.
        var score: Float
        var reason: String?
    }

    private var history: [Float] = []
    private let historySize = 40

    mutating func reset() { history.removeAll() }

    mutating func evaluate(laplacianVariance lv: Float, motionBlurPixels blur: Float,
                           settings: CaptureSettings) -> Verdict {
        let reference: Float
        if history.count >= 6 {
            let sorted = history.sorted()
            reference = max(1, sorted[Int(Float(sorted.count - 1) * 0.8)])
        } else {
            reference = max(1, history.max() ?? lv)
        }
        let rel = min(1, lv / reference)
        let motionScore = 1 / (1 + (blur / 3) * (blur / 3))
        let score = min(rel, motionScore)

        history.append(lv)
        if history.count > historySize { history.removeFirst(history.count - historySize) }

        guard settings.rejectBlurry else {
            return Verdict(accept: true, relativeSharpness: rel, motionBlurPixels: blur, score: score, reason: nil)
        }
        if blur > settings.maxMotionBlurPixels {
            return Verdict(accept: false, relativeSharpness: rel, motionBlurPixels: blur, score: score, reason: "motion")
        }
        // Only trust the relative test once we have some history.
        if history.count >= 6 && rel < settings.minRelativeSharpness {
            return Verdict(accept: false, relativeSharpness: rel, motionBlurPixels: blur, score: score, reason: "blur")
        }
        return Verdict(accept: true, relativeSharpness: rel, motionBlurPixels: blur, score: score, reason: nil)
    }
}
