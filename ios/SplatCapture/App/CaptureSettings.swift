import Foundation

enum KeyframeMode: String, Codable, CaseIterable, Identifiable, Sendable {
    case motion, timed
    var id: String { rawValue }
    var label: String { self == .motion ? "Automatic (by movement)" : "Timed interval" }
}

enum ImageFileFormat: String, Codable, CaseIterable, Identifiable, Sendable {
    case jpeg, heic
    var id: String { rawValue }
    var label: String { self == .jpeg ? "JPEG" : "HEIC" }
    var fileExtension: String { self == .jpeg ? "jpg" : "heic" }
}

enum ConfidenceLevel: Int, Codable, CaseIterable, Identifiable, Sendable {
    case low = 0, medium = 1, high = 2
    var id: Int { rawValue }
    var label: String {
        switch self {
        case .low: "Low (keep everything)"
        case .medium: "Medium"
        case .high: "High only"
        }
    }
}

enum OverlayStyle: String, Codable, CaseIterable, Identifiable, Sendable {
    case color, highlight
    var id: String { rawValue }
    var label: String { self == .color ? "True colour" : "Highlight" }
}

/// Immutable snapshot of every capture-related setting, handed to the
/// capture engine (which runs off the main actor) when a session starts.
struct CaptureSettings: Codable, Sendable, Equatable {
    var mode: KeyframeMode = .motion
    /// Metres.
    var minTranslation: Float = 0.07
    /// Degrees.
    var minRotationDegrees: Float = 6
    /// Maximum keyframes per second.
    var maxKeyframeRate: Double = 3
    /// Seconds (timed mode).
    var timedInterval: Double = 0.5

    var requireNormalTracking: Bool = true
    var rejectBlurry: Bool = true
    /// Estimated motion blur (pixels at the saved resolution) above which a frame is rejected.
    var maxMotionBlurPixels: Float = 2.5
    /// Relative sharpness (vs. recent frames) below which a frame is rejected.
    var minRelativeSharpness: Float = 0.55

    /// `VideoFormatOption.id`; nil = ARKit default.
    var videoFormatID: String? = nil
    var highResolutionStills: Bool = false
    var imageFormat: ImageFileFormat = .jpeg
    var imageQuality: Double = 0.9

    var recordDepth: Bool = true
    var smoothedDepth: Bool = false
    var saveConfidence: Bool = true

    var buildPointCloud: Bool = true
    /// Centimetres.
    var voxelSizeCm: Float = 1.5
    var minConfidence: ConfidenceLevel = .medium
    /// Metres.
    var maxDepth: Float = 4.0

    var lockExposureWhileRecording: Bool = true
    var autoFocus: Bool = true
    var maxFrames: Int = 1500

    var showOverlay: Bool = true
    var overlayStyle: OverlayStyle = .color
    var overlayPointSize: Float = 3
    var showMiniMap: Bool = true
    var haptics: Bool = true
}

extension CaptureSettings {
    /// True when both settings produce the same ARKit configuration (so tracking need not be reset).
    func sameSessionConfiguration(as o: CaptureSettings) -> Bool {
        videoFormatID == o.videoFormatID && highResolutionStills == o.highResolutionStills
            && recordDepth == o.recordDepth && smoothedDepth == o.smoothedDepth
            && showOverlay == o.showOverlay && autoFocus == o.autoFocus
    }
}
