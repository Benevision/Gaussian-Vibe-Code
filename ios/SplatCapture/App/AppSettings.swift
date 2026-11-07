import Foundation
import Observation

/// User-facing settings, persisted to UserDefaults.
@MainActor
@Observable
final class AppSettings {
    private static let key = "captureSettings.v1"

    var capture: CaptureSettings {
        didSet { persist() }
    }

    /// Last pairing code typed for each Mac (keyed by Mac name).
    var pairingCodes: [String: String] {
        didSet { UserDefaults.standard.set(pairingCodes, forKey: "pairingCodes") }
    }

    let videoFormats: [VideoFormatOption]

    init() {
        if let data = UserDefaults.standard.data(forKey: Self.key),
           let decoded = try? JSONDecoder().decode(CaptureSettings.self, from: data) {
            capture = decoded
        } else {
            capture = CaptureSettings()
        }
        pairingCodes = (UserDefaults.standard.dictionary(forKey: "pairingCodes") as? [String: String]) ?? [:]
        videoFormats = VideoFormatOption.available()
        if !DeviceInfo.hasLiDAR {
            capture.recordDepth = false
        }
    }

    func resetToDefaults() {
        capture = CaptureSettings()
        if !DeviceInfo.hasLiDAR { capture.recordDepth = false }
    }

    private func persist() {
        if let data = try? JSONEncoder().encode(capture) {
            UserDefaults.standard.set(data, forKey: Self.key)
        }
    }
}
