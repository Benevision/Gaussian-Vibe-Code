import Foundation

/// `manifest.json` of a `.splatcapture` bundle — see docs/PROTOCOL.md §1.
struct CaptureManifest: Codable, Sendable {
    var format: String = "splatcapture"
    var version: Int = 1
    var source: String? = "iphone"
    var name: String
    var createdAt: Date
    var device: String
    var appVersion: String
    var hasLiDAR: Bool
    var coordinateSystem: String = "arkit"
    var imageWidth: Int
    var imageHeight: Int
    var depthWidth: Int?
    var depthHeight: Int?
    var frames: [ManifestFrame]
}

struct ManifestFrame: Codable, Sendable {
    var index: Int
    var timestamp: Double
    var image: String
    var depth: String?
    var confidence: String?
    /// fx, fy, cx, cy in saved-image pixels.
    var intrinsics: [Float]
    /// 4x4 column-major (simd_float4x4 memory order), ARKit camera convention.
    var cameraToWorld: [Float]
    var trackingState: String
    var exposureDuration: Double?
    var iso: Double?
    var sharpness: Float?
}

/// Lenient reader used by the library — tolerates partial / older manifests.
struct ManifestSummary: Decodable, Sendable {
    var name: String?
    var createdAt: Date?
    var device: String?
    var hasLiDAR: Bool?
    var imageWidth: Int?
    var imageHeight: Int?
    var frames: [FrameStub]?

    struct FrameStub: Decodable, Sendable {}
}

enum ManifestCoding {
    static func encoder() -> JSONEncoder {
        let e = JSONEncoder()
        e.dateEncodingStrategy = .iso8601
        e.outputFormatting = [.withoutEscapingSlashes]
        return e
    }

    static func decoder() -> JSONDecoder {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601
        return d
    }
}
