import ARKit
import Foundation
import UIKit

enum DeviceInfo {
    static var hasLiDAR: Bool {
        ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
    }

    static var supportsSmoothedDepth: Bool {
        ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth)
    }

    static var isARSupported: Bool { ARWorldTrackingConfiguration.isSupported }

    /// Hardware identifier, e.g. "iPhone17,1".
    static var modelIdentifier: String {
        var info = utsname()
        uname(&info)
        let mirror = Mirror(reflecting: info.machine)
        let id = mirror.children.reduce(into: "") { result, element in
            if let v = element.value as? Int8, v != 0 { result.append(Character(UnicodeScalar(UInt8(v)))) }
        }
        return id
    }

    /// Human readable model name for the manifest ("iPhone 16 Pro").
    static var modelName: String {
        let id = modelIdentifier
        let names: [String: String] = [
            "iPhone12,1": "iPhone 11", "iPhone12,3": "iPhone 11 Pro", "iPhone12,5": "iPhone 11 Pro Max",
            "iPhone13,1": "iPhone 12 mini", "iPhone13,2": "iPhone 12",
            "iPhone13,3": "iPhone 12 Pro", "iPhone13,4": "iPhone 12 Pro Max",
            "iPhone14,4": "iPhone 13 mini", "iPhone14,5": "iPhone 13",
            "iPhone14,2": "iPhone 13 Pro", "iPhone14,3": "iPhone 13 Pro Max",
            "iPhone14,7": "iPhone 14", "iPhone14,8": "iPhone 14 Plus",
            "iPhone15,2": "iPhone 14 Pro", "iPhone15,3": "iPhone 14 Pro Max",
            "iPhone15,4": "iPhone 15", "iPhone15,5": "iPhone 15 Plus",
            "iPhone16,1": "iPhone 15 Pro", "iPhone16,2": "iPhone 15 Pro Max",
            "iPhone17,3": "iPhone 16", "iPhone17,4": "iPhone 16 Plus",
            "iPhone17,1": "iPhone 16 Pro", "iPhone17,2": "iPhone 16 Pro Max",
            "iPhone17,5": "iPhone 16e",
        ]
        if let n = names[id] { return n }
        if id.isEmpty || id == "arm64" || id == "x86_64" { return "iPhone" }
        return id
    }

    static var appVersion: String {
        let info = Bundle.main.infoDictionary
        let v = info?["CFBundleShortVersionString"] as? String ?? "1.0"
        let b = info?["CFBundleVersion"] as? String ?? "1"
        return "\(v) (\(b))"
    }
}

/// A selectable ARKit video format (Sendable description of `ARConfiguration.VideoFormat`).
struct VideoFormatOption: Identifiable, Hashable, Sendable {
    let id: String
    let width: Int
    let height: Int
    let fps: Int
    let isHighResCapable: Bool
    let is4K: Bool
    let isDefault: Bool

    var label: String {
        var s = "\(width)×\(height) @ \(fps) fps"
        if is4K { s += " (4K)" }
        if isDefault { s += " — default" }
        return s
    }

    static func identifier(for f: ARConfiguration.VideoFormat) -> String {
        "\(Int(f.imageResolution.width))x\(Int(f.imageResolution.height))@\(f.framesPerSecond)-\(f.captureDeviceType.rawValue)"
    }

    static func backFormats() -> [ARConfiguration.VideoFormat] {
        ARWorldTrackingConfiguration.supportedVideoFormats.filter { $0.captureDevicePosition == .back }
    }

    static func available() -> [VideoFormatOption] {
        let formats = backFormats()
        let fourK = ARWorldTrackingConfiguration.recommendedVideoFormatFor4KResolution.map(identifier(for:))
        let defaultID = formats.first.map(identifier(for:))
        var seen = Set<String>()
        var out: [VideoFormatOption] = []
        for f in formats {
            let id = identifier(for: f)
            guard seen.insert(id).inserted else { continue }
            let w = Int(f.imageResolution.width), h = Int(f.imageResolution.height)
            out.append(VideoFormatOption(
                id: id, width: w, height: h, fps: f.framesPerSecond,
                isHighResCapable: f.isRecommendedForHighResolutionFrameCapturing,
                is4K: id == fourK || w >= 3840,
                isDefault: id == defaultID))
        }
        return out
    }

    static func format(for id: String?) -> ARConfiguration.VideoFormat? {
        guard let id else { return nil }
        return backFormats().first { identifier(for: $0) == id }
    }
}
