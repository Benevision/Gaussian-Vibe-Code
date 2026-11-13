import Foundation
import ImageIO

/// Where a photo was taken, as stored in its EXIF GPS block, plus DJI's height above the take-off point
/// (`drone-dji:RelativeAltitude` in XMP, far steadier than GPS altitude).
///
/// Imported photos are re-encoded (downscaled, orientation applied), which drops all metadata. When the project
/// uses photo locations, only this is copied into the new file — nothing else (no camera serials, no dates) — so
/// camera solving (splatprep, on this Mac or a worker) can check the solve against GPS and put the scene in metres.
struct PhotoLocation: Sendable {
    var gps: [String: Any]
    var relativeAltitude: Double?

    static let djiNamespace = "http://www.dji.com/drone-dji/1.0/" as CFString

    /// Nil when the photo has no latitude / longitude.
    static func read(_ url: URL) -> PhotoLocation? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
              let gps = props[kCGImagePropertyGPSDictionary] as? [CFString: Any],
              gps[kCGImagePropertyGPSLatitude] != nil, gps[kCGImagePropertyGPSLongitude] != nil else { return nil }
        var rel: Double? = nil
        if let meta = CGImageSourceCopyMetadataAtIndex(src, 0, nil),
           let tag = CGImageMetadataCopyTagWithPath(meta, nil, "drone-dji:RelativeAltitude" as CFString),
           let v = CGImageMetadataTagCopyValue(tag) {
            rel = Double("\(v)".trimmingCharacters(in: .whitespaces))
        }
        return PhotoLocation(gps: Dictionary(uniqueKeysWithValues: gps.map { ($0.key as String, $0.value) }), relativeAltitude: rel)
    }

    /// Metadata holding just the location, for `CGImageDestinationAddImageAndMetadata`.
    func metadata() -> CGImageMetadata {
        let m = CGImageMetadataCreateMutable()
        for (k, v) in gps {
            CGImageMetadataSetValueMatchingImageProperty(m, kCGImagePropertyGPSDictionary, k as CFString, v as CFTypeRef)
        }
        if let rel = relativeAltitude {
            CGImageMetadataRegisterNamespaceForPrefix(m, Self.djiNamespace, "drone-dji" as CFString, nil)
            CGImageMetadataSetValueWithPath(m, nil, "drone-dji:RelativeAltitude" as CFString, String(format: "%+.2f", rel) as CFString)
        }
        return m
    }
}

/// What a set of photos (or a video) says about how it was shot. A drone is recognised by the drone's own
/// markers — the camera maker, DJI's flight / gimbal tags, DJI file names or flight-log sidecars — never by GPS
/// alone: phone photos carry locations too, and they are not aerial captures.
struct CaptureSurvey: Sendable, Equatable {
    var sampled = 0
    var withLocation = 0
    /// "DJI FC3582" and the like, when the photos come from a drone.
    var drone: String? = nil

    var isDrone: Bool { drone != nil }
    /// Share of sampled photos with a location.
    var locationFraction: Double { sampled > 0 ? Double(withLocation) / Double(sampled) : 0 }

    static let droneMakers = ["dji", "autel", "parrot", "skydio", "yuneec", "hasselblad"]   // Hasselblad: DJI Mavic cameras

    /// Looks at up to `limit` photos spread over the list.
    static func photos(_ urls: [URL], limit: Int = 24) -> CaptureSurvey {
        var s = CaptureSurvey()
        guard !urls.isEmpty else { return s }
        let step = max(1, urls.count / limit)
        var droneVotes = 0, droneName: String? = nil
        for u in stride(from: 0, to: urls.count, by: step).prefix(limit).map({ urls[$0] }) {
            guard let src = CGImageSourceCreateWithURL(u as CFURL, nil),
                  let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any] else { continue }
            s.sampled += 1
            if let g = props[kCGImagePropertyGPSDictionary] as? [CFString: Any], g[kCGImagePropertyGPSLatitude] != nil { s.withLocation += 1 }
            let tiff = props[kCGImagePropertyTIFFDictionary] as? [CFString: Any]
            let make = (tiff?[kCGImagePropertyTIFFMake] as? String ?? "").trimmingCharacters(in: .whitespaces)
            let model = (tiff?[kCGImagePropertyTIFFModel] as? String ?? "").trimmingCharacters(in: .whitespaces)
            // Hasselblad also makes hand-held cameras; only its drone cameras (L1D / L2D, in DJI Mavics) count
            var aerial = droneMakers.contains { make.lowercased().hasPrefix($0) }
                && !(make.lowercased().hasPrefix("hasselblad") && !model.uppercased().hasPrefix("L"))
            if !aerial, let meta = CGImageSourceCopyMetadataAtIndex(src, 0, nil) {
                // DJI writes its flight and gimbal state into XMP (drone-dji:FlightYawDegree, GimbalPitchDegree…)
                aerial = ["drone-dji:GimbalPitchDegree", "drone-dji:FlightYawDegree", "drone-dji:RelativeAltitude"].contains {
                    CGImageMetadataCopyTagWithPath(meta, nil, $0 as CFString) != nil
                }
            }
            if aerial { droneVotes += 1; droneName = droneName ?? [make, model].filter { !$0.isEmpty }.joined(separator: " ") }
        }
        if s.sampled > 0, droneVotes * 2 >= s.sampled { s.drone = droneName?.isEmpty == false ? droneName : "drone" }
        return s
    }

    /// A drone video: DJI's file naming (`DJI_0001.MP4`) or the flight log it writes next to it (`.SRT`).
    static func video(_ url: URL) -> CaptureSurvey {
        var s = CaptureSurvey()
        let name = url.deletingPathExtension().lastPathComponent
        let srt = ["SRT", "srt"].contains { FileManager.default.fileExists(atPath: url.deletingPathExtension().appendingPathExtension($0).path) }
        if name.uppercased().hasPrefix("DJI_") || srt { s.drone = "DJI" }
        return s
    }
}
