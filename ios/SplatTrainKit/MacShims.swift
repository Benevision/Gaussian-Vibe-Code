import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

// Stand-ins for the few Mac-app types the shared trainer sources reference that live in files the iPhone
// build doesn't compile (Pipeline/PipelineCommon.swift drags in CoreImage, Process, …).

/// Same cases as `PipelineError` in mac/Sources/SplatStudio/Pipeline/PipelineCommon.swift.
enum PipelineError: LocalizedError {
    case noInput(String)
    case unsupported(String)
    case decodeFailed(String)
    case encodeFailed(String)
    case poseSolveFailed(String)
    case processFailed(String)
    case invalidData(String)

    var errorDescription: String? {
        switch self {
        case .noInput(let s): return "No usable input: \(s)"
        case .unsupported(let s): return "Unsupported: \(s)"
        case .decodeFailed(let s): return "Could not decode \(s)"
        case .encodeFailed(let s): return "Could not write \(s)"
        case .poseSolveFailed(let s): return "Camera pose estimation failed: \(s)"
        case .processFailed(let s): return "Process failed: \(s)"
        case .invalidData(let s): return "Invalid data: \(s)"
        }
    }
}

/// The one `SystemBenchmark` helper (mac/Sources/SplatStudio/Remote/SystemBenchmark.swift) the trainer uses:
/// `MetalTrainer.evaluate(cloud:saveRendersTo:)` writing held-out renders.
enum SystemBenchmark {
    static func writeJPEG(_ rgba: [UInt8], width: Int, height: Int, to url: URL) throws {
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        guard let provider = CGDataProvider(data: Data(rgba) as CFData),
              let img = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4, space: cs,
                                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue), provider: provider,
                                decode: nil, shouldInterpolate: false, intent: .defaultIntent),
              let dst = CGImageDestinationCreateWithURL(url as CFURL, UTType.jpeg.identifier as CFString, 1, nil) else {
            throw PipelineError.encodeFailed(url.lastPathComponent)
        }
        CGImageDestinationAddImage(dst, img, [kCGImageDestinationLossyCompressionQuality: 0.92] as CFDictionary)
        guard CGImageDestinationFinalize(dst) else { throw PipelineError.encodeFailed(url.lastPathComponent) }
    }
}
