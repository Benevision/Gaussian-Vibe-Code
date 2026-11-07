import Foundation
import Metal
import AVFoundation
import CoreImage
import ImageIO
import UniformTypeIdentifiers
import simd

enum ImageWriter {
    static func writePNG(texture: MTLTexture, to url: URL) throws {
        let w = texture.width, h = texture.height
        var bytes = [UInt8](repeating: 0, count: w * h * 4)
        texture.getBytes(&bytes, bytesPerRow: w * 4, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        guard let ctx = CGContext(data: &bytes, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: cs,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue),
              let img = ctx.makeImage(),
              let dst = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw FormatError.invalid("PNG encoding")
        }
        CGImageDestinationAddImage(dst, img, nil)
        guard CGImageDestinationFinalize(dst) else { throw FormatError.invalid("PNG write") }
    }
}

/// Renders an orbit (turntable) or a keyframed path to an H.264/HEVC movie.
enum VideoExporter {
    struct Options {
        var width = 1920, height = 1080, fps = 30
        var seconds: Double = 8
        var hevc = true
        var pitch: Float? = nil      // nil = keep current
        var revolutions: Float = 1
    }

    static func renderTurntable(renderer: SplatRenderer, start: ViewerCamera, settings: ViewerSettings, options o: Options,
                                to url: URL, progress: @escaping (Double) -> Void) throws {
        try? FileManager.default.removeItem(at: url)
        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: o.hevc ? AVVideoCodecType.hevc : AVVideoCodecType.h264,
            AVVideoWidthKey: o.width, AVVideoHeightKey: o.height,
            AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: o.width * o.height * 8],
        ])
        input.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: o.width, kCVPixelBufferHeightKey as String: o.height,
        ])
        writer.add(input)
        guard writer.startWriting() else { throw writer.error ?? FormatError.invalid("video writer") }
        writer.startSession(atSourceTime: .zero)
        let frames = max(1, Int(o.seconds * Double(o.fps)))
        for f in 0..<frames {
            var cam = start
            cam.yaw = start.yaw + Float(f) / Float(frames) * 2 * .pi * o.revolutions
            if let p = o.pitch { cam.pitch = p }
            guard let tex = renderer.renderOffscreen(width: o.width, height: o.height, camera: cam, settings: settings) else { break }
            while !input.isReadyForMoreMediaData { Thread.sleep(forTimeInterval: 0.002) }
            guard let pool = adaptor.pixelBufferPool else { break }
            var pb: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pb)
            guard let pb else { break }
            CVPixelBufferLockBaseAddress(pb, [])
            tex.getBytes(CVPixelBufferGetBaseAddress(pb)!, bytesPerRow: CVPixelBufferGetBytesPerRow(pb), from: MTLRegionMake2D(0, 0, o.width, o.height), mipmapLevel: 0)
            CVPixelBufferUnlockBaseAddress(pb, [])
            adaptor.append(pb, withPresentationTime: CMTime(value: CMTimeValue(f), timescale: CMTimeScale(o.fps)))
            progress(Double(f + 1) / Double(frames))
        }
        input.markAsFinished()
        let sem = DispatchSemaphore(value: 0)
        writer.finishWriting { sem.signal() }
        sem.wait()
        if let e = writer.error { throw e }
    }
}
