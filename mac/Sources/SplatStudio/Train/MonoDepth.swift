import Foundation
import CoreML
import CoreGraphics
import CoreVideo
import ImageIO
import simd
import CryptoKit

/// Predicted depth for photos without a depth sensor (no LiDAR / stereo), for the flat-surfaces (depth-normal) loss.
///
/// Depth Anything V2 Small in Apple's Core ML conversion (apple/coreml-depth-anything-v2-small, Apache-2.0; 48 MB,
/// downloaded once into Application Support) predicts relative inverse depth. Each prediction is fitted to the
/// dataset's own 3D points (`1/z ≈ a·pred + b`, iteratively reweighted so points hidden behind a surface don't
/// drag it) and kept at about LiDAR resolution (256 px long side). Same method as the Linux worker
/// (linux/splatworker/depthprior.py); results are cached in the dataset's `depth-prior-mac/` folder.
enum MonoDepth {
    static let repo = "apple/coreml-depth-anything-v2-small"
    static let package = "DepthAnythingV2SmallF16.mlpackage"
    static let files = ["Manifest.json", "Data/com.apple.CoreML/model.mlmodel", "Data/com.apple.CoreML/weights/weight.bin"]
    static let longSide = 256
    static let cacheVersion = 1

    struct Map { var values: [Float]; var width: Int; var height: Int }

    static var modelsDir: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("SplatStudio/Models", isDirectory: true)
    }
    static var compiledURL: URL { modelsDir.appendingPathComponent("DepthAnythingV2SmallF16.mlmodelc", isDirectory: true) }

    /// The compiled model, downloading and compiling it on first use (blocking; call off the main thread).
    static func loadModel(log: (String) -> Void) throws -> MLModel {
        let fm = FileManager.default
        if !fm.fileExists(atPath: compiledURL.path) {
            log("downloading the depth model (Depth Anything V2 Small, 48 MB, once)")
            let tmp = fm.temporaryDirectory.appendingPathComponent("splatwork-depth-\(UUID().uuidString)", isDirectory: true)
            defer { try? fm.removeItem(at: tmp) }
            let pkg = tmp.appendingPathComponent(package, isDirectory: true)
            for f in files {
                let dst = pkg.appendingPathComponent(f)
                try fm.createDirectory(at: dst.deletingLastPathComponent(), withIntermediateDirectories: true)
                let src = URL(string: "https://huggingface.co/\(repo)/resolve/main/\(package)/\(f)")!
                try download(src, to: dst)
            }
            let compiled = try MLModel.compileModel(at: pkg)
            try fm.createDirectory(at: modelsDir, withIntermediateDirectories: true)
            try? fm.removeItem(at: compiledURL)
            try fm.moveItem(at: compiled, to: compiledURL)
        }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all
        return try MLModel(contentsOf: compiledURL, configuration: cfg)
    }

    private static func download(_ url: URL, to dst: URL) throws {
        let sem = DispatchSemaphore(value: 0)
        var failure: Error?
        let task = URLSession.shared.downloadTask(with: url) { tmp, resp, err in
            defer { sem.signal() }
            if let err { failure = err; return }
            guard let tmp, (resp as? HTTPURLResponse)?.statusCode == 200 else {
                failure = PipelineError.processFailed("depth model download failed (\((resp as? HTTPURLResponse)?.statusCode ?? 0))"); return
            }
            do { try? FileManager.default.removeItem(at: dst); try FileManager.default.moveItem(at: tmp, to: dst) } catch { failure = error }
        }
        task.resume()
        sem.wait()
        if let failure { throw failure }
    }

    /// Relative inverse depth (bigger = nearer) at the model's 518×392 resolution, for any photo (stretched to it).
    static func predict(_ model: MLModel, image: CGImage) throws -> Map {
        guard let c = model.modelDescription.inputDescriptionsByName.values.first?.imageConstraint,
              let inName = model.modelDescription.inputDescriptionsByName.keys.first else {
            throw PipelineError.processFailed("depth model has no image input")
        }
        let w = c.pixelsWide, h = c.pixelsHigh
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &pb)
        guard let pb else { throw PipelineError.processFailed("pixel buffer") }
        CVPixelBufferLockBaseAddress(pb, [])
        if let ctx = CGContext(data: CVPixelBufferGetBaseAddress(pb), width: w, height: h, bitsPerComponent: 8,
                               bytesPerRow: CVPixelBufferGetBytesPerRow(pb), space: CGColorSpaceCreateDeviceRGB(),
                               bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue) {
            ctx.interpolationQuality = .high
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        }
        CVPixelBufferUnlockBaseAddress(pb, [])
        let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: [inName: MLFeatureValue(pixelBuffer: pb)]))
        guard let name = model.modelDescription.outputDescriptionsByName.keys.first, let v = out.featureValue(for: name) else {
            throw PipelineError.processFailed("depth model gave no output")
        }
        if let ob = v.imageBufferValue {
            CVPixelBufferLockBaseAddress(ob, .readOnly); defer { CVPixelBufferUnlockBaseAddress(ob, .readOnly) }
            let ow = CVPixelBufferGetWidth(ob), oh = CVPixelBufferGetHeight(ob), bpr = CVPixelBufferGetBytesPerRow(ob)
            let base = CVPixelBufferGetBaseAddress(ob)!
            var vals = [Float](repeating: 0, count: ow * oh)
            let fmt = CVPixelBufferGetPixelFormatType(ob)
            for y in 0..<oh {
                let row = base.advanced(by: y * bpr)
                for x in 0..<ow {
                    vals[y * ow + x] = fmt == kCVPixelFormatType_OneComponent16Half
                        ? Float(row.load(fromByteOffset: x * 2, as: Float16.self))
                        : row.load(fromByteOffset: x * 4, as: Float.self)
                }
            }
            return Map(values: vals, width: ow, height: oh)
        }
        if let a = v.multiArrayValue {
            let n = a.count, shape = a.shape.map(\.intValue)
            var vals = [Float](repeating: 0, count: n)
            for i in 0..<n { vals[i] = a[i].floatValue }
            return Map(values: vals, width: shape.last ?? n, height: n / max(shape.last ?? n, 1))
        }
        throw PipelineError.processFailed("depth model output isn't an image")
    }

    /// Robust fit of 1/z ≈ a·pred + b. Returns (a, b, median relative depth error) or nil.
    static func fit(pred: [Float], z: [Float], iterations: Int = 8) -> (a: Double, b: Double, error: Double)? {
        var p: [Double] = [], y: [Double] = []
        for (pv, zv) in zip(pred, z) where pv.isFinite && zv.isFinite && zv > 0 { p.append(Double(pv)); y.append(1 / Double(zv)) }
        guard p.count >= 20 else { return nil }
        var w = [Double](repeating: 1, count: p.count), a = 0.0, b = 0.0
        for _ in 0..<iterations {
            // weighted least squares for (a, b)
            var sw = 0.0, sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0
            for i in p.indices { let ww = w[i] * w[i]; sw += ww; sx += ww * p[i]; sy += ww * y[i]; sxx += ww * p[i] * p[i]; sxy += ww * p[i] * y[i] }
            let det = sw * sxx - sx * sx
            guard abs(det) > 1e-18 else { return nil }
            a = (sw * sxy - sx * sy) / det; b = (sxx * sy - sx * sxy) / det
            let r = p.indices.map { abs(a * p[$0] + b - y[$0]) }
            let s = 1.4826 * median(r) + 1e-12
            w = r.map { 1 / max(1, $0 / (2 * s)) }
        }
        guard a > 0 else { return nil }
        var rel: [Double] = []
        for i in p.indices { let inv = a * p[i] + b; if inv > 1e-9 { rel.append(abs(1 / inv - 1 / y[i]) * y[i]) } }
        guard rel.count >= 20 else { return nil }
        return (a, b, median(rel))
    }

    private static func median(_ v: [Double]) -> Double {
        guard !v.isEmpty else { return 0 }
        let s = v.sorted(); return s[s.count / 2]
    }

    /// Bilinear sample of a map at (u, v) in its pixel coordinates.
    static func sample(_ m: Map, _ u: Float, _ v: Float) -> Float {
        let x0 = max(0, min(m.width - 2, Int(u.rounded(.down)))), y0 = max(0, min(m.height - 2, Int(v.rounded(.down))))
        let fx = max(0, min(1, u - Float(x0))), fy = max(0, min(1, v - Float(y0)))
        let i = y0 * m.width + x0
        return m.values[i] * (1 - fx) * (1 - fy) + m.values[i + 1] * fx * (1 - fy)
            + m.values[i + m.width] * (1 - fx) * fy + m.values[i + m.width + 1] * fx * fy
    }

    static func cacheKey(_ c: DatasetCamera) -> String {
        var h = SHA256()
        h.update(data: Data(c.image.utf8))
        var nums: [Float] = [c.fx, c.fy, c.cx, c.cy, Float(c.width), Float(c.height)]
        for col in 0..<4 { for row in 0..<4 { nums.append((c.c2w[col][row] * 1e5).rounded() / 1e5) } }
        nums.withUnsafeBytes { h.update(bufferPointer: $0) }
        return "\(c.id)-" + h.finalize().prefix(6).map { String(format: "%02x", $0) }.joined()
    }

    /// Predicted, scene-fitted depth for `cameras` (cached in the dataset). Photos whose fit is poor are skipped.
    static func priors(dataset: PreparedDataset, cameras: [DatasetCamera], points: [SIMD3<Float>],
                       log: (String) -> Void, progress: (Double) -> Void) throws -> [Int: Map] {
        let fm = FileManager.default
        let dir = dataset.root.appendingPathComponent("depth-prior-mac", isDirectory: true)
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        let metaURL = dir.appendingPathComponent("meta.json")
        let metaOK = (try? JSONSerialization.jsonObject(with: Data(contentsOf: metaURL)) as? [String: Any])?["version"] as? Int == cacheVersion
        if !metaOK {
            for f in (try? fm.contentsOfDirectory(atPath: dir.path)) ?? [] { try? fm.removeItem(at: dir.appendingPathComponent(f)) }
            try? JSONSerialization.data(withJSONObject: ["version": cacheVersion, "model": repo]).write(to: metaURL)
        }
        var out: [Int: Map] = [:], todo: [DatasetCamera] = []
        for c in cameras {
            let f = dir.appendingPathComponent(cacheKey(c) + ".bin")
            if let d = try? Data(contentsOf: f), d.count >= 8 {
                let w = Int(d.withUnsafeBytes { $0.load(fromByteOffset: 0, as: UInt32.self) }), h = Int(d.withUnsafeBytes { $0.load(fromByteOffset: 4, as: UInt32.self) })
                if d.count == 8 + w * h * 4 {
                    var v = [Float](repeating: 0, count: w * h)
                    _ = v.withUnsafeMutableBytes { d.copyBytes(to: $0, from: 8..<d.count) }
                    out[c.id] = Map(values: v, width: w, height: h); continue
                }
            }
            todo.append(c)
        }
        guard !todo.isEmpty else { return out }
        guard points.count >= 100 else { log("predicted depth: the dataset has too few 3D points to scale it; skipped"); return out }
        let model = try loadModel(log: log)
        var pts = points
        if pts.count > 200_000 { var rng = SystemRandomNumberGenerator(); pts = (0..<200_000).map { _ in points[Int.random(in: 0..<points.count, using: &rng)] } }
        var fitted = 0, errs: [Double] = []
        for (k, c) in todo.enumerated() {
            progress(Double(k) / Double(todo.count))
            try autoreleasepool {
                guard let src = CGImageSourceCreateWithURL(dataset.url(c.image) as CFURL, nil),
                      let img = CGImageSourceCreateThumbnailAtIndex(src, 0, [kCGImageSourceCreateThumbnailFromImageAlways: true,
                                                                            kCGImageSourceThumbnailMaxPixelSize: 1024,
                                                                            kCGImageSourceCreateThumbnailWithTransform: false] as CFDictionary)
                else { return }
                let raw = try predict(model, image: img)
                // resample to the prior's size (camera aspect, ~256 px long side)
                let s = Float(longSide) / Float(max(c.width, c.height))
                let ow = max(8, Int((Float(c.width) * s).rounded())), oh = max(8, Int((Float(c.height) * s).rounded()))
                var pred = [Float](repeating: 0, count: ow * oh)
                for y in 0..<oh { for x in 0..<ow {
                    pred[y * ow + x] = sample(raw, (Float(x) + 0.5) * Float(raw.width) / Float(ow) - 0.5, (Float(y) + 0.5) * Float(raw.height) / Float(oh) - 0.5)
                } }
                let pm = Map(values: pred, width: ow, height: oh)
                // project the scene's points into the prior's grid
                let w2c = c.c2w.inverse
                let fx = c.fx * Float(ow) / Float(c.width), fy = c.fy * Float(oh) / Float(c.height)
                let cx = c.cx * Float(ow) / Float(c.width), cy = c.cy * Float(oh) / Float(c.height)
                var sp: [Float] = [], sz: [Float] = []
                for p in pts {
                    let q = w2c * SIMD4(p, 1)
                    guard q.z > 1e-6 else { continue }
                    let u = fx * q.x / q.z + cx, v = fy * q.y / q.z + cy
                    guard u >= 0, v >= 0, u <= Float(ow - 1), v <= Float(oh - 1) else { continue }
                    sp.append(sample(pm, u, v)); sz.append(q.z)
                }
                guard let f = fit(pred: sp, z: sz), f.error <= 0.25 else { return }
                let medZ = sz.sorted()[sz.count / 2]
                var depth = [Float](repeating: 0, count: ow * oh)
                for i in depth.indices {
                    let inv = f.a * Double(pred[i]) + f.b
                    if inv > 1e-9 { let z = Float(1 / inv); depth[i] = z < 50 * medZ ? z : 0 }
                }
                out[c.id] = Map(values: depth, width: ow, height: oh)
                var d = Data(capacity: 8 + depth.count * 4)
                withUnsafeBytes(of: UInt32(ow).littleEndian) { d.append(contentsOf: $0) }
                withUnsafeBytes(of: UInt32(oh).littleEndian) { d.append(contentsOf: $0) }
                depth.withUnsafeBytes { d.append(contentsOf: $0) }
                try? d.write(to: dir.appendingPathComponent(cacheKey(c) + ".bin"))
                fitted += 1; errs.append(f.error)
            }
        }
        progress(1)
        if !errs.isEmpty {
            log(String(format: "predicted depth: %d of %d photos fitted to the scene (median depth error vs its 3D points %.0f %%)",
                       fitted, todo.count, 100 * errs.sorted()[errs.count / 2]))
        }
        return out
    }
}
