import Foundation
import CoreGraphics
import ImageIO
import simd

/// Imports a Nerfstudio / Instant-NGP `transforms.json` dataset.
///
/// Supports global or per-frame intrinsics (`fl_x/fl_y/cx/cy/w/h` or `camera_angle_x/y`), distortion
/// (`k1…k4, p1, p2`, `camera_model` OPENCV / OPENCV_FISHEYE / PINHOLE), `mask_path`, `depth_file_path`
/// (16-bit PNG × `depth_unit_scale_factor`, default 0.001 → metres) and `ply_file_path`.
/// Poses are OpenGL/Blender camera-to-world (camera looks −Z, +Y up) and converted to OpenCV.
enum NerfstudioImporter {
    struct Intr {
        var w: Double?, h: Double?, flx: Double?, fly: Double?, cx: Double?, cy: Double?
        var angleX: Double?, angleY: Double?
        var k1 = 0.0, k2 = 0.0, k3 = 0.0, k4 = 0.0, p1 = 0.0, p2 = 0.0
        var model: String?

        init(_ d: [String: Any], base: Intr? = nil) {
            func num(_ k: String) -> Double? { (d[k] as? NSNumber)?.doubleValue }
            w = num("w") ?? base?.w; h = num("h") ?? base?.h
            flx = num("fl_x") ?? base?.flx; fly = num("fl_y") ?? base?.fly
            cx = num("cx") ?? base?.cx; cy = num("cy") ?? base?.cy
            angleX = num("camera_angle_x") ?? base?.angleX; angleY = num("camera_angle_y") ?? base?.angleY
            k1 = num("k1") ?? base?.k1 ?? 0; k2 = num("k2") ?? base?.k2 ?? 0; k3 = num("k3") ?? base?.k3 ?? 0
            k4 = num("k4") ?? base?.k4 ?? 0; p1 = num("p1") ?? base?.p1 ?? 0; p2 = num("p2") ?? base?.p2 ?? 0
            model = (d["camera_model"] as? String) ?? base?.model
        }

        func resolve(imageW: Int, imageH: Int) -> SourceIntrinsics {
            let W = w ?? Double(imageW), H = h ?? Double(imageH)
            var fx = flx, fy = fly
            if fx == nil, let a = angleX { fx = 0.5 * W / tan(0.5 * a) }
            if fy == nil, let a = angleY { fy = 0.5 * H / tan(0.5 * a) }
            let fxv = fx ?? fy ?? Double(ImageUtil.defaultFocal(width: Int(W), height: Int(H)))
            let fyv = fy ?? fxv
            var k = SourceIntrinsics(width: Int(W), height: Int(H), fx: fxv, fy: fyv, cx: cx ?? W / 2, cy: cy ?? H / 2)
            switch (model ?? "OPENCV").uppercased() {
            case "OPENCV_FISHEYE", "FISHEYE", "EQUIDISTANT":
                k.distortion = LensDistortion(kind: .fisheye, params: [k1, k2, k3, k4])
            case "PINHOLE", "SIMPLE_PINHOLE":
                break
            default:
                k.distortion = LensDistortion(kind: .opencv, params: [k1, k2, p1, p2, k3])
            }
            return k
        }
    }

    static func importDataset(transforms url: URL, options: ImportOptions = .init(), into out: URL,
                              progress: PipelineProgressHandler? = nil) async throws -> PreparedDataset {
        let stage = "Importing Nerfstudio"
        let jsonURL = url.hasDirectoryPath || url.pathExtension.isEmpty ? url.appendingPathComponent("transforms.json") : url
        let base = jsonURL.deletingLastPathComponent()
        guard let root = try JSONSerialization.jsonObject(with: Data(contentsOf: jsonURL)) as? [String: Any],
              let frames = root["frames"] as? [[String: Any]], !frames.isEmpty else {
            throw PipelineError.invalidData("transforms.json has no frames")
        }
        let global = Intr(root)
        let depthScale = (root["depth_unit_scale_factor"] as? NSNumber)?.doubleValue ?? 0.001
        let fm = FileManager.default
        try fm.ensureDirectory(out.appendingPathComponent("images"))
        progress?(PipelineProgress(stage: stage, fraction: 0, message: "\(frames.count) frames"))

        // Pre-parse frames (JSON objects are not Sendable).
        struct F: Sendable { var file: URL; var c2w: simd_float4x4; var k: SourceIntrinsics?; var mask: URL?; var depth: URL? }
        var parsed: [F] = []
        var perFrame: [Intr] = []
        for f in frames {
            guard let path = f["file_path"] as? String, let tm = f["transform_matrix"] as? [[NSNumber]], tm.count >= 3 else { continue }
            var file = base.appendingPathComponent(path)
            if file.pathExtension.isEmpty {   // Blender synthetic datasets omit the extension
                for e in ["png", "jpg", "jpeg"] where fm.fileExists(atPath: file.appendingPathExtension(e).path) { file = file.appendingPathExtension(e); break }
            }
            let rows = (0..<4).map { r in r < tm.count ? SIMD4<Float>(tm[r].map { $0.floatValue }.padded(4)) : SIMD4<Float>(0, 0, 0, 1) }
            let glc2w = simd_float4x4(rows: rows)
            parsed.append(F(file: file, c2w: glc2w * .flipYZ, k: nil,
                            mask: (f["mask_path"] as? String).map { base.appendingPathComponent($0) },
                            depth: (f["depth_file_path"] as? String).map { base.appendingPathComponent($0) }))
            perFrame.append(Intr(f, base: global))
        }
        guard !parsed.isEmpty else { throw PipelineError.invalidData("no valid frames in transforms.json") }
        for i in parsed.indices {
            let size = ImageUtil.pixelSizeRaw(parsed[i].file) ?? (Int(perFrame[i].w ?? 0), Int(perFrame[i].h ?? 0))
            parsed[i].k = perFrame[i].resolve(imageW: size.0, imageH: size.1)
        }

        let items = parsed
        let counter = ProgressCounter(total: items.count)
        let cams: [DatasetCamera?] = try await parallelMap(items.count) { i in
            let f = items[i]
            defer { let n = counter.increment(); progress?(PipelineProgress(stage: stage, fraction: Double(n) / Double(items.count), message: f.file.lastPathComponent)) }
            guard FileManager.default.fileExists(atPath: f.file.path), let k = f.k else { return nil }
            let stem = String(format: "%05d_", i) + f.file.deletingPathExtension().lastPathComponent
            let w = try DatasetImageWriter.write(source: f.file, intrinsics: k, datasetRoot: out, rel: "images/\(stem)", options: options)
            var c = DatasetCamera.make(id: i, image: w.image, width: w.width, height: w.height, fx: w.fx, fy: w.fy, cx: w.cx, cy: w.cy, c2w: f.c2w)
            c.mask = w.mask
            if let m = f.mask, FileManager.default.fileExists(atPath: m.path) {
                // Resample user mask to the written image size (and AND it with the undistortion mask).
                var (px, _, _) = try ImageUtil.readGray8(m, width: w.width, height: w.height)
                if let um = w.mask { let (u, _, _) = try ImageUtil.readGray8(out.appendingPathComponent(um)); for j in px.indices { px[j] = min(px[j], u[j]) } }
                let rel = "masks/\(stem).png"
                try FileManager.default.ensureDirectory(out.appendingPathComponent("masks"))
                try ImageUtil.writeGray8(px.map { $0 >= 128 ? 255 : 0 }, width: w.width, height: w.height, to: out.appendingPathComponent(rel))
                c.mask = rel
            }
            if let d = f.depth, FileManager.default.fileExists(atPath: d.path), let depth = try? readDepthPNG(d, scale: depthScale) {
                let rel = "depth/\(stem).bin"
                try FileManager.default.ensureDirectory(out.appendingPathComponent("depth"))
                try PreparedDataset.writeDepth(depth.values, to: out.appendingPathComponent(rel))
                c.depth = rel; c.depthWidth = depth.width; c.depthHeight = depth.height
            }
            return c
        }
        let valid = cams.compactMap { $0 }
        guard !valid.isEmpty else { throw PipelineError.noInput("no images of transforms.json were found") }

        var pointsRel: String? = nil
        if let ply = root["ply_file_path"] as? String {
            let src = base.appendingPathComponent(ply)
            if fm.fileExists(atPath: src.path) {
                let pc = try PLYCodec.readPointCloud(url: src)
                try PLYCodec.writePointCloud(pc, to: out.appendingPathComponent("points3d.ply"))
                pointsRel = "points3d.ply"
            }
        }
        let ds = try DatasetAssembly.finalize(root: out, cameras: valid, points: pointsRel, holdoutInterval: options.holdoutInterval)
        progress?(PipelineProgress(stage: stage, fraction: 1, message: "\(valid.count) cameras"))
        return ds
    }

    /// Reads a 16-bit (or 8-bit) grayscale depth PNG and scales it to metres.
    static func readDepthPNG(_ url: URL, scale: Double) throws -> (values: [Float], width: Int, height: Int) {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil), let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            throw PipelineError.decodeFailed(url.lastPathComponent)
        }
        let w = img.width, h = img.height
        var buf = [UInt16](repeating: 0, count: w * h)
        buf.withUnsafeMutableBytes { raw in
            let ctx = CGContext(data: raw.baseAddress, width: w, height: h, bitsPerComponent: 16, bytesPerRow: w * 2,
                                space: CGColorSpaceCreateDeviceGray(),
                                bitmapInfo: CGImageAlphaInfo.none.rawValue | CGBitmapInfo.byteOrder16Little.rawValue)
            ctx?.interpolationQuality = .none
            ctx?.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
        }
        // 8-bit sources are expanded to 16-bit by CG (×257); undo so `scale` applies to stored values.
        let div: Double = img.bitsPerComponent == 8 ? 257 : 1
        return (buf.map { Float(Double($0) / div * scale) }, w, h)
    }
}

extension Array where Element == Float {
    func padded(_ n: Int) -> [Float] { count >= n ? Array(prefix(n)) : self + [Float](repeating: 0, count: n - count) }
}

