import Foundation

/// Finds parts of the frame that never change while the camera moves — a lens rim or vignette corner, a
/// burned-in timestamp or logo, the drone's own propellers or landing gear, a selfie stick or car bonnet — and
/// masks them out of training (they would otherwise be baked into the scene as a smear that follows the camera).
///
/// Per camera (each 360° view direction is its own camera), up to 48 frames are sampled at low resolution. A
/// pixel is static when it barely varies over time AND it either has structure in the average image (edges of
/// an overlay or the rig) or is nearly black (lens rim). Plain sky also barely varies but has no structure, so
/// it is kept. If the camera itself hardly moves (more than a quarter of the frame looks static), nothing is
/// masked for that camera.
enum StaticRegions {
    struct Report: Sendable { var camera: String; var frames: Int; var staticFraction: Double; var applied: Bool }

    static let sampleWidth = 192
    static let maxSamples = 48

    /// Camera key: the 360° view direction (`…_v05.jpg`) or the frame size and source.
    static func cameraKey(_ f: IngestedFrame) -> String {
        if f.rotationToRig != nil, let r = f.image.range(of: #"_v\d+"#, options: .regularExpression) { return String(f.image[r]) }
        return "\(f.width)x\(f.height) \(f.source ?? "")"
    }

    /// Low-resolution keep mask (255 keep / 0 static) for one camera, or nil when nothing should be masked.
    static func detect(_ frames: [IngestedFrame], root: URL) -> (keep: [UInt8], width: Int, height: Int, staticFraction: Double)? {
        guard frames.count >= 12, let first = frames.first else { return nil }
        let w = sampleWidth, h = max(8, Int((Double(sampleWidth) * Double(first.height) / Double(first.width)).rounded()))
        let step = max(1, frames.count / maxSamples)
        let picks = stride(from: 0, to: frames.count, by: step).map { frames[$0] }
        var sum = [Double](repeating: 0, count: w * h), sq = sum
        var used = 0
        for f in picks {
            guard let (px, _, _) = try? ImageUtil.readGray8(root.appendingPathComponent(f.image), width: w, height: h) else { continue }
            for i in 0..<(w * h) { let v = Double(px[i]); sum[i] += v; sq[i] += v * v }
            used += 1
        }
        guard used >= 10 else { return nil }
        let n = Double(used)
        let mean = sum.map { $0 / n }
        let std = (0..<(w * h)).map { (max(0, sq[$0] / n - mean[$0] * mean[$0])).squareRoot() }
        // structure of the average image (static overlays keep their edges; moving scenery averages to blur)
        var grad = [Double](repeating: 0, count: w * h)
        for y in 1..<(h - 1) {
            for x in 1..<(w - 1) {
                let i = y * w + x
                grad[i] = abs(mean[i + 1] - mean[i - 1]) + abs(mean[i + w] - mean[i - w])
            }
        }
        // A pixel counts only if its whole 3×3 neighbourhood is unchanging: the edge where a fixed band (sky above a
        // horizon that stays put in forward flight) meets moving scenery is not an overlay.
        let still = std.map { $0 < 4 }
        var stat = [Bool](repeating: false, count: w * h)
        for y in 1..<(h - 1) {
            for x in 1..<(w - 1) {
                let i = y * w + x
                guard mean[i] < 12 || grad[i] > 20 else { continue }
                var all = true
                for dy in -1...1 where all { for dx in -1...1 where !still[i + dy * w + dx] { all = false } }
                stat[i] = all
            }
        }
        // grow by 3 px (sample resolution) so the masked region covers its soft edges
        var grown = stat
        for y in 0..<h {
            for x in 0..<w where stat[y * w + x] {
                for dy in -3...3 { for dx in -3...3 {
                    let yy = y + dy, xx = x + dx
                    if yy >= 0, yy < h, xx >= 0, xx < w { grown[yy * w + xx] = true }
                } }
            }
        }
        let frac = Double(grown.filter { $0 }.count) / Double(w * h)
        guard frac >= 0.002 else { return nil }
        return (grown.map { $0 ? 0 : 255 }, w, h, frac)
    }

    /// Detects static regions per camera and ANDs them into each frame's mask. Returns one report per camera.
    @discardableResult
    static func apply(to frames: inout FrameSet, dryRun: Bool = false, progress: PipelineProgressHandler? = nil) async throws -> [Report] {
        let groups = Dictionary(grouping: frames.frames.indices, by: { cameraKey(frames.frames[$0]) })
        var reports: [Report] = []
        var k = 0
        for (key, idx) in groups.sorted(by: { $0.key < $1.key }) {
            k += 1
            progress?(PipelineProgress(stage: "Finding fixed parts of the frame", fraction: Double(k) / Double(groups.count), message: key))
            guard let r = detect(idx.map { frames.frames[$0] }, root: frames.root) else {
                reports.append(Report(camera: key, frames: idx.count, staticFraction: 0, applied: false)); continue
            }
            let apply = r.staticFraction <= 0.25          // more = the camera barely moves; masking would remove the scene
            reports.append(Report(camera: key, frames: idx.count, staticFraction: r.staticFraction, applied: apply && !dryRun))
            guard apply, !dryRun else { continue }
            let root = frames.root
            try FileManager.default.ensureDirectory(root.appendingPathComponent("masks"))
            let list = idx.map { frames.frames[$0] }
            let written: [String?] = try await parallelMap(list.count, maxConcurrent: 6) { j in
                let f = list[j]
                let keep = upsample(r.keep, r.width, r.height, to: f.width, f.height)
                return try FrameMasker.combineAndWrite(keep, frame: f, root: root)
            }
            for (j, i) in idx.enumerated() { frames.frames[i].mask = written[j] }
        }
        if !dryRun { try frames.save() }
        return reports
    }

    /// Nearest-neighbour upsampling of a low-resolution keep mask to the frame size.
    static func upsample(_ m: [UInt8], _ w: Int, _ h: Int, to W: Int, _ H: Int) -> [UInt8] {
        var out = [UInt8](repeating: 255, count: W * H)
        for y in 0..<H {
            let sy = min(h - 1, y * h / H)
            for x in 0..<W { out[y * W + x] = m[sy * w + min(w - 1, x * w / W)] }
        }
        return out
    }
}
