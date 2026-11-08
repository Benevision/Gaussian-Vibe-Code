import Foundation
import CoreGraphics
import simd

/// Measured speed of one machine on the standard benchmark (same synthetic scene and settings everywhere).
struct DeviceBenchmark: Codable, Hashable, Sendable, Identifiable {
    var deviceId: String
    var name: String
    var date: Date
    var gpu: String?
    var software: String?
    /// Training steps per second at ~1 M splats, 1600×1200 images, SH degree 3 (steps 1,000–2,000).
    var trainStepsPerSecond: Double
    var splats: Int
    var megapixels: Double
    /// Mac → worker / worker → Mac (MB/s); nil for this Mac.
    var uploadMBps: Double?
    var downloadMBps: Double?
    var id: String { deviceId }

    /// Seconds per training step for a job of this size (scaled from the benchmark load: per-step cost is
    /// roughly half splat-bound and half pixel-bound).
    func secondsPerStep(splats n: Int, megapixels mp: Double) -> Double {
        let base = 1 / max(trainStepsPerSecond, 0.01)
        let s = 0.5 * Double(max(n, 1)) / Double(max(splats, 1)) + 0.5 * max(mp, 0.05) / max(megapixels, 0.05)
        return base * s
    }
}

/// Saved benchmark results (Application Support/SplatStudio/benchmarks.json). Thread-safe: the scheduler
/// reads it from any context.
final class BenchmarkStore: @unchecked Sendable {
    static let shared = BenchmarkStore()
    private let lock = NSLock()
    private var cache: [String: DeviceBenchmark]
    private let url: URL

    private init() {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("SplatStudio")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        url = dir.appendingPathComponent("benchmarks.json")
        let dec = JSONDecoder(); dec.dateDecodingStrategy = .iso8601
        cache = (try? dec.decode([String: DeviceBenchmark].self, from: Data(contentsOf: url))) ?? [:]
    }

    func result(for deviceId: String) -> DeviceBenchmark? { lock.lock(); defer { lock.unlock() }; return cache[deviceId] }
    var all: [DeviceBenchmark] { lock.lock(); defer { lock.unlock() }; return cache.values.sorted { $0.trainStepsPerSecond > $1.trainStepsPerSecond } }

    func save(_ b: DeviceBenchmark) {
        lock.lock(); cache[b.deviceId] = b; let snapshot = cache; lock.unlock()
        let enc = JSONEncoder(); enc.dateEncodingStrategy = .iso8601; enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        try? enc.encode(snapshot).write(to: url, options: .atomic)
    }

    /// Scheduler throughput (1.0 ≈ 20 benchmark steps/s), when this device has been measured.
    func throughput(for deviceId: String) -> Double? { result(for: deviceId).map { $0.trainStepsPerSecond / 20 } }

    /// Estimated wall time to train `config` on `dataset` on a measured device (incl. uploading it to a worker).
    func estimate(deviceId: String, dataset: PreparedDataset, config: TrainingConfig, uploadBytes: Int64? = nil) -> TimeInterval? {
        guard let b = result(for: deviceId) else { return nil }
        let cams = dataset.manifest.cameras
        guard let c = cams.first else { return nil }
        let scale = min(1, Double(config.maxImageSize) / Double(max(c.width, c.height)))
        let mp = Double(c.width * c.height) * scale * scale / 1e6
        var t = Double(config.iterations) * b.secondsPerStep(splats: config.maxGaussians, megapixels: mp)
        if let up = b.uploadMBps, up > 0, let bytes = uploadBytes { t += Double(bytes) / 1e6 / up }
        return t
    }
}

/// The standard benchmark: a synthetic scene rendered by our own renderer, trained for 2,000 steps with the
/// same settings on this Mac and on each worker, plus Mac ↔ worker transfer speed. Results feed placement
/// ("which machine trains this faster, including the upload?") and the Both-mode block split.
enum SystemBenchmark {
    static let width = 1600, height = 1200, views = 48, splats = 1_000_000
    static let steps = 2000, measureFrom = 1000

    static var config: TrainingConfig {
        var c = TrainingConfig()
        c.iterations = steps; c.maxGaussians = splats; c.maxImageSize = max(width, height)
        c.densifyStart = 200; c.densifyStop = 1500; c.shDegree = 3; c.shDegreeInterval = 300
        c.evalInterval = 0; c.previewInterval = 0
        c.appearanceCorrection = false; c.transientMasking = false; c.depthWeight = 0
        c.normalWeight = 0; c.depthPriors = "off"   // a speed test: comparable with results from before these losses
        c.seed = 7
        return c
    }

    /// Creates (or reuses) the synthetic dataset: textured ground, boxes and pillars, 48 views on two rings.
    static func dataset(progress: ((String) -> Void)? = nil) throws -> PreparedDataset {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("SplatStudio/benchmark-scene-v1")
        if let ds = try? PreparedDataset.load(dir), ds.manifest.cameras.count == views,
           FileManager.default.fileExists(atPath: dir.appendingPathComponent(ds.manifest.cameras.last!.image).path) {
            return ds
        }
        try? FileManager.default.removeItem(at: dir)
        try FileManager.default.createDirectory(at: dir.appendingPathComponent("images"), withIntermediateDirectories: true)
        progress?("Building the benchmark scene")
        var rng = SystemRandomNumberGenerator.seeded(7)
        let truth = scene(&rng)

        // Cameras: two rings looking at the middle (OpenCV: x right, y down, z forward; up = −y).
        var m = DatasetManifest()
        m.upVector = [0, -1, 0]; m.sceneCenter = [0, 0, 0]; m.sceneExtent = 7
        let fx: Float = 1400
        let target = SIMD3<Float>(0, 0.3, 0)
        for i in 0..<views {
            let ring = i % 2, a = Float(i) / Float(views) * 2 * .pi
            let r: Float = ring == 0 ? 6 : 4.5, y: Float = ring == 0 ? -2.2 : -3.6
            let eye = SIMD3<Float>(r * cos(a), y, r * sin(a))
            let z = simd_normalize(target - eye)
            let x = simd_normalize(simd_cross(SIMD3<Float>(0, 1, 0), z))
            let yv = simd_cross(z, x)
            let c2w = simd_float4x4(columns: (SIMD4(x, 0), SIMD4(yv, 0), SIMD4(z, 0), SIMD4(eye, 1)))
            var cam = DatasetCamera.make(id: i, image: String(format: "images/%03d.jpg", i), width: width, height: height,
                                         fx: fx, fy: fx, cx: Float(width) / 2, cy: Float(height) / 2, c2w: c2w)
            cam.holdout = i % 8 == 7
            m.cameras.append(cam)
        }
        // Placeholder photos so the trainer can load the views, then render the truth into them.
        let gray = [UInt8](repeating: 128, count: width * height * 4)
        for c in m.cameras { try writeJPEG(gray, width: width, height: height, to: dir.appendingPathComponent(c.image)) }
        // Starting points: the truth's centres, jittered, repeated up to the splat budget (so training runs at
        // full load from the start and the measurement doesn't depend on how fast densification grows).
        var pc = PointCloud()
        let n = truth.count
        for k in 0..<splats {
            let i = k % n
            let p = SIMD3<Float>(truth.positions[i * 3], truth.positions[i * 3 + 1], truth.positions[i * 3 + 2])
            pc.points.append(p + SIMD3(Float.random(in: -0.02...0.02, using: &rng), Float.random(in: -0.02...0.02, using: &rng),
                                       Float.random(in: -0.02...0.02, using: &rng)))
            let col = (SIMD3<Float>(truth.sh[i * 48], truth.sh[i * 48 + 1], truth.sh[i * 48 + 2]) * GaussianCloud.C0 + 0.5) * 255
            pc.colors.append(SIMD3<UInt8>(clamping: SIMD3<Int>(Int(col.x), Int(col.y), Int(col.z))))
        }
        try PLYCodec.writePointCloud(pc, to: dir.appendingPathComponent("points3d.ply"))
        m.points = "points3d.ply"
        var ds = PreparedDataset(root: dir, manifest: m)
        try ds.save()
        progress?("Rendering the benchmark photos")
        var cfg = config; cfg.shDegree = 0
        let t = MetalTrainer(dataset: ds, config: cfg)
        try t.prepare { _ in }
        try t.renderEach(cloud: truth) { cam, w, h, rgba in
            try writeJPEG(rgba, width: w, height: h, to: dir.appendingPathComponent(cam.image))
        }
        ds = try PreparedDataset.load(dir)
        return ds
    }

    /// ~420 k ground-truth splats: a noise-textured ground plane, boxes and pillars.
    private static func scene(_ rng: inout SystemRandomNumberGenerator.Seeded) -> GaussianCloud {
        var g = GaussianCloud()
        g.shDegree = 0
        func add(_ p: SIMD3<Float>, _ s: SIMD3<Float>, _ c: SIMD3<Float>) {
            g.positions += [p.x, p.y, p.z]
            g.scales += [log(s.x), log(s.y), log(s.z)]
            g.rotations += [1, 0, 0, 0]
            g.opacities += [3]          // logit ≈ 0.95
            var sh = [Float](repeating: 0, count: 48)
            sh[0] = (c.x - 0.5) / GaussianCloud.C0; sh[1] = (c.y - 0.5) / GaussianCloud.C0; sh[2] = (c.z - 0.5) / GaussianCloud.C0
            g.sh += sh
        }
        func noise(_ x: Float, _ z: Float) -> Float {
            (sin(x * 3.1) * cos(z * 2.7) + sin(x * 11.3 + z * 7.1) * 0.5 + sin(x * 31 - z * 23) * 0.25) / 1.75 * 0.5 + 0.5
        }
        // Ground 12 × 12 m at y = 1 (down is +y).
        for _ in 0..<300_000 {
            let x = Float.random(in: -6...6, using: &rng), z = Float.random(in: -6...6, using: &rng)
            let v = noise(x, z), chk: Float = (Int(floor(x)) + Int(floor(z))) % 2 == 0 ? 0.1 : -0.1
            add(SIMD3(x, 1, z), SIMD3(0.025, 0.004, 0.025), SIMD3(0.35 + 0.4 * v + chk, 0.45 + 0.3 * v, 0.25 + 0.2 * v))
        }
        // Boxes: surface splats on their five visible faces.
        for b in 0..<24 {
            let cx = Float.random(in: -4.5...4.5, using: &rng), cz = Float.random(in: -4.5...4.5, using: &rng)
            let hs = SIMD3<Float>(Float.random(in: 0.2...0.7, using: &rng), Float.random(in: 0.2...1.2, using: &rng), Float.random(in: 0.2...0.7, using: &rng))
            let hue = SIMD3<Float>(Float(b % 3 == 0 ? 1 : 0.3), Float(b % 3 == 1 ? 0.9 : 0.35), Float(b % 3 == 2 ? 1 : 0.4))
            for _ in 0..<4_000 {
                var p = SIMD3<Float>(Float.random(in: -1...1, using: &rng), Float.random(in: -1...1, using: &rng), Float.random(in: -1...1, using: &rng))
                let axis = Int.random(in: 0..<3, using: &rng)
                p[axis] = p[axis] < 0 ? -1 : 1
                if axis == 1 && p.y > 0 { p.y = -1 }      // no bottom face
                let pos = SIMD3(cx, 1 - hs.y, cz) + p * hs
                let shade = 0.55 + 0.45 * noise(pos.x * 4 + pos.y * 3, pos.z * 4 - pos.y)
                var s = SIMD3<Float>(0.02, 0.02, 0.02); s[axis] = 0.004
                add(pos, s, hue * shade)
            }
        }
        // Pillars (thin detail).
        for _ in 0..<12 {
            let cx = Float.random(in: -5...5, using: &rng), cz = Float.random(in: -5...5, using: &rng)
            for k in 0..<2_000 {
                let yy = 1 - Float(k) / 2000 * 2.5
                let a = Float.random(in: 0...(2 * .pi), using: &rng)
                add(SIMD3(cx + 0.06 * cos(a), yy, cz + 0.06 * sin(a)), SIMD3(0.01, 0.015, 0.01),
                    SIMD3(0.9, 0.9, 0.85) * (0.7 + 0.3 * Float(k % 50) / 50))
            }
        }
        return g
    }

    static func writeJPEG(_ rgba: [UInt8], width: Int, height: Int, to url: URL) throws {
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        guard let provider = CGDataProvider(data: Data(rgba) as CFData),
              let img = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4, space: cs,
                                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue), provider: provider,
                                decode: nil, shouldInterpolate: false, intent: .defaultIntent) else {
            throw PipelineError.encodeFailed(url.lastPathComponent)
        }
        try ImageUtil.write(img, to: url, encoding: .jpeg, quality: 0.92)
    }

    private static var megapixels: Double { Double(width * height) / 1e6 }

    /// Trains the benchmark on this Mac.
    static func runLocal(progress: @escaping @Sendable (String, Double) -> Void) async throws -> DeviceBenchmark {
        let ds = try dataset { progress($0, 0) }
        return try await Task.detached(priority: .userInitiated) {
            let t = MetalTrainer(dataset: ds, config: config)
            try t.prepare { progress($0, 0.02) }
            var t0: Date? = nil, s0 = 0, t1 = Date(), s1 = 0
            _ = try t.run(previewEvery: 0, onStatus: { s in
                if t0 == nil, s.step >= measureFrom { t0 = Date(); s0 = s.step }
                t1 = Date(); s1 = s.step
                progress(String(format: "Training on this Mac — step %d / %d", s.step, steps), Double(s.step) / Double(steps))
            })
            guard let t0, s1 > s0 else { throw PipelineError.processFailed("benchmark run too short to measure") }
            let rate = Double(s1 - s0) / t1.timeIntervalSince(t0)
            return DeviceBenchmark(deviceId: "local", name: LocalDeviceName.current, date: Date(),
                                   gpu: MetalContext.shared.device.name, software: appVersion, trainStepsPerSecond: rate,
                                   splats: splats, megapixels: megapixels)
        }.value
    }

    /// Uploads the benchmark scene to a worker, trains it there and downloads the result (timing both transfers).
    static func runRemote(endpoint: WorkerEndpoint, client: WorkerClient,
                          progress: @escaping @Sendable (String, Double) -> Void) async throws -> DeviceBenchmark {
        let ds = try dataset { progress($0, 0) }
        let info = try await client.info(timeout: 10)
        let zip = FileManager.default.temporaryDirectory.appendingPathComponent("splat-bench-\(UUID().uuidString.prefix(6)).zip")
        defer { try? FileManager.default.removeItem(at: zip) }
        try await ZipUtil.zip(directory: ds.root, to: zip)
        let bytes = (try? FileManager.default.attributesOfItem(atPath: zip.path)[.size] as? Int64) ?? 0
        progress("Uploading the benchmark scene to \(endpoint.name)", 0.02)
        let up0 = Date()
        let did = try await client.uploadDataset(zip: zip) { f in progress("Uploading to \(endpoint.name)", 0.02 + 0.08 * f) }
        let upSecs = Date().timeIntervalSince(up0)
        let jid = try await client.createJob(datasetId: did, name: "SplatWork benchmark", config: config)
        var t0: Date? = nil, s0 = 0, t1 = Date(), s1 = 0
        let final = try await client.waitForJob(jid) { s in
            let step = s.step ?? 0
            if t0 == nil, step >= measureFrom { t0 = Date(); s0 = step }
            if step > s1 { t1 = Date(); s1 = step }
            progress(String(format: "Training on %@ — step %d / %d", endpoint.name, step, steps), 0.1 + 0.8 * Double(step) / Double(steps))
        }
        guard final.state == .completed else { throw PipelineError.processFailed("benchmark job \(final.state): \(final.message ?? "")") }
        guard let t0, s1 > s0, t1 > t0 else { throw PipelineError.processFailed("benchmark run too short to measure") }
        let rate = Double(s1 - s0) / t1.timeIntervalSince(t0)
        progress("Downloading the result from \(endpoint.name)", 0.92)
        let dst = FileManager.default.temporaryDirectory.appendingPathComponent("splat-bench-\(jid).ply")
        defer { try? FileManager.default.removeItem(at: dst) }
        let dl0 = Date()
        try await client.downloadResult(jid, to: dst)
        let dlSecs = Date().timeIntervalSince(dl0)
        let dlBytes = (try? FileManager.default.attributesOfItem(atPath: dst.path)[.size] as? Int64) ?? 0
        try? await client.deleteJob(jid)
        progress("Done", 1)
        return DeviceBenchmark(deviceId: endpoint.id, name: endpoint.name, date: Date(), gpu: info.gpus?.first?.name,
                               software: info.backend, trainStepsPerSecond: rate, splats: splats, megapixels: megapixels,
                               uploadMBps: upSecs > 0 ? Double(bytes) / 1e6 / upSecs : nil,
                               downloadMBps: dlSecs > 0 ? Double(dlBytes) / 1e6 / dlSecs : nil)
    }

    private static var appVersion: String {
        "SplatWork " + ((Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String) ?? "dev")
    }
}

extension SystemRandomNumberGenerator {
    /// Deterministic generator (SplitMix64) so every machine benchmarks the identical scene.
    struct Seeded: RandomNumberGenerator {
        var state: UInt64
        mutating func next() -> UInt64 {
            state &+= 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            return z ^ (z >> 31)
        }
    }
    static func seeded(_ s: UInt64) -> Seeded { Seeded(state: s) }
}

/// "How long would this take where?" from benchmark results — shown next to the device picker and used to
/// pick the machine for a scene too small to split.
enum TrainingTimeEstimate {
    struct Entry: Identifiable, Sendable { var deviceId: String; var name: String; var seconds: TimeInterval; var uploadSeconds: TimeInterval; var id: String { deviceId } }

    /// Bytes a remote run must upload (images, masks, depth…); 0 when the worker already holds the frames.
    static func uploadBytes(_ ds: PreparedDataset) -> Int64 {
        // Cached for a minute: SwiftUI asks on every redraw and a 360° project has ~10 k files.
        let key = ds.root.path
        cacheLock.lock()
        if let c = sizeCache[key], Date().timeIntervalSince(c.0) < 60 { cacheLock.unlock(); return c.1 }
        cacheLock.unlock()
        let v = measureUploadBytes(ds)
        cacheLock.lock(); sizeCache[key] = (Date(), v); cacheLock.unlock()
        return v
    }
    nonisolated(unsafe) private static var sizeCache: [String: (Date, Int64)] = [:]
    private static let cacheLock = NSLock()

    private static func measureUploadBytes(_ ds: PreparedDataset) -> Int64 {
        let store = RemoteStore.load(ds.root)
        var total: Int64 = 0
        let skip = Set(store?.dirs ?? [])
        guard let e = FileManager.default.enumerator(at: ds.root, includingPropertiesForKeys: [.fileSizeKey], options: [.skipsHiddenFiles]) else { return 0 }
        for case let u as URL in e {
            let rel = u.path.dropFirst(ds.root.path.count + 1)
            if let top = rel.split(separator: "/").first, skip.contains(String(top)) { continue }
            total += Int64((try? u.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
        return total
    }

    /// Estimates for this Mac and each given worker that has been benchmarked.
    static func entries(dataset: PreparedDataset, config: TrainingConfig, workers: [WorkerEndpoint]) -> [Entry] {
        let store = BenchmarkStore.shared
        var out: [Entry] = []
        if let t = store.estimate(deviceId: "local", dataset: dataset, config: config) {
            out.append(Entry(deviceId: "local", name: "This Mac", seconds: t, uploadSeconds: 0))
        }
        let measured = workers.filter { store.result(for: $0.id) != nil }
        guard !measured.isEmpty else { return out }
        let bytes = uploadBytes(dataset)
        for w in measured {
            guard let b = store.result(for: w.id), let t = store.estimate(deviceId: w.id, dataset: dataset, config: config, uploadBytes: bytes) else { continue }
            let up = (b.uploadMBps ?? 0) > 0 ? Double(bytes) / 1e6 / b.uploadMBps! : 0
            out.append(Entry(deviceId: w.id, name: w.name, seconds: t, uploadSeconds: up))
        }
        return out
    }
}

/// How much GPU memory a Mac training run needs: every photo is kept on the GPU (RGBA8 at the training size,
/// plus depth), each splat costs ≈ 1.4 KB (parameters, gradients, two Adam moments, per-frame scratch, tile
/// lists), and per-pixel scratch for the largest view.
enum MemoryForecast {
    static func macBytes(dataset: PreparedDataset, config: TrainingConfig) -> Int64 {
        var images: Int64 = 0, maxPx: Int64 = 0
        for c in dataset.manifest.cameras {
            let longest = max(c.width, c.height)
            let s = longest > config.maxImageSize ? Double(config.maxImageSize) / Double(longest) : 1
            let px = Int64((Double(c.width) * s).rounded()) * Int64((Double(c.height) * s).rounded())
            images += px * 4
            maxPx = max(maxPx, px)
            if let dw = c.depthWidth, let dh = c.depthHeight, config.depthWeight > 0 { images += Int64(dw * dh * 4) }
        }
        // measured per-splat buffers (bedroom, 1.5M): fp32 Adam ≈ 1,400 B, compact (8/16-bit) ≈ 1,080 B
        let splats = Int64(Double(config.maxGaussians) * 1.1) * (config.optimizerPrecision == "compact" ? 1_080 : 1_400)
        return images + splats + maxPx * 180 + 300_000_000      // + pipelines, sort scratch, app
    }

    /// Why a run would not fit on this worker (GPU memory for the splats; host RAM for the photos on workers
    /// that hold every photo in memory), with what to change; nil when it fits or nothing is known.
    static func workerProblem(info: WorkerInfo, dataset: PreparedDataset, config: TrainingConfig) -> String? {
        // iPhones / Mac workers size jobs to their own memory (fewer splats, smaller images down to 480 px); only
        // a job that can't fit even then needs splitting. They report what they may still allocate.
        if let v = info.workerVersion, v.hasPrefix("ios") || v.hasPrefix("mac") {
            // Ask the very sizing the worker runs (TrainKit.budget), so the Mac and the device can't disagree.
            guard let avail = info.memory?.availableMB, avail > 0 else { return nil }
            var c = config
            do {
                _ = try TrainKit.budget(dataset: dataset, config: &c, availableBytes: UInt64(avail) * 1_048_576)
                return nil
            } catch {
                return "\(info.name) can't hold all \(dataset.manifest.cameras.count) photos: \(error.localizedDescription)"
            }
        }
        var images: Int64 = 0, maxPx: Int64 = 0
        for c in dataset.manifest.cameras {
            let longest = max(c.width, c.height)
            let s = longest > config.maxImageSize ? Double(config.maxImageSize) / Double(longest) : 1
            let px = Int64((Double(c.width) * s).rounded()) * Int64((Double(c.height) * s).rounded())
            images += px * 4; maxPx = max(maxPx, px)
        }
        let name = info.name
        // Photos: workers before 1.3.13 load all of them into RAM (and pinned copies) — the PC froze on a
        // 9,824-view 360° set. Newer workers stream them from disk.
        if !info.streamsImages, info.capabilities?.contains("train") ?? true {
            let ram = Int64(info.memory?.availableMB ?? 0) * 1_048_576
            if images > 3 << 30, ram == 0 || images * 2 > ram {
                return "\(name) would hold all \(dataset.manifest.cameras.count) photos in memory (about \(formatBytes(Int(images * 2)))), "
                    + "which can freeze it. Update SplatWork Node on \(name) to 1.3.13 or later (it streams photos from disk), or lower Max image size."
            }
        }
        // GPU: splat state (params, gradients, Adam, rasteriser buffers) + per-pixel scratch + CUDA context.
        if let vram = info.gpus?.compactMap(\.memoryMB).max(), vram > 0 {
            let bytesPerSplat: Int64 = config.optimizerPrecision == "compact" ? 800 : 1_150
            let need = Int64(Double(config.maxGaussians) * 1.05) * bytesPerSplat + maxPx * 200 + 1_400_000_000
            let have = Int64(Double(vram) * 0.92) * 1_048_576
            if need > have {
                let fit = max(100_000, Int((Double(have - maxPx * 200 - 1_400_000_000) / Double(bytesPerSplat) / 1.05) / 100_000) * 100_000)
                return "\(formatCount(config.maxGaussians)) splats need about \(formatBytes(Int(need))) of GPU memory; \(name) has \(formatBytes(Int(have))). "
                    + "Set Max splats to \(formatCount(fit)) or less."
            }
        }
        return nil
    }

    /// GPU memory this Mac can use for one app (Metal's recommended working set).
    static var macAvailable: Int64 { Int64(MetalContext.shared.device.recommendedMaxWorkingSetSize) }

    /// A message when a Mac run would not fit, with what to change; nil when it fits.
    static func macProblem(dataset: PreparedDataset, config: TrainingConfig) -> String? {
        let need = macBytes(dataset: dataset, config: config), have = macAvailable
        guard need > Int64(Double(have) * 0.9) else { return nil }
        // the largest image size that would fit (images dominate for big photo sets)
        var fit = config.maxImageSize
        var c = config
        while fit > 400 {
            fit = Int(Double(fit) * 0.85); c.maxImageSize = fit
            if macBytes(dataset: dataset, config: c) <= Int64(Double(have) * 0.9) { break }
        }
        return "Training this on the Mac needs about \(formatBytes(Int(need))) of GPU memory, but this Mac has \(formatBytes(Int(have))) for the app: "
            + "\(dataset.manifest.cameras.count) photos are all kept on the GPU. Train it on a Linux worker, or set Max image size to \(fit) or less."
    }
}
