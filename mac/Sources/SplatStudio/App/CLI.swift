import Foundation

/// Headless commands, handled before any UI is created:
///
///   SplatStudio --dump-settings                         Markdown settings reference to stdout
///   SplatStudio --train <dataset dir> --out <file.ply>  Train with the Metal trainer
///         [--preset preview|balanced|high|ultra] [--config cfg.json] [--iterations N]
///   SplatStudio --convert <in.ply|spz|splat> <out.ply|spz|splat> [--sh N]
///   SplatStudio --info <file>                           Print splat statistics
enum CLI {
    static func runIfRequested() {
        let a = CommandLine.arguments
        func value(_ flag: String) -> String? { a.firstIndex(of: flag).flatMap { $0 + 1 < a.count ? a[$0 + 1] : nil } }
        if a.contains("--dump-settings") { print(SettingsCatalog.markdown()); exit(0) }
        if let ds = value("--train") {
            do {
                let dataset = try PreparedDataset.load(URL(fileURLWithPath: ds))
                var cfg = TrainingConfig.Preset(rawValue: value("--preset") ?? "balanced")?.config ?? TrainingConfig()
                // --config: keys in the file override the preset (others keep the preset's values).
                if let c = value("--config") {
                    let over = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: c))) as? [String: Any] ?? [:]
                    var base = try JSONSerialization.jsonObject(with: JSONEncoder().encode(cfg)) as? [String: Any] ?? [:]
                    for (k, v) in over { base[k] = v }
                    cfg = try JSONDecoder().decode(TrainingConfig.self, from: JSONSerialization.data(withJSONObject: base))
                }
                if let it = value("--iterations").flatMap(Int.init) { cfg.iterations = it; cfg.densifyStop = min(cfg.densifyStop, Int(Double(it) * 0.8)) }
                if !dataset.hasDepth { cfg.depthWeight = 0 }
                let out = URL(fileURLWithPath: value("--out") ?? "trained.ply")
                print("Training \(dataset.manifest.cameras.count) cameras, \(cfg.iterations) steps, budget \(cfg.maxGaussians), strategy \(cfg.strategy.rawValue)")
                let t = MetalTrainer(dataset: dataset, config: cfg)
                try t.prepare { print("  \($0)") }
                var lastPrint = -1
                let g = try t.run(previewEvery: 0) { s in
                    if s.step / 500 != lastPrint / 500 || s.step == s.totalSteps {
                        lastPrint = s.step
                        print(String(format: "  step %6d/%d  loss %.4f  psnr %5.2f  eval %@  splats %8d  %4.0f it/s  eta %@  transient %.1f%%",
                                     s.step, s.totalSteps, s.loss, s.psnr, s.evalPSNR.map { String(format: "%5.2f", $0) } ?? "  -  ",
                                     s.numGaussians, s.itersPerSecond, formatDuration(s.eta), s.transientFraction * 100))
                        fflush(stdout)
                    }
                }
                try SplatFileFormat(rawValue: out.pathExtension.lowercased())?.save(g, to: out, shDegree: nil) ?? PLYCodec.writeGaussians(g, to: out)
                print("Wrote \(out.path) (\(g.count) splats)")
                exit(0)
            } catch { fputs("error: \(error.localizedDescription)\n", stderr); exit(1) }
        }
        if let src = value("--prepare") {
            // Headless dataset preparation: SplatStudio --prepare <photos dir | video | capture> --out <project dir>
            //   [--solver auto|objectcapture|capture|<splatprep feature id>] [--mask none|people|dynamic]
            let out = URL(fileURLWithPath: value("--out") ?? "project")
            guard let source = SourceDescription.guess(URL(fileURLWithPath: src)) else { fputs("error: unrecognised source\n", stderr); exit(1) }
            var o = PrepOptions()
            switch value("--solver") ?? "auto" {
            case "auto": o.poseSolver = .automatic
            case "objectcapture": o.poseSolver = .objectCapture
            case "capture": o.poseSolver = .fromCapture
            case let id: o.poseSolver = .splatprep(featureModelId: id, matching: "auto", mapper: "auto")
            }
            switch value("--mask") ?? "none" {
            case "people": o.masking.model = .appleVisionDynamic
            case "dynamic": o.masking.model = .splatprep(maskModelId: "dynamic")
            default: break
            }
            // [--worker http://host:port [--token T]] uses a Linux worker ("Remote" placement);
            // [--frames-only] stops after frame extraction (video / 360° video); [--end seconds] trims the video.
            var worker: WorkerClient? = nil
            if let w = value("--worker"), let u = URL(string: w) {
                worker = WorkerClient(baseURL: u, token: value("--token"))
                o.compute = .remote
            }
            if let e = value("--end").flatMap(Double.init) { o.video.endTime = e }
            let sem = DispatchSemaphore(value: 0)
            var exitCode: Int32 = 0
            let start = Date()
            if a.contains("--frames-only") {
                let wk = worker, opts = o.resolved
                Task.detached {
                    do {
                        let dir = out.appendingPathComponent("frames")
                        let p: PipelineProgressHandler = { p in print("  \(p.stage) \(p.fraction >= 0 ? String(format: "%3.0f%%", p.fraction * 100) : "  …") \(p.message)"); fflush(stdout) }
                        var fs: FrameSet
                        switch source {
                        case .video(let u):
                            if let r = try await DatasetBuilder.framesOnWorker(u, equirect: nil, opts, worker: wk, into: dir, progress: p) { fs = r }
                            else { fs = try await VideoFrameExtractor.extract(video: u, options: opts.video, into: dir, progress: p) }
                        case .video360(let u):
                            if let r = try await DatasetBuilder.framesOnWorker(u, equirect: opts.equirect, opts, worker: wk, into: dir, progress: p) { fs = r }
                            else { fs = try await Equirect360.extract(video: u, video: opts.video, options: opts.equirect, into: dir, progress: p) }
                        default: throw PipelineError.unsupported("--frames-only needs a video")
                        }
                        let groups = Set(fs.frames.compactMap(\.group)).count
                        print(String(format: "Frames ready: %d images (%d groups, %d masks, %d rejected) in %.0f s → %@", fs.frames.count, groups,
                                     fs.frames.filter { $0.mask != nil }.count, fs.rejected.count, Date().timeIntervalSince(start), dir.path))
                    } catch { fputs("error: \(error.localizedDescription)\n", stderr); exitCode = 1 }
                    sem.signal()
                }
                sem.wait()
                exit(exitCode)
            }
            Task.detached {
                do {
                    var last = ""
                    var ds = try await DatasetBuilder.build(source: source, options: o, into: out.appendingPathComponent("dataset"), worker: worker) { p in
                        let line = "\(p.stage) \(p.fraction >= 0 ? String(format: "%3.0f%%", p.fraction * 100) : "  …") \(p.message)"
                        if line != last { print("  " + line); fflush(stdout); last = line }
                    }
                    try ds.normalizeOrientation()
                    print(String(format: "Dataset ready: %d cameras in %.0f s → %@", ds.manifest.cameras.count, Date().timeIntervalSince(start), ds.root.path))
                } catch { fputs("error: \(error.localizedDescription)\n", stderr); exitCode = 1 }
                sem.signal()
            }
            sem.wait()
            exit(exitCode)
        }
        if let i = a.firstIndex(of: "--convert"), i + 2 < a.count {
            do {
                let g = try SplatFileFormat.load(url: URL(fileURLWithPath: a[i + 1]))
                let out = URL(fileURLWithPath: a[i + 2])
                guard let f = SplatFileFormat(rawValue: out.pathExtension.lowercased()) else { throw FormatError.unsupported(out.pathExtension) }
                try f.save(g, to: out, shDegree: value("--sh").flatMap(Int.init))
                print("Converted \(g.count) splats → \(out.path)")
                exit(0)
            } catch { fputs("error: \(error.localizedDescription)\n", stderr); exit(1) }
        }
        if let i = a.firstIndex(of: "--render"), i + 2 < a.count {
            do {
                let g = try SplatFileFormat.load(url: URL(fileURLWithPath: a[i + 1]))
                let r = try SplatRenderer()
                let sc = SplatScene(cloud: g)
                r.scene = sc
                var cam = ViewerCamera()
                let url = URL(fileURLWithPath: a[i + 1])
                cam.up = Orientation.upFromPLYComment(url) ?? Orientation.upFromProject(of: url) ?? Orientation.standardUp
                cam.frame(center: sc.center, radius: sc.radius)
                cam.yaw = Float(value("--yaw").flatMap(Double.init) ?? 0.6)
                cam.pitch = Float(value("--pitch").flatMap(Double.init) ?? 0.35)
                // --dataset <dir> --view <n>: render from training camera n (to compare with its photo)
                var size = (1600, 1000)
                if let dsPath = value("--dataset"), let v = value("--view").flatMap(Int.init) {
                    let ds = try PreparedDataset.load(URL(fileURLWithPath: dsPath))
                    let c = ds.manifest.cameras[min(max(0, v), ds.manifest.cameras.count - 1)]
                    cam.adopt(cameraToWorld: c.c2w, focusDistance: 1, fovY: 2 * atan(Float(c.height) / (2 * c.fy)), worldUp: cam.up)
                    size = (c.width, c.height)
                }
                var st = ViewerSettings()
                if a.contains("--aa") { st.antialiasOverride = true }
                if a.contains("--no-aa") { st.antialiasOverride = false }
                if let m = value("--mode").flatMap(Int.init), let dm = ViewerSettings.DisplayMode(rawValue: m) { st.displayMode = dm }
                if st.displayMode == .coverage, let dsPath = value("--dataset") {
                    let t0 = Date()
                    let c = try Coverage.compute(scene: sc, dataset: try PreparedDataset.load(URL(fileURLWithPath: dsPath)))
                    sc.coverage = c.score
                    print(String(format: "coverage: %.1f %% poor, %.1f %% unseen, median %d views, depth %@, %.2f s", c.poorFraction * 100, c.unseenFraction * 100, c.medianViews,
                                 c.usedDepth ? "yes" : "no", Date().timeIntervalSince(t0)))
                }
                if a.contains("--auto-clean"), let dsPath = value("--dataset") {
                    let c = try Coverage.compute(scene: sc, dataset: try PreparedDataset.load(URL(fileURLWithPath: dsPath)))
                    let plan = Coverage.cleanPlan(scene: sc, result: c, minViewsOutside: value("--clean-views").flatMap(Int.init) ?? 3)
                    let f = sc.flagsPointer, p = sc.positionsPointer
                    var del = 0, delWellSeen = 0
                    _ = p
                    for i in 0..<sc.count where plan.remove[i] { f[i] |= 1; del += 1; if c.views[i] >= 3 { delWellSeen += 1 } }
                    print("auto clean: removed \(del) of \(sc.count) (\(plan.unseenCount) unseen, \(plan.outsideCount) outside box; \(delWellSeen) of them seen by >= 3 photos); box \(plan.boxMin) – \(plan.boxMax)")
                }
                guard let tex = r.renderOffscreen(width: size.0, height: size.1, camera: cam, settings: st) else { throw FormatError.invalid("render failed") }
                try ImageWriter.writePNG(texture: tex, to: URL(fileURLWithPath: a[i + 2]))
                print("Rendered \(g.count) splats → \(a[i + 2])")
                exit(0)
            } catch { fputs("error: \(error.localizedDescription)\n", stderr); exit(1) }
        }
        // Developer aid: --fly-test <splat> flies the camera through the scene (forward, strafe, up) rendering
        // every step like the viewer does, to reproduce navigation crashes headlessly.
        // --eval <splat> --dataset <dir> [--sh N] [--aa|--no-aa]: held-out PSNR of any splat file (ours, Spirula,
        // gsplat…) on a dataset's holdout photos, rendered by the Metal trainer's rasteriser with exact intrinsics.
        if let f = value("--eval"), let dsPath = value("--dataset") {
            do {
                var g = try SplatFileFormat.load(url: URL(fileURLWithPath: f))
                let ds = try PreparedDataset.load(URL(fileURLWithPath: dsPath))
                var cfg = TrainingConfig()
                cfg.shDegree = value("--sh").flatMap(Int.init) ?? g.shDegree
                cfg.antialiased = a.contains("--aa") ? true : (a.contains("--no-aa") ? false : g.antialiased)
                cfg.maxImageSize = value("--max-size").flatMap(Int.init) ?? 1600
                cfg.randomInitCount = 10; cfg.depthWeight = 0
                let t = MetalTrainer(dataset: ds, config: cfg)
                try t.prepare { _ in }
                if a.contains("--peak-clean") {
                    let thr = value("--peak-thr").flatMap(Float.init) ?? (1.0 / 255.0)
                    let t0 = Date()
                    let peaks = try t.peakContribution(cloud: g)
                    let keep = peaks.map { $0 >= thr }
                    print(String(format: "peak clean: removed %d of %d splats (max contribution < %.4f) in %.1f s", keep.filter { !$0 }.count, g.count, thr, Date().timeIntervalSince(t0)))
                    g = g.filtered(keep)
                    if let out = value("--clean-out") { try PLYCodec.writeGaussians(g, to: URL(fileURLWithPath: out)) }
                }
                if let vf = value("--view-clean").flatMap(Float.init) {
                    // --view-clean 0.25 [--clean-out out.ply] : Auto clean (invisible + seen by too few photos)
                    let t0 = Date()
                    let c = try t.contributionCensus(cloud: g, viewThreshold: AutoClean.peakThreshold)
                    let p = AutoClean.plan(peak: c.peak, views: c.views, opacityLogits: g.opacities, fewViewsFraction: vf)
                    print(String(format: "view clean: removed %d invisible + %d seen by < %.0f of %d photos (typical %d) in %.1f s",
                                 p.invisible, p.fewViews, vf * Float(p.typicalViews), c.photos, p.typicalViews, Date().timeIntervalSince(t0)))
                    g = g.filtered(p.remove.map { !$0 })
                    if let out = value("--clean-out") { try PLYCodec.writeGaussians(g, to: URL(fileURLWithPath: out)) }
                }
                if let ct = value("--census").flatMap(Float.init) {
                    // --census 0.05 --census-out views.bin : per-splat count of photos it contributes ≥ thr to (uint32 LE)
                    let t0 = Date()
                    let c = try t.contributionCensus(cloud: g, viewThreshold: ct)
                    print(String(format: "census: %d splats over %d photos in %.1f s", g.count, c.photos, Date().timeIntervalSince(t0)))
                    if let out = value("--census-out") {
                        try c.views.withUnsafeBufferPointer { Data(buffer: $0) }.write(to: URL(fileURLWithPath: out))
                    }
                }
                if a.contains("--auto-clean") {
                    let sc = SplatScene(cloud: g)
                    let c = try Coverage.compute(scene: sc, dataset: ds)
                    let plan = Coverage.cleanPlan(scene: sc, result: c, minViewsOutside: value("--clean-views").flatMap(Int.init) ?? 3)
                    g = g.filtered(plan.remove.map { !$0 })
                    print("auto clean removed \(plan.remove.filter { $0 }.count) splats (\(plan.unseenCount) unseen, \(plan.outsideCount) outside box)")
                }
                let r = try t.evaluate(cloud: g)
                print(String(format: "eval %@: %d splats, SH %d, antialiased %@ → held-out PSNR %.2f dB over %d views (min %.2f, max %.2f)",
                             (f as NSString).lastPathComponent, g.count, cfg.shDegree, cfg.antialiased ? "yes" : "no", r.mean, r.perView.count,
                             r.perView.min() ?? 0, r.perView.max() ?? 0))
                if !r.failed.isEmpty { print("  not rendered: " + r.failed.joined(separator: ", ")) }
                if a.contains("--per-view") { for (n, p) in zip(r.names, r.perView) { print(String(format: "  %@ %.2f", n, p)) } }
                exit(0)
            } catch { fputs("error: \(error.localizedDescription)\n", stderr); exit(1) }
        }
        if let i = a.firstIndex(of: "--fetch-job"), i + 2 < a.count, let u = URL(string: a[i + 1]) {
            // SplatStudio --fetch-job http://host:7878 <jobId> --out result.ply : the re-attach path (wait through
            // outages, then download), for testing.
            let jid = a[i + 2], out = URL(fileURLWithPath: value("--out") ?? "\(jid).ply")
            let client = WorkerClient(baseURL: u, token: value("--token"))
            let sem = DispatchSemaphore(value: 0); var code: Int32 = 0
            Task.detached {
                do {
                    let g = try await client.finishJob(jid, progress: { s in print("  \(s.state.rawValue) \(s.step ?? 0)/\(s.totalSteps ?? 0)"); fflush(stdout) },
                                                       onOffline: { gone in print(String(format: "  unreachable for %.0f s, retrying", gone)); fflush(stdout) })
                    try PLYCodec.writeGaussians(g, to: out)
                    print("fetched \(g.count) splats → \(out.path)")
                } catch { fputs("error: \(error.localizedDescription)\n", stderr); code = 1 }
                sem.signal()
            }
            sem.wait(); exit(code)
        }
        if let f = value("--fly-test") {
            do {
                let g = try SplatFileFormat.load(url: URL(fileURLWithPath: f))
                let r = try SplatRenderer()
                let sc = SplatScene(cloud: g)
                r.scene = sc
                var cam = ViewerCamera()
                cam.frame(center: sc.center, radius: sc.radius)
                cam.mode = .fly
                let st = ViewerSettings()
                let step = max(sc.radius * 0.25, 0.05) * 0.05
                let moves: [SIMD3<Float>] = [SIMD3(0, 0, 1), SIMD3(1, 0, 0), SIMD3(0, 1, 0), SIMD3(0, 0, -1), SIMD3(-1, 0, 0)]
                var frames = 0
                for (k, mv) in moves.enumerated() {
                    for i in 0..<(k == 0 ? 400 : 120) {
                        cam.fly(mv, speed: step)
                        if i % 7 == 0 { cam.look(dx: 9, dy: (i % 14 == 0 ? 4 : -4)) }
                        guard r.renderOffscreen(width: 800, height: 500, camera: cam, settings: st) != nil else { throw FormatError.invalid("render failed at frame \(frames)") }
                        frames += 1
                    }
                    print("move \(k) ok, eye \(cam.eye), frames \(frames)"); fflush(stdout)
                }
                print("fly test passed: \(frames) frames"); exit(0)
            } catch { fputs("error: \(error.localizedDescription)\n", stderr); exit(1) }
        }
        if let f = value("--info") {
            do {
                let g = try SplatFileFormat.load(url: URL(fileURLWithPath: f))
                let b = g.robustBounds()
                print("splats: \(g.count)\nSH degree: \(g.shDegree)\nantialiased: \(g.antialiased)\nbounds (1–99 %): \(b.min) … \(b.max)")
                exit(0)
            } catch { fputs("error: \(error.localizedDescription)\n", stderr); exit(1) }
        }
    }
}
