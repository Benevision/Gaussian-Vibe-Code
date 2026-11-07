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
                if let c = value("--config") { cfg = try JSONDecoder().decode(TrainingConfig.self, from: Data(contentsOf: URL(fileURLWithPath: c))) }
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
            let sem = DispatchSemaphore(value: 0)
            var exitCode: Int32 = 0
            let start = Date()
            Task.detached {
                do {
                    var last = ""
                    var ds = try await DatasetBuilder.build(source: source, options: o, into: out.appendingPathComponent("dataset")) { p in
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
                if let m = value("--mode").flatMap(Int.init), let dm = ViewerSettings.DisplayMode(rawValue: m) { st.displayMode = dm }
                guard let tex = r.renderOffscreen(width: size.0, height: size.1, camera: cam, settings: st) else { throw FormatError.invalid("render failed") }
                try ImageWriter.writePNG(texture: tex, to: URL(fileURLWithPath: a[i + 2]))
                print("Rendered \(g.count) splats → \(a[i + 2])")
                exit(0)
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
