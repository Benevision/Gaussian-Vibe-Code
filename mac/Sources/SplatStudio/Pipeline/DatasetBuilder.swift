import Foundation
import simd

/// What the user dropped into SplatWork.
enum SourceDescription: Codable, Hashable, Sendable {
    /// A folder of photos (JPEG / HEIC / PNG / TIFF / RAW).
    case photos(folder: URL)
    /// An explicit list of photo files.
    case photoFiles([URL])
    /// A regular (monoscopic) video.
    case video(URL)
    /// A 360° equirectangular video.
    case video360(URL)
    /// A folder of 360° equirectangular photos.
    case photos360(folder: URL)
    /// Stereo footage: one file (SBS / TB / MV-HEVC / dual-stream) or two files (left, right).
    case stereoVideo(left: URL, right: URL?)
    /// A `.splatcapture` bundle (directory or zip) from SplatWork Capture / Steam Frame.
    case captureBundle(URL)
    /// A COLMAP project (`sparse/0` + `images/`), text or binary model.
    case colmap(project: URL, images: URL?)
    /// A Nerfstudio / Instant-NGP `transforms.json` (or its directory).
    case nerfstudio(transforms: URL)

    var displayName: String {
        switch self {
        case .photos(let u), .photos360(let u), .video(let u), .video360(let u), .captureBundle(let u), .nerfstudio(let u): return u.lastPathComponent
        case .photoFiles(let us): return "\(us.count) photos"
        case .stereoVideo(let l, _): return l.lastPathComponent
        case .colmap(let p, _): return p.lastPathComponent
        }
    }

    /// Whether this source already carries camera poses.
    var hasPoses: Bool {
        switch self { case .captureBundle, .colmap, .nerfstudio: return true; default: return false }
    }

    /// Guesses the source kind of a dropped file/folder.
    static func guess(_ url: URL) -> SourceDescription? {
        let fm = FileManager.default
        let ext = url.pathExtension.lowercased()
        if ext == "splatcapture" || fm.fileExists(atPath: url.appendingPathComponent("manifest.json").path) { return .captureBundle(url) }
        if ext == "zip" { return .captureBundle(url) }
        if url.lastPathComponent == "transforms.json" || fm.fileExists(atPath: url.appendingPathComponent("transforms.json").path) { return .nerfstudio(transforms: url) }
        if (try? COLMAPImporter.locate(url)) != nil { return .colmap(project: url, images: nil) }
        if VideoFrameExtractor.videoExtensions.contains(ext) {
            let n = url.lastPathComponent.lowercased()
            if n.contains("360") || n.contains("equirect") || ext == "insv" { return .video360(url) }
            if n.contains("sbs") || n.contains("stereo") || n.contains("spatial") || n.contains("arcturus") { return .stereoVideo(left: url, right: nil) }
            return .video(url)
        }
        var isDir: ObjCBool = false
        if fm.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue { return .photos(folder: url) }
        if ImageUtil.isImageFile(url) { return .photoFiles([url]) }
        return nil
    }
}

/// How camera poses are obtained.
enum PoseSolverChoice: Codable, Hashable, Sendable {
    /// Object Capture for photos/video when supported, splatprep otherwise; existing poses for captures/COLMAP/Nerfstudio.
    case automatic
    /// Apple Object Capture (`PhotogrammetrySession`, local only).
    case objectCapture
    /// splatprep learned SfM (local Python or remote worker per `compute`).
    case splatprep(featureModelId: String, matching: String, mapper: String)
    /// Use the poses stored in the source (capture bundle ARKit/OpenXR poses, COLMAP, Nerfstudio).
    case fromCapture
}

/// All dataset-preparation options. Every field has a default; decoding tolerates missing keys.
struct PrepOptions: Codable, Hashable, Sendable {
    /// Photo-folder ingest (blur / duplicate rejection, …).
    var photos: PhotoIngestOptions = .init()
    /// Video frame extraction (fps / frame count, sharpest-in-window, cuts, trimming).
    var video: VideoExtractionOptions = .init()
    /// 360° → pinhole views (count, FOV, pitch rings, nadir/zenith handling).
    var equirect: Equirect360Options = .init()
    /// Stereo layout, rig baseline / calibration, stereo depth.
    var stereo: StereoOptions = .init()
    /// Capture-bundle import (depth, confidence, point cloud, tracking filters).
    var capture: CaptureImportOptions = .init()
    /// COLMAP / Nerfstudio import (undistortion, resizing).
    var importing: ImportOptions = .init()
    /// Pose estimation method.
    var poseSolver: PoseSolverChoice = .automatic
    /// Object Capture feature sensitivity (high = slower, more images registered in low texture).
    var objectCaptureHighSensitivity: Bool = false
    /// splatprep camera model: "pinhole" keeps our images; "opencv"/"fisheye"/"auto" use splatprep's undistorted copies.
    var sfmCameraModel: String = "pinhole"
    /// Per-image masks.
    var masking: MaskingOptions = .init()
    /// Mark every Nth camera as holdout for evaluation (nil = none). Overrides per-source settings.
    var holdoutInterval: Int? = nil
    /// Longest image side for all sources (nil = per-source settings / full resolution).
    var maxImageSize: Int? = 1600
    /// Where processing runs.
    var compute: ComputePlacement = .thisMac
    /// Mask parts of the frame that never change while the camera moves (lens rim, timestamp, propellers, rig).
    var maskStaticRegions = false
    /// Worker used for remote / both placement (WorkerBrowser id).
    var remoteWorkerId: String? = nil
    /// With a worker in use, split videos (incl. 360°) on it: the video is uploaded once and the views come back.
    var videoOnWorker: Bool = true
    /// Captures with their own poses (iPhone/ARKit): polish those poses with a quick SfM refinement
    /// (triangulate + pose-only bundle adjustment, keeping capture intrinsics and scale). +2.1 dB held-out on a
    /// 649-frame bedroom scan, ≈ 4 min on an M4 Pro.
    var refineCapturePoses: Bool = true
    /// Use the known stereo baseline to rescale the solved scene to metres.
    var metricScaleFromStereo: Bool = true
    /// After solving, re-derive right-eye poses from the left eye and the rig (stereo), and 360 view poses
    /// from the per-frame rig (fills unregistered views).
    var enforceRigConstraints: Bool = true
    /// Keep intermediate files (frames.json, splatprep work dirs).
    var keepIntermediates: Bool = false

    init() {}
    init(from decoder: Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        photos = c(.photos, photos); video = c(.video, video); equirect = c(.equirect, equirect); stereo = c(.stereo, stereo)
        capture = c(.capture, capture); importing = c(.importing, importing); poseSolver = c(.poseSolver, poseSolver)
        objectCaptureHighSensitivity = c(.objectCaptureHighSensitivity, objectCaptureHighSensitivity)
        sfmCameraModel = c(.sfmCameraModel, sfmCameraModel); masking = c(.masking, masking)
        holdoutInterval = c(.holdoutInterval, holdoutInterval); maxImageSize = c(.maxImageSize, maxImageSize)
        compute = c(.compute, compute); remoteWorkerId = c(.remoteWorkerId, remoteWorkerId); videoOnWorker = c(.videoOnWorker, videoOnWorker)
        refineCapturePoses = c(.refineCapturePoses, refineCapturePoses)
        metricScaleFromStereo = c(.metricScaleFromStereo, metricScaleFromStereo)
        enforceRigConstraints = c(.enforceRigConstraints, enforceRigConstraints); keepIntermediates = c(.keepIntermediates, keepIntermediates)
    }

    /// Applies the global `maxImageSize` / `holdoutInterval` to the per-source options.
    var resolved: PrepOptions {
        var o = self
        if let m = maxImageSize {
            o.photos.maxImageSize = m; o.video.maxImageSize = m; o.importing.maxImageSize = m; o.capture.images.maxImageSize = m
            o.equirect.outputSize = min(o.equirect.outputSize, m)
        }
        if let h = holdoutInterval { o.importing.holdoutInterval = h; o.capture.images.holdoutInterval = h }
        return o
    }
}

/// Single entry point that turns any `SourceDescription` into a `PreparedDataset`:
/// ingest → masks → poses → rig constraints / metric scale → `dataset.json`.
enum DatasetBuilder {
    /// - Parameters:
    ///   - dir: output dataset directory (created; images are written into `dir/images`).
    ///   - worker: client used when `options.compute` puts SfM on a remote worker.
    /// Matcher for refining capture poses: COLMAP SIFT when the COLMAP tool is installed here (fastest on a Mac,
    /// multi-threaded matching and verification), SIFT + LightGlue otherwise and on a Linux worker.
    static func refineFeatures(_ o: PrepOptions, worker: WorkerClient?) -> String {
        if o.compute != .thisMac, worker != nil { return "sift_lightglue" }
        let fm = FileManager.default
        return ["/opt/homebrew/bin/colmap", "/usr/local/bin/colmap"].contains { fm.isExecutableFile(atPath: $0) } ? "colmap_sift" : "sift_lightglue"
    }

    /// Splits a video on the worker when one is in use and supports it; nil = do it on this Mac.
    /// A worker failure (old version, network) falls back to this Mac with a note instead of failing the build.
    static func framesOnWorker(_ u: URL, equirect: Equirect360Options?, _ o: PrepOptions, worker: WorkerClient?,
                               into dir: URL, progress: PipelineProgressHandler?) async throws -> FrameSet? {
        guard o.videoOnWorker, o.compute != .thisMac, let worker else { return nil }
        let stage = equirect == nil ? "Extracting frames" : "Splitting 360° video"
        guard let info = try? await worker.info(timeout: 10), info.canExtractFrames else {
            progress?(PipelineProgress(stage: stage, fraction: -1, message: "the worker can't split videos yet (update it and install the processing models) — doing it on this Mac"))
            return nil
        }
        do {
            return try await RemoteFrameExtraction.extract(video: u, options: o.video, equirect: equirect, into: dir, worker: worker, progress: progress)
        } catch let e as CancellationError {
            throw e
        } catch {
            if Task.isCancelled { throw CancellationError() }
            for sub in ["images", "masks"] { try? FileManager.default.removeItem(at: dir.appendingPathComponent(sub)) }
            progress?(PipelineProgress(stage: stage, fraction: -1, message: "worker couldn't split the video (\(error.localizedDescription)) — doing it on this Mac"))
            return nil
        }
    }

    static func build(source: SourceDescription, options: PrepOptions = .init(), into dir: URL,
                      worker: WorkerClient? = nil, progress: PipelineProgressHandler? = nil) async throws -> PreparedDataset {
        let o = options.resolved
        let fm = FileManager.default
        try fm.ensureDirectory(dir)
        let ingestP = PipelineProgress.scaled(progress, 0...0.35)
        let maskP = PipelineProgress.scaled(progress, 0.35...0.45)
        let poseP = PipelineProgress.scaled(progress, 0.45...0.97)

        // 1. Sources that already carry poses.
        switch source {
        case .captureBundle(let u):
            var ds = try await CaptureImporter.importBundle(u, options: o.capture, into: dir, progress: PipelineProgress.scaled(progress, 0...0.6))
            ds = try await maskDataset(ds, o, worker: worker, progress: PipelineProgress.scaled(progress, 0.6...0.7))
            let refine: Bool = {
                switch o.poseSolver {
                case .splatprep: return true
                case .objectCapture: return false
                default: return o.refineCapturePoses
                }
            }()
            if refine {
                // Refine with SfM, using the capture poses (and per-frame intrinsics) as priors: splatprep then
                // triangulates with them and adjusts only the poses (prior_refine.py) instead of solving.
                try PriorsFile.write(dataset: ds, to: dir.appendingPathComponent("priors.json"))
                var frames = frameSet(from: ds)
                var ro = o
                if case .splatprep = o.poseSolver {} else {
                    ro.poseSolver = .splatprep(featureModelId: refineFeatures(o, worker: worker), matching: "auto", mapper: "auto")
                }
                let sol: PoseSolution
                do {
                    sol = try await solvePoses(&frames, ro, worker: worker, priors: "priors.json", progress: PipelineProgress.scaled(progress, 0.7...0.97))
                } catch let e as CancellationError {
                    throw e
                } catch {
                    if Task.isCancelled { throw CancellationError() }
                    // Refinement is a bonus: without the Models pack / an offline worker, keep the capture poses.
                    progress?(PipelineProgress(stage: "Camera solving", fraction: -1,
                        message: "couldn't refine the capture poses (\(error.localizedDescription)) — using them as recorded"))
                    return finish(ds, progress)
                }
                // SfM that places only a fraction of the photos (bad masks, textureless rooms) must never
                // replace the capture: keep the iPhone's own ARKit poses, which are complete, and say so.
                if Double(sol.poses.count) >= 0.5 * Double(frames.frames.count) {
                    let stem = { (p: String) in ((p as NSString).lastPathComponent as NSString).deletingPathExtension }
                    let captured = Dictionary(ds.manifest.cameras.map { (stem($0.image), $0) }, uniquingKeysWith: { a, _ in a })
                    let gravity = ds.manifest.gravityUp
                    ds = try PoseAssembly.makeDataset(frames: frames, solution: sol, holdoutInterval: o.capture.images.holdoutInterval)
                    // The refined poses are Sim3-aligned back onto the capture's frame (rotation ≈ 0.5°), so the
                    // phone's measured gravity still holds — without it, levelling falls back to guessing from how
                    // the camera was held, which tilts handheld phone scans badly.
                    if ds.manifest.gravityUp == nil { ds.manifest.gravityUp = gravity }
                    // Per-frame capture data the pose solution doesn't carry: exposure (lets training see that
                    // exposure was locked and skip per-photo colour correction), time and source group.
                    for i in ds.manifest.cameras.indices {
                        guard let c = captured[stem(ds.manifest.cameras[i].image)] else { continue }
                        if ds.manifest.cameras[i].exposure == nil { ds.manifest.cameras[i].exposure = c.exposure }
                        if ds.manifest.cameras[i].time == nil { ds.manifest.cameras[i].time = c.time }
                        if ds.manifest.cameras[i].group == nil { ds.manifest.cameras[i].group = c.group }
                    }
                    try ds.save()
                } else {
                    progress?(PipelineProgress(stage: "Camera solving", fraction: -1,
                        message: "SfM placed only \(sol.poses.count) of \(frames.frames.count) photos — keeping the capture's own (ARKit) camera poses"))
                }
            }
            return finish(ds, progress)
        case .colmap(let p, let images):
            var ds = try await COLMAPImporter.importProject(at: p, imagesDir: images, options: o.importing, into: dir, progress: PipelineProgress.scaled(progress, 0...0.85))
            ds = try await maskDataset(ds, o, worker: worker, progress: PipelineProgress.scaled(progress, 0.85...0.97))
            return finish(ds, progress)
        case .nerfstudio(let t):
            var ds = try await NerfstudioImporter.importDataset(transforms: t, options: o.importing, into: dir, progress: PipelineProgress.scaled(progress, 0...0.85))
            ds = try await maskDataset(ds, o, worker: worker, progress: PipelineProgress.scaled(progress, 0.85...0.97))
            return finish(ds, progress)
        default: break
        }

        // 2. Ingest frames.
        var frames: FrameSet
        switch source {
        case .photos(let folder):
            frames = try await PhotoIngest.ingest(folder: folder, options: o.photos, into: dir, progress: ingestP)
        case .photoFiles(let files):
            frames = try await PhotoIngest.ingest(files: files, options: o.photos, into: dir, progress: ingestP)
        case .video(let u):
            let stamp = FrameSourceStamp.make(video: u, video: o.video, equirect: nil)
            if let f = FrameSourceStamp.reusable(dir, stamp: stamp, worker: worker) {
                ingestP(PipelineProgress(stage: "Extracting frames", fraction: 1, message: "reusing the \(f.frames.count) frames from the last run"))
                frames = f
            } else {
                if let f = try await framesOnWorker(u, equirect: nil, o, worker: worker, into: dir, progress: ingestP) { frames = f }
                else { frames = try await VideoFrameExtractor.extract(video: u, options: o.video, into: dir, progress: ingestP) }
                stamp?.save(dir)
            }
        case .video360(let u):
            let stamp = FrameSourceStamp.make(video: u, video: o.video, equirect: o.equirect)
            if let f = FrameSourceStamp.reusable(dir, stamp: stamp, worker: worker) {
                ingestP(PipelineProgress(stage: "Splitting 360° video", fraction: 1, message: "reusing the \(f.frames.count) views from the last run"))
                frames = f
            } else {
                if let f = try await framesOnWorker(u, equirect: o.equirect, o, worker: worker, into: dir, progress: ingestP) { frames = f }
                else { frames = try await Equirect360.extract(video: u, video: o.video, options: o.equirect, into: dir, progress: ingestP) }
                stamp?.save(dir)
            }
        case .photos360(let folder):
            frames = try await Equirect360.extract(photos: PhotoIngest.listImages(in: folder, recursive: false), options: o.equirect,
                                                   encoding: o.photos.encoding, into: dir, progress: ingestP)
        case .stereoVideo(let l, let r):
            frames = try await StereoVideoIngest.ingest(left: l, right: r, video: o.video, stereo: o.stereo, into: dir, progress: ingestP)
            if !o.stereo.trainOnBothEyes { frames.frames.removeAll { $0.eye == "right" } ; if frames.frames.isEmpty { throw PipelineError.noInput("no left frames") } }
        default:
            fatalError("handled above")
        }

        // 3. Masks.
        if o.maskStaticRegions {
            try await FrameMasker.waitForFrames(frames.root, progress: maskP)
            let reps = try await StaticRegions.apply(to: &frames, progress: maskP)
            for r in reps where r.staticFraction > 0 {
                maskP(PipelineProgress(stage: "Finding fixed parts of the frame", fraction: -1, message: String(format: "%@: %.1f %% of the frame %@",
                    r.camera, r.staticFraction * 100, r.applied ? "masked" : "looks fixed — not masked (the camera barely moves)")))
            }
        }
        if o.masking.model != .none {
            try await FrameMasker.apply(o.masking, to: &frames, runner: SplatPrepRunner(),
                                        worker: o.compute != .thisMac ? worker : nil, progress: maskP)
        }
        if o.masking.sky {   // stacked on top of the masks above: a separate layer for the skybox
            try await FrameMasker.applySky(to: &frames, runner: SplatPrepRunner(),
                                           worker: o.compute != .thisMac ? worker : nil, progress: maskP)
        }

        // 4. Poses (+ rig constraints / metric scale).
        let sequential: Bool = { switch source { case .video, .video360, .stereoVideo: return true; default: return false } }()
        let sol = try await solvePoses(&frames, o, worker: worker, priors: nil, sequential: sequential, progress: poseP)
        if frames.frames.count >= 10 && Double(sol.poses.count) < 0.3 * Double(frames.frames.count) {
            throw PipelineError.processFailed("Camera solving placed only \(sol.poses.count) of \(frames.frames.count) photos. "
                + (o.masking.model != .none ? "Masks may be hiding most of each photo — try without masks. " : "")
                + "Try another camera solver, or capture with more overlap between photos.")
        }
        var ds = try PoseAssembly.makeDataset(frames: frames, solution: sol, holdoutInterval: o.holdoutInterval)
        if !o.keepIntermediates {
            try? fm.removeItem(at: dir.appendingPathComponent(".oc-input"))
            try? fm.removeItem(at: dir.appendingPathComponent("request.json"))
        }
        ds.recomputeSceneBounds()
        try ds.save()
        return finish(ds, progress)
    }

    private static func finish(_ ds: PreparedDataset, _ progress: PipelineProgressHandler?) -> PreparedDataset {
        progress?(PipelineProgress(stage: "Done", fraction: 1, message: "\(ds.manifest.cameras.count) cameras"))
        return ds
    }

    /// Runs the selected pose solver and applies rig constraints.
    static func solvePoses(_ frames: inout FrameSet, _ o: PrepOptions, worker: WorkerClient?, priors: String?, sequential: Bool = false,
                           progress: PipelineProgressHandler?) async throws -> PoseSolution {
        var choice = o.poseSolver
        var cameraModel = o.sfmCameraModel
        let fisheye = !frames.isMultiViewRig && frames.looksCircularFisheye
        if fisheye && (cameraModel == "pinhole" || cameraModel == "auto") {
            // Circular fisheye: solve with a fisheye model; splatprep then trains on undistorted copies.
            cameraModel = "fisheye"
            progress?(PipelineProgress(stage: "Camera solving", fraction: -1,
                                       message: "fisheye footage: solving with a fisheye camera model and undistorting"))
        }
        if choice == .automatic || choice == .fromCapture {
            if frames.isMultiViewRig {
                // 360° footage: Object Capture is built for photographing one object and placed only 1,041 of
                // 9,824 views of a 360 walk-through. Learned SfM (on the worker when one is in use) handles it;
                // retrieval matching, because a frame's views are interleaved in file order.
                choice = .splatprep(featureModelId: "superpoint_lightglue", matching: "auto", mapper: "auto")
            } else if fisheye {
                // Object Capture returns pinhole cameras for fisheye frames (the distortion ends up in the scene).
                choice = .splatprep(featureModelId: "superpoint_lightglue", matching: sequential ? "sequential" : "auto", mapper: "auto")
            } else {
                choice = ObjectCapturePoseSolver.isSupported ? .objectCapture
                    : .splatprep(featureModelId: "superpoint_lightglue", matching: sequential ? "sequential" : "auto", mapper: "auto")
            }
        }
        var sol: PoseSolution
        switch choice {
        case .splatprep(let feat, let matching, let mapper):
            var sp = SplatPrepPoseSolver(features: feat, matching: matching, mapper: mapper, cameraModel: cameraModel)
            sp.priorPoses = priors
            sp.resizeMax = o.maxImageSize
            if o.compute != .thisMac, let worker {
                sol = try await RemoteSplatPrepPoseSolver(client: worker, local: sp).solve(frames, progress: progress)
            } else {
                try await FrameMasker.waitForFrames(frames.root, progress: progress)
                sol = try await sp.solve(frames, progress: progress)
            }
        default:
            try await FrameMasker.waitForFrames(frames.root, progress: progress)
            let oc = ObjectCapturePoseSolver(sensitivity: o.objectCaptureHighSensitivity ? .high : .normal, sequentialOrdering: sequential)
            sol = try await oc.solve(frames, progress: progress)
        }
        if o.enforceRigConstraints && frames.isMultiViewRig {
            RigConstraints.enforceMultiViewRig(&sol, frames: frames)
        }
        if let rig = frames.rig, let b = rig.baseline {
            if o.metricScaleFromStereo, let s = RigConstraints.stereoScale(sol, frames: frames, baseline: b) {
                RigConstraints.applyScale(s, to: &sol)
                sol.notes.append(String(format: "metric scale from %.1f mm stereo baseline: ×%.4f", b * 1000, s))
                sol.stats["metricScale"] = .number(Double(s))
            }
            if o.enforceRigConstraints { RigConstraints.enforceStereoRig(&sol, frames: frames, rig: rig) }
        }
        return sol
    }

    /// Applies masking to an already imported dataset.
    static func maskDataset(_ ds: PreparedDataset, _ o: PrepOptions, worker: WorkerClient? = nil,
                            progress: PipelineProgressHandler?) async throws -> PreparedDataset {
        guard o.masking.model != .none || o.masking.sky else { return ds }
        var frames = frameSet(from: ds)
        if o.masking.model != .none {
            try await FrameMasker.apply(o.masking, to: &frames, worker: o.compute != .thisMac ? worker : nil, progress: progress)
        }
        if o.masking.sky {
            try await FrameMasker.applySky(to: &frames, worker: o.compute != .thisMac ? worker : nil, progress: progress)
        }
        var out = ds
        let masks = Dictionary(frames.frames.map { ($0.image, ($0.mask, $0.sky)) }, uniquingKeysWith: { a, _ in a })
        for i in out.manifest.cameras.indices {
            if let m = masks[out.manifest.cameras[i].image] {
                if let mk = m.0 { out.manifest.cameras[i].mask = mk }
                if let sk = m.1 { out.manifest.cameras[i].sky = sk }
            }
        }
        try out.save()
        try? FileManager.default.removeItem(at: ds.root.appendingPathComponent(FrameSet.fileName))
        return out
    }

    /// A frame view of a dataset's cameras (for masking / re-solving).
    static func frameSet(from ds: PreparedDataset) -> FrameSet {
        FrameSet(root: ds.root, frames: ds.manifest.cameras.map { c in
            var f = IngestedFrame(image: c.image, width: c.width, height: c.height)
            f.focalPx = c.fx; f.cx = c.cx; f.cy = c.cy; f.mask = c.mask; f.sky = c.sky
            f.depth = c.depth; f.depthWidth = c.depthWidth; f.depthHeight = c.depthHeight
            return f
        })
    }
}
