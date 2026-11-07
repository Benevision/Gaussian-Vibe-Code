import Foundation
import RealityKit
import simd

/// Camera poses + sparse point cloud from Apple Object Capture (`PhotogrammetrySession` `.poses` and
/// `.pointCloud` requests, macOS 14+). On macOS 26+ the per-image intrinsics reported by the session
/// (`Pose.intrinsics`) are used; otherwise EXIF / known focal lengths of the frames (or a 53° default).
///
/// RealityKit poses use camera looks −Z, +Y up; they are converted to OpenCV c2w via `· diag(1,−1,−1,1)`.
struct ObjectCapturePoseSolver: PoseSolver {
    /// Sensitivity for low-texture scenes (`.high` is slower but registers more images).
    enum Sensitivity: String, Codable, Sendable { case normal, high }

    var sensitivity: Sensitivity = .normal
    /// Frames are in capture order (video) — enables faster sequential matching.
    var sequentialOrdering: Bool = false
    /// Isolate a central object (turntable captures). Off for scenes.
    var objectMasking: Bool = false

    static var isSupported: Bool { PhotogrammetrySession.isSupported }

    func solve(_ frames: FrameSet, progress: PipelineProgressHandler?) async throws -> PoseSolution {
        let stage = "Object Capture poses"
        guard PhotogrammetrySession.isSupported else { throw PipelineError.unsupported("Object Capture is not supported on this Mac") }
        let fm = FileManager.default
        // PhotogrammetrySession reads a flat folder: hard-link every frame image into one.
        let input = frames.root.appendingPathComponent(".oc-input")
        try? fm.removeItem(at: input)
        try fm.ensureDirectory(input)
        defer { try? fm.removeItem(at: input) }
        var byName: [String: IngestedFrame] = [:]
        for (i, f) in frames.frames.enumerated() {
            let name = String(format: "%06d_", i) + (f.image as NSString).lastPathComponent
            let dst = input.appendingPathComponent(name)
            let src = frames.url(f.image)
            do { try fm.linkItem(at: src, to: dst) } catch { try fm.copyItem(at: src, to: dst) }
            byName[name] = f
        }

        var config = PhotogrammetrySession.Configuration()
        config.featureSensitivity = sensitivity == .high ? .high : .normal
        config.sampleOrdering = sequentialOrdering ? .sequential : .unordered
        config.isObjectMaskingEnabled = objectMasking
        let session = try PhotogrammetrySession(input: input, configuration: config)
        progress?(PipelineProgress(stage: stage, fraction: 0, message: "\(frames.frames.count) images"))

        var solution = PoseSolution()
        var posesDone = false, pointsDone = false
        // The two requests report progress independently (and sometimes interleaved); combine them
        // monotonically so the bar never jumps backwards, and keep Object Capture's own ETA.
        var posesFrac = 0.0, pointsFrac = 0.0, phase = "Aligning images", eta: Double? = nil
        func report() {
            let f = min(1, 0.7 * posesFrac + 0.3 * pointsFrac)
            progress?(PipelineProgress(stage: stage, fraction: f, message: phase, stageETA: eta))
        }
        try await withTaskCancellationHandler {
            try session.process(requests: [.poses, .pointCloud])
            for try await output in session.outputs {
                switch output {
                case .requestProgress(let req, let fraction):
                    if case .poses = req { posesFrac = max(posesFrac, fraction) } else { pointsFrac = max(pointsFrac, fraction) }
                    report()
                case .requestProgressInfo(_, let info):
                    if let st = info.processingStage {
                        switch st {
                        case .preProcessing: phase = "Analysing images"
                        case .imageAlignment: phase = "Aligning images"
                        case .pointCloudGeneration: phase = "Building point cloud"
                        default: phase = "\(st)"
                        }
                    }
                    if let r = info.estimatedRemainingTime { eta = r }
                    report()
                case .requestComplete(_, let result):
                    switch result {
                    case .poses(let poses):
                        posesDone = true
                        for (id, pose) in poses.posesBySample {
                            guard let url = poses.urlsBySample[id], let f = byName[url.lastPathComponent] else { continue }
                            solution.poses[f.image] = Self.convert(pose, frame: f)
                        }
                    case .pointCloud(let pc):
                        pointsDone = true
                        var out = PointCloud()
                        out.points.reserveCapacity(pc.points.count); out.colors.reserveCapacity(pc.points.count)
                        for p in pc.points {
                            out.points.append(p.position)
                            out.colors.append(SIMD3(p.color.x, p.color.y, p.color.z))
                        }
                        solution.points = out
                    default: break
                    }
                case .requestError(let req, let error):
                    if case .poses = req { throw PipelineError.poseSolveFailed(error.localizedDescription) }
                    solution.notes.append("point cloud failed: \(error.localizedDescription)")
                case .invalidSample(let id, let reason):
                    solution.notes.append("sample \(id) invalid: \(reason)")
                case .skippedSample(let id):
                    solution.notes.append("sample \(id) skipped")
                case .processingCancelled:
                    throw CancellationError()
                case .processingComplete:
                    break
                default:
                    break
                }
                if case .processingComplete = output { break }
            }
        } onCancel: {
            session.cancel()
        }
        guard posesDone, !solution.poses.isEmpty else { throw PipelineError.poseSolveFailed("Object Capture registered no images") }
        if !pointsDone { solution.notes.append("no point cloud returned") }
        let missing = frames.frames.count - solution.poses.count
        if missing > 0 { solution.notes.append("\(missing) images not registered") }
        solution.stats["numRegistered"] = .number(Double(solution.poses.count))
        solution.stats["numPoints3D"] = .number(Double(solution.points?.count ?? 0))
        progress?(PipelineProgress(stage: stage, fraction: 1, message: "\(solution.poses.count)/\(frames.frames.count) registered"))
        return solution
    }

    /// RealityKit (−Z forward, +Y up) → OpenCV c2w; intrinsics from the session (macOS 26+) or the frame.
    static func convert(_ pose: PhotogrammetrySession.Pose, frame f: IngestedFrame) -> SolvedPose {
        let rk = simd_float4x4(rotation: simd_float3x3(pose.rotation), translation: pose.translation)
        var p = SolvedPose(c2w: rk * .flipYZ)
        if #available(macOS 26.0, *), let K = pose.intrinsics {
            // simd 3x3 is column-major: K[0][0]=fx, K[1][1]=fy, K[2][0]=cx, K[2][1]=cy.
            p.fx = K[0][0]; p.fy = K[1][1]; p.cx = K[2][0]; p.cy = K[2][1]
            // Intrinsics refer to the image as read by the session; rescale defensively if they look like
            // they were reported for a different resolution (principal point far outside the image).
            if let cx = p.cx, cx > Float(f.width) * 1.5 || cx < 0 {
                let s = Float(f.width) / (2 * cx)
                p.fx = p.fx.map { $0 * s }; p.fy = p.fy.map { $0 * s }; p.cx = Float(f.width) / 2; p.cy = p.cy.map { $0 * s }
            }
        } else if let fp = f.focalPx {
            p.fx = fp; p.fy = fp
        }
        return p
    }
}
