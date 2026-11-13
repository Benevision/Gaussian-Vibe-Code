import Foundation
import Observation
import simd

// MARK: - Request

/// `ProcessingRequest` of PROTOCOL.md §6 (sent to `python -m splatprep run` or worker `/v1/process`).
struct ProcessingRequest: Codable, Hashable, Sendable {
    enum Task: String, Codable, Sendable { case sfm, masks, benchmark, frames }
    var task: Task
    var imagesDir: String = "images"
    var outputDir: String = "out"
    /// Feature/matcher id: colmap_sift, sift_lightglue, superpoint_lightglue, disk_lightglue, aliked_lightglue, loftr, mast3r, vggt.
    var features: String? = "superpoint_lightglue"
    /// sequential | exhaustive | retrieval | auto
    var matching: String? = "auto"
    var retrievalTopK: Int? = nil
    var sequentialOverlap: Int? = nil
    /// glomap | colmap | auto
    var mapper: String? = "auto"
    /// auto | pinhole | opencv | fisheye
    var cameraModel: String? = "auto"
    var singleCamera: Bool? = nil
    var maxKeypoints: Int? = nil
    var resizeMax: Int? = nil
    /// Relative path of a priors file (`PriorsFile`), e.g. ARKit poses.
    var priorPoses: String? = nil
    /// Keep-masks used to drop keypoints on masked-out pixels (splatprep extension).
    var masksDir: String? = nil
    /// Sky masks (255 = sky): no keypoints on the sky when solving cameras.
    var skyMasksDir: String? = nil
    /// sam2_auto | sky | person (task = masks)
    var maskModel: String? = nil
    var maskPrompt: MaskPrompt? = nil
    /// Prompts on any frame, tracked forward and backward with SAM 2:
    /// `[{"image": "000123.jpg", "points": [[x, y, label]], "boxes": [[x0, y0, x1, y1]], "objectId": 1}]`.
    var maskPrompts: [JSONValue]? = nil
    /// Object classes for the `dynamic` mask model (open-vocabulary detection).
    var maskClasses: [String]? = nil
    /// Detection confidence for the `dynamic` model (OWLv2). 0.35 avoids most static false positives.
    var maskThreshold: Double? = nil
    /// Overwrite existing masks instead of AND-combining with them.
    var replaceMasks: Bool? = nil
    var invertMask: Bool? = nil
    /// auto | cuda | mps | cpu (splatprep extension)
    var device: String? = nil
    /// Rig description for multi-camera captures (SplatWork extension; ignored by toolkits that don't support it).
    var rig: RigHint? = nil
    /// task = frames: the video (relative to the work dir), its extraction options and, for 360° footage, the views.
    var video: String? = nil
    var videoOptions: VideoExtractionOptions? = nil
    var equirect: Equirect360Options? = nil
    /// task = frames: old home video restoration and clips instead of plain extraction (splatprep `vhs.py`).
    var vhs: VHS? = nil
    /// task = sfm: {image basename: group}; the images of a group share one camera (home-video clips / zoom settings).
    var cameraGroups: [String: Int]? = nil

    struct VHS: Codable, Hashable, Sendable {
        var analyseOnly: Bool? = nil
        var clips: [Int]? = nil
        var maxHeight: Int? = nil
        var denoise: Bool? = nil
        var colour: Bool? = nil
        var repairDropouts: Bool? = nil
        var totalFrames: Int? = nil
    }

    struct MaskPrompt: Codable, Hashable, Sendable {
        var points: [[Float]]? = nil   // [x, y, label]
        var boxes: [[Float]]? = nil    // [x0, y0, x1, y1]
    }

    /// Rig constraint hint: stereo pairs with a fixed relative pose, or 360° view groups with known rotations.
    struct RigHint: Codable, Hashable, Sendable {
        var kind: String                       // "stereo" | "multiview"
        var baseline: Float? = nil
        /// Right-to-left (stereo) relative pose, row-major 4x4, OpenCV.
        var rightToLeft: [[Float]]? = nil
        /// Stereo: [[left image, right image]] basenames.
        var pairs: [[String]]? = nil
        /// Multiview: groups of {image basename: camera-to-rig rotation (row-major 3x3)}.
        var groups: [[String: [[Float]]]]? = nil
        /// Multiview: the views' exact pinhole intrinsics [fx, fy, cx, cy] (all views share them). splatprep ≥
        /// 2026-11-08 then registers whole frames with these fixed instead of guessing a focal length.
        var intrinsics: [Float]? = nil
    }

    init(task: Task) { self.task = task }
}

/// A line of `--progress-jsonl` output.
enum SplatPrepEvent: Sendable {
    case progress(stage: String, fraction: Double, message: String)
    case done(stats: [String: JSONValue])
    case error(String)

    static func parse(_ line: Substring) -> SplatPrepEvent? {
        guard let data = line.data(using: .utf8),
              let obj = try? JSONDecoder().decode([String: JSONValue].self, from: data) else { return nil }
        if let e = obj["error"]?.string { return .error(e) }
        if obj["done"]?.bool == true { return .done(stats: obj["stats"]?.object ?? [:]) }
        return .progress(stage: obj["stage"]?.string ?? "", fraction: obj["progress"]?.double ?? -1, message: obj["message"]?.string ?? "")
    }
}

// MARK: - Runner

/// Runs `python -m splatprep run --request request.json --progress-jsonl - --workdir <dir>` and streams progress.
struct SplatPrepRunner: Sendable {
    /// Python interpreter (default: the managed venv).
    var python: URL = PythonEnvironment.defaultPython
    /// Directory containing the `splatprep` package (added to PYTHONPATH so an editable install isn't required).
    var toolsDirectory: URL? = PythonEnvironment.locateToolsDirectory()
    /// Extra environment (e.g. `PYTORCH_ENABLE_MPS_FALLBACK=1`).
    var environment: [String: String] = ["PYTORCH_ENABLE_MPS_FALLBACK": "1"]

    /// Writes `request.json` into `workDir` and runs it. Relative paths in the request resolve against `workDir`.
    /// - Parameter log: receives stderr lines (Python logging, tracebacks).
    @discardableResult
    func run(_ request: ProcessingRequest, workDir: URL, progress: PipelineProgressHandler? = nil,
             log: (@Sendable (String) -> Void)? = nil) async throws -> [String: JSONValue] {
        let reqURL = workDir.appendingPathComponent("request.json")
        let enc = JSONEncoder(); enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        try enc.encode(request).write(to: reqURL, options: .atomic)
        guard FileManager.default.isExecutableFile(atPath: python.path) else {
            throw PipelineError.processFailed("Python environment not installed (\(python.path))")
        }
        var env = ProcessInfo.processInfo.environment
        for (k, v) in environment { env[k] = v }
        if let t = toolsDirectory { env["PYTHONPATH"] = t.path + (env["PYTHONPATH"].map { ":" + $0 } ?? "") }
        env["PYTHONUNBUFFERED"] = "1"
        let args = ["-m", "splatprep", "run", "--request", reqURL.path, "--progress-jsonl", "-", "--workdir", workDir.path]
        let result = ResultBox()
        let stderrTail = LineTail()
        let status = try await LineStreamingProcess.run(executable: python, arguments: args, cwd: workDir, environment: env,
            stdout: { line in
                guard let ev = SplatPrepEvent.parse(line) else { return }
                switch ev {
                case let .progress(stage, f, m): progress?(PipelineProgress(stage: stage.isEmpty ? "splatprep" : stage, fraction: f, message: m))
                case let .done(stats): result.set(stats: stats)
                case let .error(e): result.set(error: e)
                }
            },
            stderr: { line in stderrTail.append(String(line)); log?(String(line)) })
        if let e = result.error { throw PipelineError.processFailed("splatprep: \(e)") }
        guard status == 0, let stats = result.stats else {
            throw PipelineError.processFailed("splatprep exited with \(status): \(stderrTail.text)")
        }
        return stats
    }

    final class ResultBox: @unchecked Sendable {
        private let lock = NSLock()
        private(set) var stats: [String: JSONValue]?
        private(set) var error: String?
        func set(stats s: [String: JSONValue]) { lock.lock(); stats = s; lock.unlock() }
        func set(error e: String) { lock.lock(); error = e; lock.unlock() }
    }
}

/// Keeps the last N lines (for error messages).
final class LineTail: @unchecked Sendable {
    private let lock = NSLock()
    private var lines: [String] = []
    let limit: Int
    init(limit: Int = 30) { self.limit = limit }
    func append(_ l: String) { lock.lock(); lines.append(l); if lines.count > limit { lines.removeFirst(lines.count - limit) }; lock.unlock() }
    var text: String { lock.lock(); defer { lock.unlock() }; return lines.joined(separator: "\n") }
}

/// Runs a process delivering stdout / stderr line by line; terminates the process on task cancellation.
enum LineStreamingProcess {
    static func run(executable: URL, arguments: [String], cwd: URL? = nil, environment: [String: String]? = nil,
                    stdout: @escaping @Sendable (Substring) -> Void, stderr: @escaping @Sendable (Substring) -> Void) async throws -> Int32 {
        #if os(macOS)
        let p = Process()
        p.executableURL = executable
        p.arguments = arguments
        if let cwd { p.currentDirectoryURL = cwd }
        if let environment { p.environment = environment }
        let out = Pipe(), err = Pipe()
        p.standardOutput = out; p.standardError = err
        let outSplit = LineSplitter(stdout), errSplit = LineSplitter(stderr)
        out.fileHandleForReading.readabilityHandler = { h in outSplit.feed(h.availableData) }
        err.fileHandleForReading.readabilityHandler = { h in errSplit.feed(h.availableData) }
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (c: CheckedContinuation<Void, Error>) in
                p.terminationHandler = { _ in c.resume() }
                do { try p.run() } catch { p.terminationHandler = nil; c.resume(throwing: error) }
            }
        } onCancel: { if p.isRunning { p.terminate() } }
        out.fileHandleForReading.readabilityHandler = nil
        err.fileHandleForReading.readabilityHandler = nil
        outSplit.feed(out.fileHandleForReading.readDataToEndOfFile()); outSplit.flush()
        errSplit.feed(err.fileHandleForReading.readDataToEndOfFile()); errSplit.flush()
        try Task.checkCancellation()
        return p.terminationStatus
        #else
        throw PipelineError.unsupported("\(executable.lastPathComponent) can't run on iOS (needs a Mac or a Linux worker)")
        #endif
    }

    final class LineSplitter: @unchecked Sendable {
        private let lock = NSLock()
        private var buffer = Data()
        private let sink: @Sendable (Substring) -> Void
        init(_ sink: @escaping @Sendable (Substring) -> Void) { self.sink = sink }
        func feed(_ d: Data) {
            guard !d.isEmpty else { return }
            lock.lock()
            buffer.append(d)
            var lines: [Substring] = []
            while let nl = buffer.firstIndex(of: 0x0A) {
                let lineData = buffer[buffer.startIndex..<nl]
                lines.append(Substring(String(decoding: lineData, as: UTF8.self)))
                buffer.removeSubrange(buffer.startIndex...nl)
            }
            lock.unlock()
            for l in lines where !l.isEmpty { sink(l.hasSuffix("\r") ? l.dropLast() : l) }
        }
        func flush() {
            lock.lock(); let rest = buffer; buffer = Data(); lock.unlock()
            if !rest.isEmpty { sink(Substring(String(decoding: rest, as: UTF8.self))) }
        }
    }
}

// MARK: - splatprep-backed pose solver and masker

/// SfM with learned features through splatprep (local subprocess). For remote execution see `WorkerClient.process`.
struct SplatPrepPoseSolver: PoseSolver {
    var runner = SplatPrepRunner()
    var features: String = "superpoint_lightglue"
    var matching: String = "auto"
    var mapper: String = "auto"
    var cameraModel: String = "pinhole"
    var maxKeypoints: Int? = nil
    var resizeMax: Int? = nil
    /// Priors file relative to the frame-set root (e.g. ARKit poses).
    var priorPoses: String? = nil
    /// Use masks of the frames to drop keypoints on masked pixels.
    var useMasks: Bool = true

    /// Builds the request for a frame set (also used for remote `/v1/process`).
    func request(for frames: FrameSet) -> ProcessingRequest {
        var r = ProcessingRequest(task: .sfm)
        r.imagesDir = "images"; r.outputDir = "sfm"
        r.features = features; r.matching = matching; r.mapper = mapper; r.cameraModel = cameraModel
        r.maxKeypoints = maxKeypoints; r.resizeMax = resizeMax; r.priorPoses = priorPoses
        let sizes = Set(frames.frames.map { "\($0.width)x\($0.height)" })
        r.singleCamera = sizes.count == 1 && frames.rig == nil
        if frames.rig == nil, !frames.isMultiViewRig, frames.frames.contains(where: { $0.cameraGroup != nil }) {
            // home video: a camera per clip (or per zoom setting within one)
            r.singleCamera = false
            r.cameraGroups = Dictionary(frames.frames.compactMap { f in f.cameraGroup.map { ((f.image as NSString).lastPathComponent, $0) } },
                                        uniquingKeysWith: { a, _ in a })
        }
        if useMasks, frames.frames.contains(where: { $0.mask != nil }) { r.masksDir = "masks" }
        if frames.frames.contains(where: { $0.sky != nil }) { r.skyMasksDir = "splatprep-sky/masks" }
        if let rig = frames.rig {
            r.rig = .init(kind: "stereo", baseline: rig.baseline, rightToLeft: rig.rightToLeft,
                          pairs: RigConstraints.stereoPairs(frames).map { [($0.0.image as NSString).lastPathComponent, ($0.1.image as NSString).lastPathComponent] })
        } else if frames.isMultiViewRig {
            var groups: [Int: [String: [[Float]]]] = [:]
            for f in frames.frames { if let g = f.group, let R = f.rotationToRig { groups[g, default: [:]][(f.image as NSString).lastPathComponent] = R } }
            var hint = ProcessingRequest.RigHint(kind: "multiview", groups: groups.keys.sorted().map { groups[$0]! })
            if let f = frames.frames.first, let fp = f.focalPx,
               frames.frames.allSatisfy({ $0.focalPx == fp && $0.width == f.width && $0.height == f.height }) {
                hint.intrinsics = [fp, fp, f.cx ?? Float(f.width) / 2, f.cy ?? Float(f.height) / 2]
            }
            r.rig = hint
        }
        return r
    }

    func solve(_ frames: FrameSet, progress: PipelineProgressHandler?) async throws -> PoseSolution {
        let req = request(for: frames)
        let stats = try await runner.run(req, workDir: frames.root, progress: progress)
        var sol = try Self.readSolution(sfmDir: frames.root.appendingPathComponent(req.outputDir), frames: frames, prefix: req.outputDir,
                                        preferOutputImages: cameraModel != "pinhole")
        sol.stats = stats
        return sol
    }

    /// Reads splatprep's output dataset and maps cameras back to frames by image basename.
    /// - Parameter preferOutputImages: use splatprep's (undistorted) image copies instead of the original frames.
    static func readSolution(sfmDir: URL, frames: FrameSet, prefix: String, preferOutputImages: Bool = false) throws -> PoseSolution {
        let ds = try PreparedDataset.load(sfmDir)
        // splatprep ≥ 2026-11-09 says whether it undistorted, and writes undistorted masks next to the images
        let raw = (try? JSONSerialization.jsonObject(with: Data(contentsOf: sfmDir.appendingPathComponent("dataset.json")))) as? [String: Any]
        let undistorted = (raw?["splatprep"] as? [String: Any])?["undistorted"] as? Bool ?? false
        let rawMasks: [String: String] = Dictionary(((raw?["cameras"] as? [[String: Any]]) ?? []).compactMap { c in
            guard let i = c["image"] as? String, let m = c["mask"] as? String else { return nil }
            return (i, m)
        }, uniquingKeysWith: { a, _ in a })
        var byBase: [String: IngestedFrame] = [:]
        for f in frames.frames { byBase[(f.image as NSString).lastPathComponent] = f; byBase[((f.image as NSString).lastPathComponent as NSString).deletingPathExtension] = f }
        var sol = PoseSolution()
        for c in ds.manifest.cameras {
            let base = (c.image as NSString).lastPathComponent
            guard let f = byBase[base] ?? byBase[(base as NSString).deletingPathExtension] else { continue }
            var p = SolvedPose(c2w: c.c2w, fx: c.fx, fy: c.fy, cx: c.cx, cy: c.cy)
            // Undistorted copies / resized images: point at splatprep's file.
            let exported = FileManager.default.fileExists(atPath: sfmDir.appendingPathComponent(c.image).path)
            if exported && (preferOutputImages || c.width != f.width || c.height != f.height) {
                p.image = "\(prefix)/\(c.image)"; p.width = c.width; p.height = c.height
                if undistorted {
                    p.undistorted = true
                    p.mask = rawMasks[c.image].map { "\(prefix)/\($0)" }
                }
            }
            sol.poses[f.image] = p
        }
        if let pu = ds.pointCloudURL, FileManager.default.fileExists(atPath: pu.path) { sol.points = try PLYCodec.readPointCloud(url: pu) }
        return sol
    }
}

// MARK: - Python environment

#if os(macOS)
/// Managed Python virtual environment for splatprep at `~/Library/Application Support/SplatStudio/python`.
@MainActor
@Observable
final class PythonEnvironment {
    enum State: Equatable { case unknown, missing, installing, ready(version: String), failed(String) }

    static let shared = PythonEnvironment()

    nonisolated static var root: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("SplatStudio/python", isDirectory: true)
    }
    nonisolated static var defaultPython: URL { root.appendingPathComponent("bin/python") }

    private(set) var state: State = .unknown
    /// Installer output (pip / venv), newest last; capped.
    private(set) var log: [String] = []
    /// Models importable in the environment (`python -m splatprep models --json` → "available").
    private(set) var availableModels: [String] = []
    private var installTask: Task<Void, Error>?

    /// Finds `tools/` (the directory containing the `splatprep` package):
    /// `$SPLATSTUDIO_TOOLS`, app bundle `Resources/tools`, or walking up from the executable / cwd.
    nonisolated static func locateToolsDirectory() -> URL? {
        let fm = FileManager.default
        func ok(_ u: URL) -> Bool { fm.fileExists(atPath: u.appendingPathComponent("splatprep/__init__.py").path) }
        if let e = ProcessInfo.processInfo.environment["SPLATSTUDIO_TOOLS"] { let u = URL(fileURLWithPath: e); if ok(u) { return u } }
        if let r = Bundle.main.resourceURL?.appendingPathComponent("tools"), ok(r) { return r }
        for start in [Bundle.main.executableURL?.deletingLastPathComponent(), URL(fileURLWithPath: fm.currentDirectoryPath)].compactMap({ $0 }) {
            var d = start
            for _ in 0..<8 {
                let t = d.appendingPathComponent("tools")
                if ok(t) { return t }
                d = d.deletingLastPathComponent()
            }
        }
        return nil
    }

    /// Candidate base interpreters. 3.11/3.12 first: several model packages (VGGT pins numpy 1.26)
    /// have no wheels for 3.13 yet. Homebrew / python.org / CLT.
    nonisolated static func findBasePython() -> URL? {
        let fm = FileManager.default
        var c: [String] = []
        for v in ["3.11", "3.12", "3.10", "3.13"] {
            c += ["/opt/homebrew/bin/python\(v)", "/usr/local/bin/python\(v)", "/Library/Frameworks/Python.framework/Versions/\(v)/bin/python3"]
        }
        c += ["/opt/homebrew/bin/python3", "/usr/local/bin/python3", "/usr/bin/python3"]
        return c.first { fm.isExecutableFile(atPath: $0) }.map { URL(fileURLWithPath: $0) }
    }

    /// Checks whether the venv exists and splatprep imports.
    func refresh() async {
        let py = Self.defaultPython
        guard FileManager.default.isExecutableFile(atPath: py.path) else { state = .missing; return }
        do {
            let v = try await ProcessUtil.run(py.path, ["-c", "import sys; print('%d.%d.%d' % sys.version_info[:3])"])
            var env = ProcessInfo.processInfo.environment
            if let t = Self.locateToolsDirectory() { env["PYTHONPATH"] = t.path }
            let models = try? await ProcessUtil.run(py.path, ["-m", "splatprep", "models", "--json"], environment: env)
            if let m = models, let d = m.data(using: .utf8), let obj = try? JSONDecoder().decode([String: JSONValue].self, from: d) {
                availableModels = obj["available"]?.array?.compactMap(\.string) ?? []
                state = .ready(version: v.trimmingCharacters(in: .whitespacesAndNewlines))
            } else {
                state = .failed("splatprep is not importable — reinstall")
            }
        } catch { state = .failed(error.localizedDescription) }
    }

    /// Creates the venv (if needed) and pip-installs splatprep with the given extras, streaming output to `log`.
    /// - Parameters:
    ///   - extras: pip extras of the splatprep project (e.g. `["torch", "lightglue", "sam2"]`), if it has a pyproject.
    ///   - extraPackages: additional pip requirement specifiers.
    func install(extras: [String] = [], extraPackages: [String] = [], basePython: URL? = nil) async throws {
        if state == .installing, let t = installTask { return try await t.value }
        state = .installing
        log.removeAll()
        let task = Task { @MainActor in
            let fm = FileManager.default
            let root = Self.root
            let py = Self.defaultPython
            if !fm.isExecutableFile(atPath: py.path) {
                guard let base = basePython ?? Self.findBasePython() else {
                    throw PipelineError.processFailed("No Python 3.10+ found. Install it with `brew install python@3.12`.")
                }
                try fm.ensureDirectory(root.deletingLastPathComponent())
                try await stream(base, ["-m", "venv", "--upgrade-deps", root.path])
            }
            try await stream(py, ["-m", "pip", "install", "--upgrade", "pip", "wheel"])
            var specs: [String] = []
            if let tools = Self.locateToolsDirectory() {
                let project = [tools.appendingPathComponent("splatprep"), tools].first {
                    fm.fileExists(atPath: $0.appendingPathComponent("pyproject.toml").path) || fm.fileExists(atPath: $0.appendingPathComponent("setup.py").path)
                }
                if let project {
                    specs.append(project.path + (extras.isEmpty ? "" : "[\(extras.joined(separator: ","))]"))
                    specs.insert("-e", at: 0)
                } else {
                    for req in [tools.appendingPathComponent("splatprep/requirements.txt"), tools.appendingPathComponent("requirements.txt")]
                    where fm.fileExists(atPath: req.path) { specs += ["-r", req.path]; break }
                    if specs.isEmpty { specs += ["numpy", "pillow", "opencv-python-headless"] }
                }
            }
            specs += extraPackages
            if !specs.isEmpty { try await stream(py, ["-m", "pip", "install"] + specs) }
        }
        installTask = task
        do {
            try await task.value
            installTask = nil
            await refresh()
        } catch {
            installTask = nil
            state = .failed(error.localizedDescription)
            throw error
        }
    }

    /// Cancels a running installation.
    func cancelInstall() { installTask?.cancel() }

    private func appendLog(_ s: String) {
        log.append(s)
        if log.count > 2000 { log.removeFirst(log.count - 2000) }
    }

    private func stream(_ exe: URL, _ args: [String]) async throws {
        appendLog("$ \(exe.lastPathComponent) \(args.joined(separator: " "))")
        let tail = LineTail()
        let sink: @Sendable (Substring) -> Void = { [weak self] l in
            let s = String(l); tail.append(s)
            Task { @MainActor in self?.appendLog(s) }
        }
        let status = try await LineStreamingProcess.run(executable: exe, arguments: args, stdout: sink, stderr: sink)
        guard status == 0 else { throw PipelineError.processFailed("\(exe.lastPathComponent) \(args.first ?? "") failed (\(status)):\n\(tail.text)") }
    }
}
#else
/// iOS: no Python. splatprep steps (learned masks, camera solving) run on a Linux worker instead.
enum PythonEnvironment {
    static var root: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("SplatStudio/python", isDirectory: true)
    }
    static var defaultPython: URL { root.appendingPathComponent("bin/python") }
    static func locateToolsDirectory() -> URL? { nil }
}
#endif

// MARK: - Model cards & benchmarks (for UI display)

/// One entry of `tools/splatprep/model_cards.json`. All fields optional; unknown keys preserved in `extra`.
struct ModelCard: Codable, Identifiable, Hashable, Sendable {
    var id: String
    var name: String?
    /// "features" | "matcher" | "dense" | "feedforward" | "mask" …
    var kind: String?
    var summary: String?
    var description: String?
    var license: String?
    var paper: String?
    var url: String?
    var weightsMB: Double?
    var devices: [String]?
    var bestFor: [String]?
    var strengths: [String]?
    var weaknesses: [String]?
    var speed: String?
    var quality: String?
    var pipExtra: String?
    var extra: [String: JSONValue] = [:]

    private struct AnyKey: CodingKey {
        var stringValue: String; var intValue: Int? { nil }
        init(_ s: String) { stringValue = s }
        init?(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { nil }
    }
    private static let known: Set<String> = ["id", "name", "title", "kind", "type", "summary", "description", "license", "paper", "url", "homepage",
                                             "weightsMB", "devices", "bestFor", "recommendedFor", "strengths", "weaknesses", "speed", "quality", "pipExtra", "extra"]

    init(id: String) { self.id = id }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: AnyKey.self)
        func s(_ k: String) -> String? { try? c.decodeIfPresent(String.self, forKey: AnyKey(k)) }
        func a(_ k: String) -> [String]? { try? c.decodeIfPresent([String].self, forKey: AnyKey(k)) }
        id = s("id") ?? decoder.codingPath.last?.stringValue ?? "unknown"
        name = s("name") ?? s("title"); kind = s("kind") ?? s("type"); summary = s("summary"); description = s("description")
        license = s("license"); paper = s("paper"); url = s("url") ?? s("homepage")
        weightsMB = try? c.decodeIfPresent(Double.self, forKey: AnyKey("weightsMB"))
        devices = a("devices"); bestFor = a("bestFor") ?? a("recommendedFor"); strengths = a("strengths"); weaknesses = a("weaknesses")
        speed = s("speed") ?? (try? c.decodeIfPresent(JSONValue.self, forKey: AnyKey("speed")))??.description
        quality = s("quality") ?? (try? c.decodeIfPresent(JSONValue.self, forKey: AnyKey("quality")))??.description
        pipExtra = s("pipExtra")
        for k in c.allKeys where !Self.known.contains(k.stringValue) {
            if let v = try? c.decode(JSONValue.self, forKey: k) { extra[k.stringValue] = v }
        }
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: AnyKey.self)
        try c.encode(id, forKey: AnyKey("id"))
        try c.encodeIfPresent(name, forKey: AnyKey("name")); try c.encodeIfPresent(kind, forKey: AnyKey("kind"))
        try c.encodeIfPresent(summary, forKey: AnyKey("summary")); try c.encodeIfPresent(description, forKey: AnyKey("description"))
        try c.encodeIfPresent(license, forKey: AnyKey("license")); try c.encodeIfPresent(paper, forKey: AnyKey("paper"))
        try c.encodeIfPresent(url, forKey: AnyKey("url")); try c.encodeIfPresent(weightsMB, forKey: AnyKey("weightsMB"))
        try c.encodeIfPresent(devices, forKey: AnyKey("devices")); try c.encodeIfPresent(bestFor, forKey: AnyKey("bestFor"))
        try c.encodeIfPresent(strengths, forKey: AnyKey("strengths")); try c.encodeIfPresent(weaknesses, forKey: AnyKey("weaknesses"))
        try c.encodeIfPresent(speed, forKey: AnyKey("speed")); try c.encodeIfPresent(quality, forKey: AnyKey("quality"))
        try c.encodeIfPresent(pipExtra, forKey: AnyKey("pipExtra"))
        for (k, v) in extra { try c.encode(v, forKey: AnyKey(k)) }
    }

    var displayName: String { name ?? id }

    /// Loads model cards. Accepts `[card]`, `{"models": [card] | {id: card}}` or `{id: card}`.
    static func load(_ url: URL) throws -> [ModelCard] {
        let data = try Data(contentsOf: url)
        let dec = JSONDecoder()
        if let arr = try? dec.decode([ModelCard].self, from: data) { return arr }
        let obj = try dec.decode([String: JSONValue].self, from: data)
        let body: JSONValue = obj["models"] ?? obj["cards"] ?? .object(obj)
        func card(_ v: JSONValue, id: String?) -> ModelCard? {
            guard var o = v.object else { return nil }
            if o["id"] == nil, let id { o["id"] = .string(id) }
            guard let d = try? JSONEncoder().encode(JSONValue.object(o)) else { return nil }
            return try? dec.decode(ModelCard.self, from: d)
        }
        switch body {
        case .array(let a): return a.compactMap { card($0, id: nil) }
        case .object(let o): return o.sorted { $0.key < $1.key }.compactMap { k, v in k.hasPrefix("$") || k == "format" || k == "version" ? nil : card(v, id: k) }
        default: return []
        }
    }

    /// `tools/splatprep/model_cards.json` if found.
    static func loadDefault() -> [ModelCard] {
        guard let t = PythonEnvironment.locateToolsDirectory() else { return [] }
        return (try? load(t.appendingPathComponent("splatprep/model_cards.json"))) ?? []
    }
}

/// `benchmark.json` written by `splatprep benchmark` (format "splatbenchmark").
struct BenchmarkReport: Codable, Sendable {
    struct Host: Codable, Sendable {
        var platform: String?; var release: String?; var machine: String?; var processor: String?
        var python: String?; var device: String?; var cpuCount: Int?
    }
    struct DatasetInfo: Codable, Sendable { var name: String?; var path: String?; var numImages: Int?; var groundTruth: String?; var numGroundTruthPoses: Int? }
    struct ModelStatus: Codable, Sendable { var available: Bool?; var sfm: Bool?; var reason: String?; var kind: String? }
    var format: String?
    var version: Int?
    var splatprepVersion: String?
    var createdAt: String?
    var host: Host?
    var options: [String: JSONValue]?
    var models: [String: ModelStatus]?
    var datasets: [DatasetInfo]?
    var results: [BenchmarkResult]?
    var summary: [String: BenchmarkSummary]?

    static func load(_ url: URL) throws -> BenchmarkReport { try JSONDecoder().decode(BenchmarkReport.self, from: Data(contentsOf: url)) }

    /// Newest `benchmark.json` (by modification date) anywhere under `directory`.
    /// The most complete `benchmark.json` under `directory` (most model × dataset results), newest first
    /// among equals — so a partial rerun (e.g. one model) never hides the full report.
    static func latest(in directory: URL) -> (url: URL, report: BenchmarkReport)? {
        let fm = FileManager.default
        guard let e = fm.enumerator(at: directory, includingPropertiesForKeys: [.contentModificationDateKey]) else { return nil }
        var best: (url: URL, report: BenchmarkReport, rows: Int, date: Date)? = nil
        while let u = e.nextObject() as? URL {
            guard u.lastPathComponent == "benchmark.json", let r = try? load(u) else { continue }
            let d = (try? u.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            let rows = r.results?.count ?? 0
            if best == nil || rows > best!.rows || (rows == best!.rows && d > best!.date) { best = (u, r, rows, d) }
        }
        return best.map { ($0.url, $0.report) }
    }

    /// Default location for benchmark runs: `~/Library/Application Support/SplatStudio/benchmarks`.
    static var defaultDirectory: URL { PythonEnvironment.root.deletingLastPathComponent().appendingPathComponent("benchmarks") }
}

/// One (model, dataset) run of a benchmark.
struct BenchmarkResult: Codable, Sendable, Identifiable {
    struct Metrics: Codable, Sendable {
        var numImages: Int?; var numRegistered: Int?; var registeredPct: Double?
        var meanReprojError: Double?; var meanTrackLength: Double?; var numPoints3D: Int?
        var wallTimeSec: Double?; var peakMemoryMB: Double?; var peakGpuMemoryMB: Double?
        var device: String?; var mapper: String?; var numPairs: Int?
        /// e.g. aucAt5, aucAt10, ateRmseRel
        var pose: [String: JSONValue]?
        /// e.g. meanInlierRatio, pairPoseAucAt10
        var matching: [String: JSONValue]?
    }
    var model: String
    var dataset: String?
    /// ok | matching-only | failed | unavailable
    var status: String?
    var error: String?
    var metrics: Metrics?
    var id: String { model + "/" + (dataset ?? "") }
}

/// Per-model aggregate of a benchmark.
struct BenchmarkSummary: Codable, Sendable {
    var status: String?
    var numDatasets: Int?; var numSucceeded: Int?
    var meanRegisteredPct: Double?; var meanReprojError: Double?; var meanTrackLength: Double?
    var meanNumPoints3D: Double?; var meanWallTimeSec: Double?; var maxPeakMemoryMB: Double?
    var meanAucAt5: Double?; var meanAucAt10: Double?; var meanAteRmseRel: Double?
    var meanInlierRatio: Double?; var meanPairPoseAucAt10: Double?
}
