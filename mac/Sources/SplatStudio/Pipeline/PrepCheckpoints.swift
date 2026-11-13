import Foundation

/// Stages of dataset preparation that leave a checkpoint behind.
enum PrepStage: String, Codable, CaseIterable, Comparable, Sendable {
    /// Frames from the source (photos ingested, video split, 360° views cut, or a posed dataset imported).
    case frames
    /// Masks (fixed parts of the frame, moving objects / people / subject, sky).
    case masks
    /// Camera positions (solved, or capture poses refined), assembled into a dataset.
    case poses
    /// Levelled, centred and ready to train.
    case done

    static func < (a: PrepStage, b: PrepStage) -> Bool { a.order < b.order }
    private var order: Int { Self.allCases.firstIndex(of: self)! }

    var title: String {
        switch self {
        case .frames: return "Frames"
        case .masks: return "Masks"
        case .poses: return "Camera positions"
        case .done: return "Dataset"
        }
    }
}

/// Checkpoints of a project's dataset preparation, in `dataset/.prep/`. After each stage the stage's output
/// (the frame list, or a copy of `dataset.json` and its point cloud) is saved with a fingerprint of everything that
/// produced it: the source files and the settings of this stage and all earlier ones. A later run of the same
/// project skips every stage whose fingerprint still matches, so quitting the app, a crash, a lost worker or
/// pressing Stop never costs more than the stage that was running. Changing a setting redoes that stage and the
/// ones after it, nothing earlier. "Redo from…" drops the checkpoints from a stage on.
struct PrepCheckpoints: Sendable {
    /// The dataset folder (`<project>/dataset`).
    let root: URL
    var dir: URL { root.appendingPathComponent(".prep", isDirectory: true) }

    struct Entry: Codable, Sendable {
        var fingerprint: String
        var savedAt: Date
        /// Short description for the project page ("812 frames", "3 masks layers", "798 of 812 cameras").
        var summary: String
    }

    private func entryURL(_ s: PrepStage) -> URL { dir.appendingPathComponent("\(s.rawValue).json") }
    private func framesURL(_ s: PrepStage) -> URL { dir.appendingPathComponent("\(s.rawValue).frames.json") }
    private func manifestURL(_ s: PrepStage) -> URL { dir.appendingPathComponent("\(s.rawValue).dataset.json") }
    private func pointsURL(_ s: PrepStage) -> URL { dir.appendingPathComponent("\(s.rawValue).points.ply") }
    /// Copies of the mask files a checkpoint refers to. Later stages write combined masks to the same names
    /// (splatprep finds masks by image name), so a checkpoint keeps its own copies and puts them back on restore.
    private func filesURL(_ s: PrepStage) -> URL { dir.appendingPathComponent("\(s.rawValue).files", isDirectory: true) }

    private func snapshot(_ rels: [String], as s: PrepStage, from source: URL? = nil) {
        let fm = FileManager.default
        let base = filesURL(s)
        try? fm.removeItem(at: base)
        for rel in Set(rels) {
            let src = (source ?? root).appendingPathComponent(rel), dst = base.appendingPathComponent(rel)
            guard fm.fileExists(atPath: src.path) else { continue }
            try? fm.createDirectory(at: dst.deletingLastPathComponent(), withIntermediateDirectories: true)
            try? fm.copyItem(at: src, to: dst)
        }
    }

    /// Puts the checkpoint's mask files back (checkpoints saved before copies were kept have none: nothing to do).
    private func restoreFiles(_ rels: [String], of s: PrepStage) {
        let fm = FileManager.default
        let base = filesURL(s)
        guard fm.fileExists(atPath: base.path) else { return }
        for rel in Set(rels) {
            let src = base.appendingPathComponent(rel), dst = root.appendingPathComponent(rel)
            guard fm.fileExists(atPath: src.path) else { continue }
            try? fm.createDirectory(at: dst.deletingLastPathComponent(), withIntermediateDirectories: true)
            try? fm.removeItem(at: dst)
            try? fm.copyItem(at: src, to: dst)
        }
    }

    func entry(_ s: PrepStage) -> Entry? {
        guard let d = try? Data(contentsOf: entryURL(s)) else { return nil }
        return try? JSONDecoder.iso.decode(Entry.self, from: d)
    }

    /// Checkpoints that exist, in stage order (whatever their fingerprints).
    var saved: [(stage: PrepStage, entry: Entry)] { PrepStage.allCases.compactMap { s in entry(s).map { (s, $0) } } }

    private func writeEntry(_ s: PrepStage, _ fp: String, _ summary: String) throws {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try JSONEncoder.iso.encode(Entry(fingerprint: fp, savedAt: Date(), summary: summary)).write(to: entryURL(s), options: .atomic)
    }

    // MARK: frame lists (photos / video / 360° / stereo sources)

    /// - Parameter filesFrom: where the mask files it names are, when not in the dataset folder as they are now
    ///   (a worker's snapshot of an earlier stage: later stages wrote combined masks under the same names).
    func save(_ frames: FrameSet, as s: PrepStage, fingerprint fp: String, summary: String, filesFrom: URL? = nil) throws {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try JSONEncoder().encode(frames).write(to: framesURL(s), options: .atomic)
        snapshot(frames.frames.flatMap { [$0.mask, $0.sky].compactMap { $0 } }, as: s, from: filesFrom)
        try writeEntry(s, fp, summary)
    }

    /// The saved frames when the fingerprint matches and every file they name is still there (here, or on the
    /// worker that holds this project's frames).
    func frames(_ s: PrepStage, fingerprint fp: String) -> FrameSet? {
        guard entry(s)?.fingerprint == fp, let d = try? Data(contentsOf: framesURL(s)),
              var fs = try? JSONDecoder().decode(FrameSet.self, from: d), !fs.frames.isEmpty else { return nil }
        fs.root = root      // the project may have been moved
        let store = RemoteStore.load(root)
        let fm = FileManager.default
        func present(_ rel: String?) -> Bool {
            guard let rel else { return true }
            if fm.fileExists(atPath: root.appendingPathComponent(rel).path) { return true }
            let top = rel.split(separator: "/").first.map(String.init) ?? rel
            return store?.dirs.contains(top) == true && store?.local == false
        }
        restoreFiles(fs.frames.flatMap { [$0.mask, $0.sky].compactMap { $0 } }, of: s)
        guard fs.frames.allSatisfy({ present($0.image) && present($0.mask) && present($0.sky) }) else { return nil }
        return fs
    }

    // MARK: dataset copies (imported posed sources, and the solved dataset)

    func save(_ ds: PreparedDataset, as s: PrepStage, fingerprint fp: String, summary: String) throws {
        let fm = FileManager.default
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        try ds.save()
        try? fm.removeItem(at: manifestURL(s)); try? fm.removeItem(at: pointsURL(s))
        try fm.copyItem(at: root.appendingPathComponent(PreparedDataset.fileName), to: manifestURL(s))
        if let p = ds.pointCloudURL, fm.fileExists(atPath: p.path) { try fm.copyItem(at: p, to: pointsURL(s)) }
        snapshot(ds.manifest.cameras.flatMap { [$0.mask, $0.sky].compactMap { $0 } }, as: s)
        try writeEntry(s, fp, summary)
    }

    /// Restores the saved dataset (as `dataset.json` and its point cloud) when the fingerprint matches and its
    /// images are still there.
    func dataset(_ s: PrepStage, fingerprint fp: String) -> PreparedDataset? {
        let fm = FileManager.default
        guard entry(s)?.fingerprint == fp, fm.fileExists(atPath: manifestURL(s).path),
              let saved = try? Data(contentsOf: manifestURL(s)) else { return nil }
        let live = root.appendingPathComponent(PreparedDataset.fileName)
        let previous = try? Data(contentsOf: live)
        do {
            try saved.write(to: live, options: .atomic)
            var ds = try PreparedDataset.load(root)
            let store = RemoteStore.load(root)
            let imagesElsewhere = store?.dirs.contains("images") == true && store?.local == false
            guard imagesElsewhere || ds.manifest.cameras.allSatisfy({ fm.fileExists(atPath: root.appendingPathComponent($0.image).path) })
            else { throw PipelineError.noInput("images missing") }
            // only once the restore will be used: the point cloud and mask files it refers to
            if let p = ds.pointCloudURL, fm.fileExists(atPath: pointsURL(s).path) {
                try? fm.removeItem(at: p)
                try fm.copyItem(at: pointsURL(s), to: p)
            }
            restoreFiles(ds.manifest.cameras.flatMap { [$0.mask, $0.sky].compactMap { $0 } }, of: s)
            ds.root = root
            return ds
        } catch {
            if let previous { try? previous.write(to: live, options: .atomic) }
            return nil
        }
    }

    /// The finished (levelled) dataset is `dataset.json` itself: only the marker is saved.
    func markDone(fingerprint fp: String, summary: String) throws { try writeEntry(.done, fp, summary) }

    func isDone(fingerprint fp: String) -> Bool { entry(.done)?.fingerprint == fp }

    // MARK: redo

    /// Forgets the checkpoints of `stage` and every later one, and the worker jobs they came from, so the next run
    /// does that work again. Redoing the frames also forgets the reused video split.
    func invalidate(from stage: PrepStage) {
        let fm = FileManager.default
        for s in PrepStage.allCases where s >= stage {
            for u in [entryURL(s), framesURL(s), manifestURL(s), pointsURL(s), filesURL(s)] { try? fm.removeItem(at: u) }
        }
        var steps: [String] = []
        if stage <= .frames { steps.append("frames"); FrameSourceStamp.forget(root) }
        if stage <= .masks { steps.append("masks") }
        if stage <= .poses { steps.append("sfm") }
        RemoteJobLedger.forget(root, steps: steps)
    }

    // MARK: fingerprints

    /// Source files: path, size and modification time of each file (a folder: of each file directly inside it).
    static func sourceIdentity(_ source: SourceDescription) -> String {
        func id(_ u: URL?) -> String {
            guard let u else { return "-" }
            let fm = FileManager.default
            func one(_ f: URL) -> String {
                let a = try? fm.attributesOfItem(atPath: f.path)
                let size = (a?[.size] as? NSNumber)?.int64Value ?? -1
                let mod = (a?[.modificationDate] as? Date)?.timeIntervalSince1970.rounded() ?? -1
                return "\(f.lastPathComponent)|\(size)|\(Int(mod))"
            }
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: u.path, isDirectory: &isDir) else { return u.standardizedFileURL.path + "|missing" }
            if !isDir.boolValue { return u.standardizedFileURL.path + "|" + one(u) }
            // every file below the folder (sub-folders too: a recursive photo import, COLMAP's sparse/0), by its
            // path inside the folder, size and modification time
            var files: [String] = []
            let base = u.standardizedFileURL.path
            if let e = fm.enumerator(at: u, includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles]) {
                for case let f as URL in e where (try? f.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true {
                    let rel = String(f.standardizedFileURL.path.dropFirst(base.count))
                    files.append(rel + "|" + one(f))
                    if files.count > 200_000 { break }
                }
            }
            return base + "/[" + files.sorted().joined(separator: ",") + "]"
        }
        switch source {
        case .photos(let u), .video(let u), .video360(let u), .homeVideo(let u), .photos360(let u), .captureBundle(let u), .nerfstudio(let u):
            return id(u)
        case .photoFiles(let us): return us.map { id($0) }.joined(separator: ";")
        case .stereoVideo(let l, let r): return id(l) + ";" + id(r)
        case .colmap(let p, let i): return id(p) + ";" + id(i)
        }
    }

    private static func hash(_ parts: [String]) -> String { sha256Hex(Data(parts.joined(separator: "\n").utf8)) }

    /// Frames: the source and every setting that changes which frames come out and how they look.
    static func framesFingerprint(_ source: SourceDescription, _ o: PrepOptions) -> String {
        let o = o.resolved
        var parts = ["frames v1", canonicalJSON(source), sourceIdentity(source)]
        switch source {
        case .photos, .photoFiles: parts.append(canonicalJSON(o.photos))
        case .video: parts.append(canonicalJSON(o.video))
        case .homeVideo: parts.append(canonicalJSON(o.homeVideo))
        case .video360: parts += [canonicalJSON(o.video), canonicalJSON(o.equirect)]
        case .photos360: parts += [canonicalJSON(o.equirect), canonicalJSON(o.photos.encoding)]
        case .stereoVideo: parts += [canonicalJSON(o.video), canonicalJSON(o.stereo)]
        case .captureBundle: parts.append(canonicalJSON(o.capture))
        case .colmap, .nerfstudio: parts.append(canonicalJSON(o.importing))
        }
        return hash(parts)
    }

    static func masksFingerprint(after frames: String, _ o: PrepOptions) -> String {
        hash(["masks v1", frames, canonicalJSON(o.masking), String(o.maskStaticRegions)])
    }

    static func posesFingerprint(after masks: String, _ o: PrepOptions) -> String {
        let o = o.resolved
        return hash(["poses v1", masks, canonicalJSON(o.poseSolver), o.sfmCameraModel, String(o.objectCaptureHighSensitivity),
                     String(o.refineCapturePoses), String(o.metricScaleFromStereo), String(o.enforceRigConstraints),
                     o.holdoutInterval.map(String.init) ?? "-", o.maxImageSize.map(String.init) ?? "-"])
    }

    static func doneFingerprint(after poses: String) -> String { hash(["done v1", poses]) }

    /// All four, for one source and set of options.
    static func fingerprints(_ source: SourceDescription, _ o: PrepOptions) -> [PrepStage: String] {
        let f = framesFingerprint(source, o), m = masksFingerprint(after: f, o), p = posesFingerprint(after: m, o)
        return [.frames: f, .masks: m, .poses: p, .done: doneFingerprint(after: p)]
    }

    /// The first stage that still has to run for these settings (`nil`: the dataset is ready).
    func firstStageToRun(_ source: SourceDescription, _ o: PrepOptions) -> PrepStage? {
        let fps = Self.fingerprints(source, o)
        if isDone(fingerprint: fps[.done]!), (try? PreparedDataset.load(root)) != nil { return nil }
        for s in [PrepStage.poses, .masks, .frames] where entry(s)?.fingerprint == fps[s] { return PrepStage.allCases.first { $0 > s } }
        return .frames
    }
}
