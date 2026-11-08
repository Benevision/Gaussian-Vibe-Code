import Foundation
import AppKit
import CryptoKit
import Observation

/// Updates from the GitHub Releases of this repository (the `release.yml` workflow publishes
/// `SplatWork-<version>.dmg` plus `SHA256SUMS.txt` for every `v*` tag).
///
/// Flow: check at launch and every 6 h → download the .dmg in the background → verify its SHA-256 against the
/// release's checksum file → copy the app out of the disk image and check its signature, bundle id and version →
/// prompt. "Relaunch to Update" swaps the bundle once this process has exited (a small shell helper waits for our
/// PID, so the running app's files are never replaced underneath it) and opens the new version; "Later" installs
/// it when you next quit. Builds made from source (`-dev` versions) only see releases newer than the release they
/// were built after.
@MainActor
@Observable
final class AppUpdater {
    static let shared = AppUpdater()
    nonisolated static let repository = "Benevision/Gaussian-Vibe-Code"
    static let checkInterval: TimeInterval = 6 * 3600

    struct Release: Equatable {
        var version: String
        var tag: String
        var notes: String
        var page: URL
        var dmg: URL
        var dmgName: String
        var dmgSize: Int
        var checksums: URL?
    }

    enum Phase: Equatable {
        case idle, checking, upToDate
        case available            // found, not downloaded (automatic download off)
        case downloading(Double)
        case ready                // verified and staged: relaunch or quit to install
        case failed(String)
    }

    private(set) var phase: Phase = .idle
    private(set) var release: Release?
    private(set) var lastChecked: Date?
    /// The update sheet.
    var showPrompt = false
    /// True while the work this app is doing would be interrupted by a relaunch (set by the app at launch).
    var isBusy: () -> Bool = { false }

    var autoCheck: Bool = UserDefaults.standard.object(forKey: "updates.autoCheck") as? Bool ?? true {
        didSet { UserDefaults.standard.set(autoCheck, forKey: "updates.autoCheck") }
    }
    var autoDownload: Bool = UserDefaults.standard.object(forKey: "updates.autoDownload") as? Bool ?? true {
        didSet { UserDefaults.standard.set(autoDownload, forKey: "updates.autoDownload") }
    }
    private var skippedVersion: String? {
        get { UserDefaults.standard.string(forKey: "updates.skipped") }
        set { UserDefaults.standard.set(newValue, forKey: "updates.skipped") }
    }

    let currentVersion: String? = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String
    /// Only an app bundle can update itself (not `swift run`, not the CLI).
    var isSupported: Bool { currentVersion != nil && Bundle.main.bundleURL.pathExtension == "app" }

    private var timer: Timer?
    private var stagedApp: URL?
    private var downloadTask: URLSessionDownloadTask?

    // MARK: Scheduling

    func startAutomaticChecks() {
        guard isSupported, timer == nil else { return }
        cleanOldStaging()
        DispatchQueue.main.asyncAfter(deadline: .now() + 15) { [weak self] in
            Task { @MainActor in if self?.autoCheck == true { await self?.check(userInitiated: false) } }
        }
        timer = Timer.scheduledTimer(withTimeInterval: Self.checkInterval, repeats: true) { [weak self] _ in
            Task { @MainActor in if self?.autoCheck == true { await self?.check(userInitiated: false) } }
        }
    }

    // MARK: Check

    func check(userInitiated: Bool) async {
        guard isSupported, let current = currentVersion else {
            if userInitiated { phase = .failed("Updates work only in the SplatWork app bundle."); showPrompt = true }
            return
        }
        switch phase {
        case .checking, .downloading: if userInitiated { showPrompt = true }; return
        case .ready: if userInitiated { showPrompt = true }; return
        default: break
        }
        phase = .checking
        if userInitiated { showPrompt = true }
        do {
            guard let r = try await Self.fetchLatest(), Self.isNewer(r.version, than: current) else {
                lastChecked = Date(); phase = .upToDate; release = nil
                return
            }
            lastChecked = Date()
            release = r
            if !userInitiated && skippedVersion == r.version { phase = .idle; return }
            if autoDownload || userInitiated { await download(r, promptWhenReady: true) }
            else { phase = .available; showPrompt = true }
        } catch {
            phase = userInitiated ? .failed(error.localizedDescription) : .idle
        }
    }

    nonisolated static func fetchLatest() async throws -> Release? {
        var req = URLRequest(url: URL(string: "https://api.github.com/repos/\(repository)/releases/latest")!)
        req.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        req.setValue("SplatWork-Updater", forHTTPHeaderField: "User-Agent")
        req.timeoutInterval = 20
        let (data, resp) = try await URLSession.shared.data(for: req)
        let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
        if code == 404 { return nil }   // no releases yet
        guard code == 200 else { throw UpdateError("GitHub answered \(code)") }
        return try parseRelease(data)
    }

    nonisolated static func parseRelease(_ data: Data) throws -> Release? {
        guard let j = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let tag = j["tag_name"] as? String else { throw UpdateError("Unexpected answer from GitHub") }
        if j["draft"] as? Bool == true || j["prerelease"] as? Bool == true { return nil }
        let assets = j["assets"] as? [[String: Any]] ?? []
        func asset(_ match: (String) -> Bool) -> (URL, String, Int)? {
            for a in assets {
                if let n = a["name"] as? String, match(n), let s = a["browser_download_url"] as? String, let u = URL(string: s) {
                    return (u, n, a["size"] as? Int ?? 0)
                }
            }
            return nil
        }
        guard let dmg = asset({ $0.lowercased().hasSuffix(".dmg") }) else { return nil }
        let sums = asset { $0.uppercased().hasPrefix("SHA256SUMS") }
        let version = tag.hasPrefix("v") ? String(tag.dropFirst()) : tag
        let page = (j["html_url"] as? String).flatMap(URL.init(string:))
            ?? URL(string: "https://github.com/\(repository)/releases")!
        return Release(version: version, tag: tag, notes: j["body"] as? String ?? "", page: page,
                       dmg: dmg.0, dmgName: dmg.1, dmgSize: dmg.2, checksums: sums?.0)
    }

    /// Numeric comparison of `1.2.3[-suffix]`. A suffix marks a build *before* that release (`-rc1`), except
    /// `-dev`, which this repo's local builds use for "built from source after that release".
    nonisolated static func isNewer(_ remote: String, than local: String) -> Bool {
        func split(_ v: String) -> ([Int], String) {
            let parts = v.split(separator: "-", maxSplits: 1).map(String.init)
            let nums = (parts.first ?? "").split(separator: ".").map { Int($0) ?? 0 }
            return (nums, parts.count > 1 ? parts[1].lowercased() : "")
        }
        let (r, rs) = split(remote), (l, ls) = split(local)
        for i in 0..<max(r.count, l.count) {
            let a = i < r.count ? r[i] : 0, b = i < l.count ? l[i] : 0
            if a != b { return a > b }
        }
        // Same numbers: a final release is newer than a pre-release of it, but not newer than a dev build after it.
        if rs.isEmpty { return !ls.isEmpty && !ls.hasPrefix("dev") }
        return false
    }

    // MARK: Download and stage

    func downloadAvailable() {
        guard let r = release else { return }
        Task { await download(r, promptWhenReady: true) }
    }

    private func download(_ r: Release, promptWhenReady: Bool) async {
        phase = .downloading(0)
        let dir = Self.stagingRoot.appendingPathComponent(r.version, isDirectory: true)
        do {
            try? FileManager.default.removeItem(at: dir)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            let dmg = dir.appendingPathComponent(r.dmgName)
            try await fetch(r.dmg, to: dmg, expected: r.dmgSize)
            if let sums = r.checksums {
                let (data, _) = try await URLSession.shared.data(from: sums)
                guard let want = Self.checksum(in: String(decoding: data, as: UTF8.self), for: r.dmgName) else {
                    throw UpdateError("The release's checksum file doesn't list \(r.dmgName).")
                }
                let have = try await Task.detached { try Self.sha256(of: dmg) }.value
                guard have == want else { throw UpdateError("The download is damaged (checksum mismatch). Try again.") }
            }
            let app = try await Task.detached { try Self.extractApp(from: dmg, into: dir, version: r.version) }.value
            try? FileManager.default.removeItem(at: dmg)
            stagedApp = app
            phase = .ready
            if promptWhenReady { showPrompt = true }
        } catch {
            try? FileManager.default.removeItem(at: dir)
            phase = .failed(error.localizedDescription)
            showPrompt = true
        }
    }

    private func fetch(_ url: URL, to dest: URL, expected: Int) async throws {
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            let task = URLSession.shared.downloadTask(with: url) { tmp, resp, err in
                if let err { cont.resume(throwing: err); return }
                guard let tmp, (resp as? HTTPURLResponse)?.statusCode == 200 else {
                    cont.resume(throwing: UpdateError("Download failed (\((resp as? HTTPURLResponse)?.statusCode ?? 0))")); return
                }
                do { try FileManager.default.moveItem(at: tmp, to: dest); cont.resume() } catch { cont.resume(throwing: error) }
            }
            downloadTask = task
            task.resume()
            Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] t in
                Task { @MainActor in
                    guard let self, case .downloading = self.phase, task.state == .running else { t.invalidate(); return }
                    let total = task.countOfBytesExpectedToReceive > 0 ? task.countOfBytesExpectedToReceive : Int64(expected)
                    if total > 0 { self.phase = .downloading(min(1, Double(task.countOfBytesReceived) / Double(total))) }
                }
            }
        }
    }

    nonisolated static func checksum(in text: String, for name: String) -> String? {
        for line in text.split(whereSeparator: \.isNewline) {
            let f = line.split(separator: " ", omittingEmptySubsequences: true)
            if f.count >= 2, f.last.map({ $0.hasPrefix("*") ? String($0.dropFirst()) : String($0) }) == name {
                return f[0].lowercased()
            }
        }
        return nil
    }

    nonisolated static func sha256(of url: URL) throws -> String {
        let h = try FileHandle(forReadingFrom: url)
        defer { try? h.close() }
        var hasher = SHA256()
        while let chunk = try h.read(upToCount: 8 << 20), !chunk.isEmpty { hasher.update(data: chunk) }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// Mounts the disk image, copies the app next to it and checks it is a signed SplatWork of that version.
    nonisolated static func extractApp(from dmg: URL, into dir: URL, version: String) throws -> URL {
        let mount = dir.appendingPathComponent("mnt", isDirectory: true)
        try FileManager.default.createDirectory(at: mount, withIntermediateDirectories: true)
        try run("/usr/bin/hdiutil", ["attach", "-nobrowse", "-readonly", "-noautoopen", "-quiet", "-mountpoint", mount.path, dmg.path])
        defer { _ = try? run("/usr/bin/hdiutil", ["detach", "-quiet", "-force", mount.path]); try? FileManager.default.removeItem(at: mount) }
        guard let name = try FileManager.default.contentsOfDirectory(atPath: mount.path).first(where: { $0.hasSuffix(".app") }) else {
            throw UpdateError("The disk image has no app in it.")
        }
        let app = dir.appendingPathComponent(name)
        try run("/usr/bin/ditto", [mount.appendingPathComponent(name).path, app.path])
        let info = NSDictionary(contentsOf: app.appendingPathComponent("Contents/Info.plist")) as? [String: Any] ?? [:]
        guard info["CFBundleIdentifier"] as? String == (Bundle.main.bundleIdentifier ?? "com.example.splatstudio") else {
            throw UpdateError("The downloaded app isn't SplatWork (bundle id \(info["CFBundleIdentifier"] ?? "none")).")
        }
        guard info["CFBundleShortVersionString"] as? String == version else {
            throw UpdateError("The downloaded app is version \(info["CFBundleShortVersionString"] ?? "?"), expected \(version).")
        }
        try run("/usr/bin/codesign", ["--verify", "--deep", "--strict", app.path])
        _ = try? run("/usr/bin/xattr", ["-dr", "com.apple.quarantine", app.path])
        return app
    }

    @discardableResult
    nonisolated static func run(_ tool: String, _ args: [String]) throws -> String {
        let p = Process(); p.executableURL = URL(fileURLWithPath: tool); p.arguments = args
        let out = Pipe(); p.standardOutput = out; p.standardError = out
        try p.run()
        let data = out.fileHandleForReading.readDataToEndOfFile()
        p.waitUntilExit()
        let text = String(decoding: data, as: UTF8.self)
        guard p.terminationStatus == 0 else {
            throw UpdateError("\((tool as NSString).lastPathComponent) failed: \(text.trimmingCharacters(in: .whitespacesAndNewlines))")
        }
        return text
    }

    // MARK: Install

    /// Why the running copy can't be replaced in place (nil = it can).
    var installProblem: String? {
        let dest = Bundle.main.bundleURL
        if dest.path.contains("/AppTranslocation/") {
            return "macOS is running SplatWork from a temporary location. Move it to Applications (drag it out of the disk image), open it from there, and check again."
        }
        if !FileManager.default.isWritableFile(atPath: dest.deletingLastPathComponent().path) {
            return "SplatWork can't write to \(dest.deletingLastPathComponent().path). Open the disk image and drag the new version over the old one."
        }
        return nil
    }

    /// Quits and relaunches into the new version.
    func relaunchToUpdate() {
        guard launchInstaller(relaunch: true) else { return }
        NSApp.terminate(nil)
    }

    /// "Later": the staged update goes in when the app quits (called from applicationWillTerminate).
    func installOnQuitIfReady() {
        guard phase == .ready, !installStarted else { return }
        _ = launchInstaller(relaunch: false)
    }

    func skipThisVersion() {
        skippedVersion = release?.version
        showPrompt = false
        if phase == .available { phase = .idle }
        if phase == .ready, let app = stagedApp { try? FileManager.default.removeItem(at: app.deletingLastPathComponent()); stagedApp = nil; phase = .idle }
    }

    func openStagedDiskImageInstead() {
        if let r = release { NSWorkspace.shared.open(r.page) }
    }

    private var installStarted = false

    private func launchInstaller(relaunch: Bool) -> Bool {
        guard let app = stagedApp, FileManager.default.fileExists(atPath: app.path) else {
            phase = .failed("The downloaded update is gone. Check again."); return false
        }
        if let problem = installProblem { phase = .failed(problem); showPrompt = true; return false }
        let script = app.deletingLastPathComponent().appendingPathComponent("install.sh")
        do {
            try Self.installScript.write(to: script, atomically: true, encoding: .utf8)
            let p = Process()
            p.executableURL = URL(fileURLWithPath: "/bin/sh")
            p.arguments = [script.path, String(ProcessInfo.processInfo.processIdentifier), app.path,
                           Bundle.main.bundleURL.path, relaunch ? "1" : "0"]
            p.standardOutput = FileHandle.nullDevice; p.standardError = FileHandle.nullDevice
            try p.run()
            installStarted = true
            return true
        } catch {
            phase = .failed("Couldn't start the installer: \(error.localizedDescription)"); showPrompt = true
            return false
        }
    }

    /// Waits for the app to exit, swaps the bundle (the old one is restored if the copy fails), relaunches.
    nonisolated static let installScript = """
    #!/bin/sh
    PID="$1"; NEW="$2"; DEST="$3"; RELAUNCH="$4"
    LOG="$(dirname "$NEW")/install.log"
    exec >>"$LOG" 2>&1
    echo "$(date) waiting for $PID"
    i=0; while kill -0 "$PID" 2>/dev/null; do sleep 0.5; i=$((i+1)); [ $i -gt 600 ] && { echo "app did not quit"; exit 1; }; done
    OLD="$DEST.updating-$$"
    mv "$DEST" "$OLD" || { echo "cannot move old app"; exit 1; }
    if /usr/bin/ditto "$NEW" "$DEST"; then
      rm -rf "$OLD" "$NEW"
      /usr/bin/xattr -dr com.apple.quarantine "$DEST" 2>/dev/null
      echo "installed"
    else
      echo "copy failed: restoring"; rm -rf "$DEST"; mv "$OLD" "$DEST"
    fi
    [ "$RELAUNCH" = 1 ] && /usr/bin/open "$DEST"
    exit 0
    """

    static var stagingRoot: URL {
        FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("SplatWork/Updates", isDirectory: true)
    }

    private func cleanOldStaging() {
        let fm = FileManager.default
        for v in (try? fm.contentsOfDirectory(atPath: Self.stagingRoot.path)) ?? [] {
            try? fm.removeItem(at: Self.stagingRoot.appendingPathComponent(v))
        }
    }
}

struct UpdateError: LocalizedError {
    let message: String
    init(_ m: String) { message = m }
    var errorDescription: String? { message }
}
