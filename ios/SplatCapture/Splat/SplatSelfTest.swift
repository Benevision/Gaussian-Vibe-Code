#if DEBUG
import Foundation
import SplatTrainKit
import SwiftUI

/// Debug-only end-to-end check of "Make splat" without the UI (the Simulator has no ARKit to record with):
///
///     xcrun simctl launch <device> com.gunntower.splatcapture -splatSelfTest
///
/// Environment (pass as SIMCTL_CHILD_…): SPLAT_TEST_CAPTURE (bundle path; default: newest capture in the
/// library), SPLAT_TEST_ITERS, SPLAT_TEST_STRIDE, SPLAT_TEST_DEVICES (`local` | `remote` | `all`),
/// SPLAT_TEST_WAIT (seconds of worker discovery), SPLAT_TEST_MASKS (a `SplatMaskMode` raw value),
/// SPLAT_TEST_WORKER (host[:port],… to add by hand; SPLAT_TEST_DEVICES `manual` / `local+manual` uses only those). Progress goes to Documents/selftest.log; the
/// result opens in the viewer.
@MainActor
enum SplatSelfTest {
    static var isRequested: Bool { ProcessInfo.processInfo.arguments.contains("-splatSelfTest") }

    static func run(library: CaptureLibrary, onResult: @escaping (SplatOutput) -> Void) async {
        let env = ProcessInfo.processInfo.environment
        let logURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("selftest.log")
        try? FileManager.default.removeItem(at: logURL)
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        let sink = LogSink(try? FileHandle(forWritingTo: logURL))
        let log: @Sendable (String) -> Void = { sink.write($0) }
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        if env["SPLAT_TEST_RENDER_ONLY"] != nil {
            let out = SplatOutput(folder: docs.appendingPathComponent("Splats/SelfTest", isDirectory: true))
            do { try SplatDebugRender.render(ply: out.plyURL, dataset: out.datasetURL, view: 50, to: out.folder); log("offscreen renders written") }
            catch { log("offscreen render failed: \(error.localizedDescription)") }
            onResult(out)
            return
        }
        await library.reload()
        let capture = env["SPLAT_TEST_CAPTURE"].map { URL(fileURLWithPath: $0) } ?? library.items.first?.url
        guard let capture else { log("FAILED: no capture"); return }
        let devices = SplatDevices.shared
        devices.start()
        var manual: [String] = []
        for w in (env["SPLAT_TEST_WORKER"] ?? "").split(separator: ",") {
            let p = w.split(separator: ":")
            manual.append(devices.addManual(host: String(p[0]), port: p.count > 1 ? Int(p[1]) ?? 7878 : 7878))
        }
        try? await Task.sleep(for: .seconds(Double(env["SPLAT_TEST_WAIT"] ?? "6") ?? 6))
        await devices.refresh()
        for d in devices.all { log("device \(d.id) \(d.name) [\(d.kind.rawValue)] online=\(d.isOnline) train=\(d.canTrain) process=\(d.canProcess) — \(d.detail)") }
        var o = SplatMakeOptions()
        o.iterationsOverride = env["SPLAT_TEST_ITERS"].flatMap(Int.init) ?? 500
        o.frameStride = env["SPLAT_TEST_STRIDE"].flatMap(Int.init) ?? 4
        o.masks = env["SPLAT_TEST_MASKS"].flatMap(SplatMaskMode.init(rawValue:)) ?? .none
        o.refinePoses = env["SPLAT_TEST_REFINE"] == "1"
        o.sky = env["SPLAT_TEST_SKY"] == "1"
        o.learnedMovingMasks = env["SPLAT_TEST_LEARNED"] == "1"
        o.processorId = manual.first ?? "none"   // never the real PCs on the network
        let online = devices.remote.filter(\.isOnline).map(\.id)
        switch env["SPLAT_TEST_DEVICES"] ?? "local" {
        case "manual": o.deviceIds = Set(manual)                                  // only the workers added by address
        case "local+manual": o.deviceIds = Set(manual + [SplatDeviceInfo.localId])
        case "remote": o.deviceIds = Set(online)
        case "all": o.deviceIds = Set(online + [SplatDeviceInfo.localId])
        default: o.deviceIds = [SplatDeviceInfo.localId]
        }
        log("capture \(capture.lastPathComponent), devices \(o.deviceIds.sorted()), iterations \(o.iterationsOverride ?? 0), stride \(o.frameStride), masks \(o.masks.rawValue)")
        let work = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0].appendingPathComponent("SelfTest", isDirectory: true)
        let last = LastStage()
        do {
            let r = try await SplatMaker.make(capture: capture, workDir: work, options: o, devices: devices) { p in
                if last.changed(p.stage) || p.step % 100 == 0 && p.step > 0 {
                    let line = "\(p.stage) \(Int(max(0, p.fraction) * 100))% \(p.message)" + (p.psnr.map { String(format: " psnr %.2f", $0) } ?? "") + (p.lines.isEmpty ? "" : " | " + p.lines.joined(separator: " | "))
                    log(line)
                }
            }
            for n in r.notes { log("note: \(n)") }
            log("DONE: \(r.splatCount) splats, \(r.cameras) cameras, \(Int(r.seconds)) s → \(r.plyURL.path)")
            let out = SplatOutput(folder: FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("Splats/SelfTest", isDirectory: true))
            try? FileManager.default.removeItem(at: out.folder)
            try? FileManager.default.createDirectory(at: out.folder, withIntermediateDirectories: true)
            try? FileManager.default.copyItem(at: r.plyURL, to: out.plyURL)
            try? FileManager.default.copyItem(at: r.datasetURL.appendingPathComponent("dataset.json"), to: out.folder.appendingPathComponent("dataset.json"))
            do { try SplatDebugRender.render(ply: out.plyURL, dataset: out.datasetURL, view: 50, to: out.folder); log("offscreen renders written") }
            catch { log("offscreen render failed: \(error.localizedDescription)") }
            if env["SPLAT_TEST_NO_VIEWER"] == nil { onResult(out) }
        } catch {
            log("FAILED: \(error.localizedDescription)")
        }
    }

    final class LogSink: @unchecked Sendable {
        private let lock = NSLock()
        private let fh: FileHandle?
        init(_ fh: FileHandle?) { self.fh = fh }
        func write(_ s: String) {
            lock.lock(); defer { lock.unlock() }
            fh?.write(Data("[\(ISO8601DateFormatter().string(from: Date()))] \(s)\n".utf8))
            try? fh?.synchronize()
        }
    }

    final class LastStage: @unchecked Sendable {
        private let lock = NSLock()
        private var stage = ""
        func changed(_ s: String) -> Bool { lock.lock(); defer { lock.unlock() }; if s == stage { return false }; stage = s; return true }
    }
}
#endif
