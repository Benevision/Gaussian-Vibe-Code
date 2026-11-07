import Foundation
import Observation
import UIKit

/// Zips a capture and POSTs it to Splat Studio (PROTOCOL §5), with progress and retry.
@MainActor
@Observable
final class CaptureUploader {
    enum Phase: Equatable {
        case idle
        case connecting
        case zipping(Double)
        case uploading(Double)
        case done(savedAs: String)
        case failed(message: String, needsPairingCode: Bool)

        var isBusy: Bool {
            switch self {
            case .connecting, .zipping, .uploading: true
            default: false
            }
        }
    }

    enum UploadError: LocalizedError {
        case http(Int, String?)
        case pairing
        case notSplatStudio

        var errorDescription: String? {
            switch self {
            case let .http(code, msg): msg.map { "The Mac refused the upload (\(code)): \($0)" } ?? "The Mac refused the upload (HTTP \(code))."
            case .pairing: "The Mac requires the 6-digit pairing code shown in Splat Studio › Receive from iPhone."
            case .notSplatStudio: "That device doesn't look like Splat Studio."
            }
        }
    }

    private(set) var phase: Phase = .idle
    private(set) var macName: String?
    private(set) var bytesTotal: Int64 = 0
    private(set) var attempt = 0
    private var task: Task<Void, Never>?

    private static let session: URLSession = {
        let c = URLSessionConfiguration.default
        c.timeoutIntervalForRequest = 60
        c.timeoutIntervalForResource = 6 * 3600
        c.waitsForConnectivity = false
        return URLSession(configuration: c)
    }()

    func send(captureURL: URL, captureName: String, to mac: DiscoveredMac, pairingCode: String?) {
        cancel()
        macName = mac.name
        let code = pairingCode?.filter(\.isNumber)
        task = Task { [weak self] in
            await self?.run(captureURL: captureURL, captureName: captureName, mac: mac,
                            pairingCode: (code?.isEmpty ?? true) ? nil : code)
        }
    }

    func cancel() {
        task?.cancel()
        task = nil
        if phase.isBusy { phase = .idle }
    }

    func reset() {
        cancel()
        phase = .idle
    }

    private func run(captureURL: URL, captureName: String, mac: DiscoveredMac, pairingCode: String?) async {
        let app = UIApplication.shared
        var bgTask: UIBackgroundTaskIdentifier = .invalid
        bgTask = app.beginBackgroundTask(withName: "Send capture") {
            // Expired — the upload will fail and can be retried.
        }
        app.isIdleTimerDisabled = true
        defer {
            app.isIdleTimerDisabled = false
            if bgTask != .invalid { app.endBackgroundTask(bgTask) }
        }

        let zipURL = CaptureStore.exportsDirectory.appendingPathComponent("\(captureURL.deletingPathExtension().lastPathComponent).zip")
        defer { try? FileManager.default.removeItem(at: zipURL) }

        do {
            phase = .connecting
            let base = try await EndpointResolver.baseURL(for: mac.target)
            try await hello(base)

            phase = .zipping(0)
            try await Self.zip(captureURL, to: zipURL) { [weak self] p in
                Task { @MainActor in
                    if case .zipping = self?.phase { self?.phase = .zipping(p) }
                }
            }
            try Task.checkCancellation()
            bytesTotal = (try? FileManager.default.attributesOfItem(atPath: zipURL.path)[.size] as? NSNumber)?.int64Value ?? 0

            // Upload with up to 3 attempts for transient network errors.
            var lastError: (any Error)?
            for i in 0..<3 {
                attempt = i + 1
                do {
                    let saved = try await upload(zipURL, base: base, name: captureName, pairingCode: pairingCode)
                    phase = .done(savedAs: saved)
                    UINotificationFeedbackGenerator().notificationOccurred(.success)
                    return
                } catch let e as UploadError {
                    throw e
                } catch {
                    if Task.isCancelled { throw error }
                    lastError = error
                    try await Task.sleep(for: .seconds(Double(i + 1) * 2))
                }
            }
            throw lastError ?? URLError(.unknown)
        } catch is CancellationError {
            phase = .idle
        } catch let e as URLError where e.code == .cancelled {
            phase = .idle
        } catch UploadError.pairing {
            phase = .failed(message: UploadError.pairing.localizedDescription, needsPairingCode: true)
            UINotificationFeedbackGenerator().notificationOccurred(.warning)
        } catch {
            phase = .failed(message: error.localizedDescription, needsPairingCode: false)
            UINotificationFeedbackGenerator().notificationOccurred(.error)
        }
    }

    private func hello(_ base: URL) async throws {
        var req = URLRequest(url: base.appendingPathComponent("v1/hello"))
        req.timeoutInterval = 8
        let (data, resp) = try await Self.session.data(for: req)
        guard let http = resp as? HTTPURLResponse, http.statusCode == 200 else {
            throw UploadError.notSplatStudio
        }
        struct Hello: Decodable { var name: String?; var app: String?; var version: Int? }
        if let h = try? JSONDecoder().decode(Hello.self, from: data), let n = h.name, !n.isEmpty {
            macName = n
        }
    }

    private func upload(_ zipURL: URL, base: URL, name: String, pairingCode: String?) async throws -> String {
        phase = .uploading(0)
        var req = URLRequest(url: base.appendingPathComponent("v1/captures"))
        req.httpMethod = "POST"
        req.setValue("application/zip", forHTTPHeaderField: "Content-Type")
        var allowed = CharacterSet.alphanumerics
        allowed.insert(charactersIn: "-._~")
        req.setValue(name.addingPercentEncoding(withAllowedCharacters: allowed) ?? "Capture",
                     forHTTPHeaderField: "X-Capture-Name")
        if let pairingCode { req.setValue(pairingCode, forHTTPHeaderField: "X-Pairing-Code") }

        let delegate = UploadProgressDelegate { [weak self] fraction in
            Task { @MainActor in
                if case .uploading = self?.phase { self?.phase = .uploading(fraction) }
            }
        }
        let (data, resp) = try await Self.session.upload(for: req, fromFile: zipURL, delegate: delegate)
        guard let http = resp as? HTTPURLResponse else { throw URLError(.badServerResponse) }
        struct Reply: Decodable { var ok: Bool?; var savedAs: String?; var error: String? }
        let reply = try? JSONDecoder().decode(Reply.self, from: data)
        if http.statusCode == 403 || http.statusCode == 401 { throw UploadError.pairing }
        guard (200..<300).contains(http.statusCode) else { throw UploadError.http(http.statusCode, reply?.error) }
        return reply?.savedAs ?? "\(name).splatcapture"
    }

    /// Zips off the main actor.
    nonisolated static func zip(_ source: URL, to dest: URL,
                                progress: @escaping @Sendable (Double) -> Void) async throws {
        let work = Task.detached(priority: .userInitiated) {
            var lastReport = 0.0
            try ZipWriter.zipDirectory(source, to: dest) { done, total in
                let f = total > 0 ? Double(done) / Double(total) : 1
                if f - lastReport > 0.01 || f >= 1 {
                    lastReport = f
                    progress(f)
                }
            }
        }
        // Detached tasks don't inherit cancellation; forward it so ZipWriter stops early.
        try await withTaskCancellationHandler {
            try await work.value
        } onCancel: {
            work.cancel()
        }
    }
}

final class UploadProgressDelegate: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    private let handler: @Sendable (Double) -> Void
    private var last = 0.0

    init(_ handler: @escaping @Sendable (Double) -> Void) { self.handler = handler }

    func urlSession(_ session: URLSession, task: URLSessionTask, didSendBodyData bytesSent: Int64,
                    totalBytesSent: Int64, totalBytesExpectedToSend: Int64) {
        guard totalBytesExpectedToSend > 0 else { return }
        let f = Double(totalBytesSent) / Double(totalBytesExpectedToSend)
        if f - last > 0.005 || f >= 1 {
            last = f
            handler(f)
        }
    }
}
