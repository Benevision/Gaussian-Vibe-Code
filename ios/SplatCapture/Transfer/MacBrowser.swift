import Foundation
import Network
import Observation
import os

/// A SplatWork instance advertising `_splatcapture._tcp` (PROTOCOL §5).
struct DiscoveredMac: Identifiable, Hashable, Sendable {
    enum Target: Hashable, Sendable {
        case bonjour(NWEndpoint)
        case manual(host: String, port: Int)
    }

    let id: String
    let name: String
    let target: Target
    let version: String?
}

/// Browses the local network for Macs running SplatWork.
@MainActor
@Observable
final class MacBrowser {
    enum State: Equatable {
        case idle
        case browsing
        case waiting(String)
        case failed(String)
    }

    static let serviceType = "_splatcapture._tcp"

    private(set) var macs: [DiscoveredMac] = []
    private(set) var state: State = .idle
    private var browser: NWBrowser?

    func start() {
        guard browser == nil else { return }
        let params = NWParameters()
        params.includePeerToPeer = true
        let b = NWBrowser(for: .bonjourWithTXTRecord(type: Self.serviceType, domain: nil), using: params)
        b.stateUpdateHandler = { [weak self] newState in
            MainActor.assumeIsolated { self?.handle(newState) }
        }
        b.browseResultsChangedHandler = { [weak self] results, _ in
            MainActor.assumeIsolated { self?.update(results) }
        }
        browser = b
        state = .browsing
        b.start(queue: .main)
    }

    func stop() {
        browser?.cancel()
        browser = nil
        state = .idle
    }

    func restart() {
        stop()
        macs = []
        start()
    }

    private func handle(_ s: NWBrowser.State) {
        switch s {
        case .ready, .setup: state = .browsing
        case .waiting(let error):
            state = .waiting("Can't search the local network (\(error.localizedDescription)). Check Settings › Privacy & Security › Local Network › SplatWork Capture, and that you're on the same Wi-Fi as your Mac.")
        case .failed(let error):
            state = .failed(error.localizedDescription)
            browser?.cancel()
            browser = nil
        case .cancelled: break
        @unknown default: break
        }
    }

    private func update(_ results: Set<NWBrowser.Result>) {
        var found: [DiscoveredMac] = []
        for r in results {
            guard case let .service(serviceName, _, _, _) = r.endpoint else { continue }
            var display = serviceName
            var version: String?
            if case let .bonjour(txt) = r.metadata {
                if let n = txt["name"], !n.isEmpty { display = n }
                version = txt["version"]
            }
            found.append(DiscoveredMac(id: serviceName, name: display, target: .bonjour(r.endpoint), version: version))
        }
        macs = found.sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
    }
}

/// Resolves a Bonjour service endpoint to a concrete host/port for URLSession.
enum EndpointResolver {
    enum ResolveError: LocalizedError {
        case timeout
        case failed(String)
        var errorDescription: String? {
            switch self {
            case .timeout: "Timed out finding the Mac on the network."
            case .failed(let s): "Could not reach the Mac: \(s)"
            }
        }
    }

    static func baseURL(for target: DiscoveredMac.Target) async throws -> URL {
        switch target {
        case let .manual(host, port):
            return try url(host: host, port: port)
        case let .bonjour(endpoint):
            // Prefer IPv4 (simpler URLs), fall back to whatever the system picks.
            if let v4 = try? await resolve(endpoint, ipv4Only: true, timeout: 5) {
                return try url(host: v4.0, port: v4.1)
            }
            let any = try await resolve(endpoint, ipv4Only: false, timeout: 8)
            return try url(host: any.0, port: any.1)
        }
    }

    private static func url(host: String, port: Int) throws -> URL {
        var h = host
        if h.contains(":") && !h.hasPrefix("[") {
            h = "[" + h.replacingOccurrences(of: "%", with: "%25") + "]"
        }
        guard let u = URL(string: "http://\(h):\(port)") else { throw ResolveError.failed("bad address \(host)") }
        return u
    }

    private static func resolve(_ endpoint: NWEndpoint, ipv4Only: Bool, timeout: TimeInterval) async throws -> (String, Int) {
        let params = NWParameters.tcp
        params.includePeerToPeer = true
        if ipv4Only, let ip = params.defaultProtocolStack.internetProtocol as? NWProtocolIP.Options {
            ip.version = .v4
        }
        let connection = NWConnection(to: endpoint, using: params)
        let box = UncheckedSendableBox(connection)
        let once = OSAllocatedUnfairLock(initialState: false)
        let queue = DispatchQueue(label: "splatcapture.resolve")

        return try await withCheckedThrowingContinuation { (cont: CheckedContinuation<(String, Int), any Error>) in
            @Sendable func finish(_ result: Result<(String, Int), any Error>) {
                let first = once.withLock { done -> Bool in
                    if done { return false }
                    done = true
                    return true
                }
                guard first else { return }
                box.value.cancel()
                cont.resume(with: result)
            }
            connection.stateUpdateHandler = { state in
                switch state {
                case .ready:
                    if case let .hostPort(host, port)? = box.value.currentPath?.remoteEndpoint {
                        finish(.success((hostString(host), Int(port.rawValue))))
                    } else {
                        finish(.failure(ResolveError.failed("no address")))
                    }
                case .failed(let e):
                    finish(.failure(ResolveError.failed(e.localizedDescription)))
                case .cancelled:
                    finish(.failure(ResolveError.failed("cancelled")))
                default:
                    break
                }
            }
            queue.asyncAfter(deadline: .now() + timeout) { finish(.failure(ResolveError.timeout)) }
            connection.start(queue: queue)
        }
    }

    private static func hostString(_ host: NWEndpoint.Host) -> String {
        switch host {
        case .ipv4(let a):
            return a.rawValue.map { String($0) }.joined(separator: ".")
        case .ipv6(let a):
            return "\(a)"
        case .name(let n, _):
            return n
        @unknown default:
            return "\(host)"
        }
    }
}
