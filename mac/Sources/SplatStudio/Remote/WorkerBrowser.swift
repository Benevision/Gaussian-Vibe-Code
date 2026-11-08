import Foundation
import Network
import Observation
import Security

/// A worker the Mac can talk to (discovered via Bonjour or added manually).
struct WorkerEndpoint: Codable, Hashable, Identifiable, Sendable {
    /// Stable id: "bonjour:<service name>" or "manual:<host>:<port>".
    var id: String
    var name: String
    var host: String
    var port: Int
    var isManual: Bool
    /// Bonjour TXT record (`version`, `gpu`, `auth`).
    var txt: [String: String] = [:]
    /// Other addresses of the same worker (address → "ethernet"/"wifi"), from its TXT `addrs` or `/v1/info`.
    var alternates: [String: String] = [:]

    var baseURL: URL {
        let h = host.contains(":") && !host.hasPrefix("[") ? "[\(host)]" : host   // IPv6 literal
        return URL(string: "http://\(h):\(port)")!
    }
    var requiresAuth: Bool { txt["auth"] == "1" }
    /// "ethernet" / "wifi" for the address in use, when the worker said.
    var linkKind: String? { alternates[host] }

    enum CodingKeys: String, CodingKey { case id, name, host, port, isManual, txt, alternates }
    init(id: String, name: String, host: String, port: Int, isManual: Bool, txt: [String: String] = [:]) {
        self.id = id; self.name = name; self.host = host; self.port = port; self.isManual = isManual; self.txt = txt
        self.alternates = WorkerEndpoint.parseAddrs(txt["addrs"])
    }
    init(from d: Decoder) throws {
        let c = try d.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id); name = try c.decode(String.self, forKey: .name)
        host = try c.decode(String.self, forKey: .host); port = try c.decode(Int.self, forKey: .port)
        isManual = try c.decode(Bool.self, forKey: .isManual)
        txt = try c.decodeIfPresent([String: String].self, forKey: .txt) ?? [:]
        alternates = try c.decodeIfPresent([String: String].self, forKey: .alternates) ?? [:]
    }

    /// TXT `addrs` = "192.168.0.113:ethernet,192.168.0.120:wifi" (worker ≥ 1.3.10).
    static func parseAddrs(_ s: String?) -> [String: String] {
        var out: [String: String] = [:]
        for part in (s ?? "").split(separator: ",") {
            let kv = part.split(separator: ":", maxSplits: 1).map(String.init)
            if kv.count == 2, !kv[0].isEmpty { out[kv[0]] = kv[1] }
        }
        return out
    }

    /// The address to use: an ethernet address if the worker has one, else nil (keep the current one).
    static func preferredAddress(current: String, alternates: [String: String]) -> String? {
        if alternates[current] == "ethernet" { return nil }
        return alternates.filter { $0.value == "ethernet" }.keys.sorted().first
    }
    var gpuName: String? { txt["gpu"] }
}

/// Result of the last `/v1/info` health check.
struct WorkerHealth: Hashable, Sendable {
    enum Status: Hashable, Sendable { case unknown, checking, online, unauthorized, offline(String) }
    var status: Status = .unknown
    var info: WorkerInfo? = nil
    var latencyMs: Double? = nil
    var checkedAt: Date? = nil
}

/// Discovers SplatWork Workers (`_splatworker._tcp`), keeps manually added workers (UserDefaults; tokens in
/// the Keychain) and health-checks them via `GET /v1/info`.
@MainActor
@Observable
final class WorkerBrowser {
    nonisolated static let serviceType = "_splatworker._tcp"
    static let shared = WorkerBrowser()

    /// Discovered + manual workers, sorted by name.
    private(set) var workers: [WorkerEndpoint] = []
    /// Health per worker id.
    private(set) var health: [String: WorkerHealth] = [:]
    private(set) var isBrowsing = false
    private(set) var browserError: String?

    private var discovered: [String: WorkerEndpoint] = [:]
    private var manual: [WorkerEndpoint] = []
    private var browser: NWBrowser?
    private var resolving: [String: NWConnection] = [:]
    private var healthTimer: Task<Void, Never>?
    private let defaultsKey = "SplatStudio.manualWorkers"
    private let hiddenKey = "SplatStudio.hiddenWorkers"
    /// Discovered workers the user removed (by Bonjour id); they stay hidden until shown again.
    private(set) var hidden: Set<String> = []
    private let queue = DispatchQueue(label: "SplatStudio.WorkerBrowser")

    init() {
        if let d = UserDefaults.standard.data(forKey: defaultsKey), let m = try? JSONDecoder().decode([WorkerEndpoint].self, from: d) { manual = m }
        hidden = Set(UserDefaults.standard.stringArray(forKey: hiddenKey) ?? [])
        rebuild()
    }

    // MARK: Browsing

    /// Starts Bonjour discovery and periodic health checks (every `healthInterval` seconds).
    func start(healthInterval: TimeInterval = 15) {
        guard browser == nil else { return }
        let params = NWParameters()
        params.includePeerToPeer = true
        let b = NWBrowser(for: .bonjourWithTXTRecord(type: Self.serviceType, domain: nil), using: params)
        b.stateUpdateHandler = { [weak self] st in
            Task { @MainActor in
                guard let self else { return }
                switch st {
                case .ready: self.isBrowsing = true; self.browserError = nil
                case .failed(let e): self.isBrowsing = false; self.browserError = e.localizedDescription
                case .cancelled: self.isBrowsing = false
                default: break
                }
            }
        }
        b.browseResultsChangedHandler = { [weak self] results, _ in
            let snapshot = results.map { ($0.endpoint, $0.metadata) }
            Task { @MainActor in self?.handle(results: snapshot) }
        }
        b.start(queue: queue)
        browser = b
        healthTimer?.cancel()
        healthTimer = Task { [weak self] in
            while !Task.isCancelled {
                await self?.refreshHealth()
                try? await Task.sleep(nanoseconds: UInt64(healthInterval * 1e9))
            }
        }
    }

    func stop() {
        browser?.cancel(); browser = nil
        healthTimer?.cancel(); healthTimer = nil
        resolving.values.forEach { $0.cancel() }; resolving.removeAll()
        isBrowsing = false
    }

    private func handle(results: [(NWEndpoint, NWBrowser.Result.Metadata)]) {
        var seen = Set<String>()
        for (ep, md) in results {
            guard case let .service(name, _, _, _) = ep else { continue }
            let id = "bonjour:\(name)"
            seen.insert(id)
            var txt: [String: String] = [:]
            if case let .bonjour(record) = md { txt = record.dictionary }
            if var existing = discovered[id] {
                existing.txt = txt
                existing.alternates.merge(WorkerEndpoint.parseAddrs(txt["addrs"])) { _, new in new }
                discovered[id] = existing; continue
            }
            resolve(ep, id: id, name: name, txt: txt)
        }
        for id in discovered.keys where !seen.contains(id) { discovered.removeValue(forKey: id); health.removeValue(forKey: id) }
        rebuild()
    }

    /// Resolves a Bonjour service to host/port by opening a short-lived connection.
    private func resolve(_ ep: NWEndpoint, id: String, name: String, txt: [String: String]) {
        guard resolving[id] == nil else { return }
        let params = NWParameters.tcp
        if let ip = params.defaultProtocolStack.internetProtocol as? NWProtocolIP.Options { ip.version = .v4 }
        let c = NWConnection(to: ep, using: params)
        resolving[id] = c
        c.stateUpdateHandler = { [weak self] st in
            switch st {
            case .ready:
                var host: String?, port: Int?
                if case let .hostPort(h, p)? = c.currentPath?.remoteEndpoint {
                    switch h {
                    case .ipv4(let a): host = "\(a)".components(separatedBy: "%").first
                    case .ipv6(let a): host = "\(a)".components(separatedBy: "%").first
                    case .name(let n, _): host = n
                    @unknown default: host = "\(h)"
                    }
                    port = Int(p.rawValue)
                }
                c.cancel()
                Task { @MainActor in
                    guard let self else { return }
                    self.resolving.removeValue(forKey: id)
                    guard let host, let port else { return }
                    self.discovered[id] = WorkerEndpoint(id: id, name: name, host: host, port: port, isManual: false, txt: txt)
                    self.rebuild()
                    await self.checkHealth(id)
                }
            case .failed, .cancelled:
                Task { @MainActor in self?.resolving.removeValue(forKey: id) }
            default: break
            }
        }
        c.start(queue: queue)
    }

    // MARK: Manual workers

    /// Adds (or updates) a manually entered worker. The token is stored in the Keychain.
    @discardableResult
    func addManual(host: String, port: Int = 7878, name: String? = nil, token: String? = nil) -> WorkerEndpoint {
        let id = "manual:\(host):\(port)"
        let ep = WorkerEndpoint(id: id, name: name ?? host, host: host, port: port, isManual: true)
        manual.removeAll { $0.id == id }
        manual.append(ep)
        persist()
        if let token { setToken(token, for: id) }
        rebuild()
        Task { await checkHealth(id) }
        return ep
    }

    func removeManual(_ id: String) {
        manual.removeAll { $0.id == id }
        health.removeValue(forKey: id)
        Keychain.delete(account: id)
        persist(); rebuild()
    }

    /// Removes a Bonjour-discovered worker from the list (e.g. to add it manually by its ethernet address).
    func hide(_ id: String) {
        hidden.insert(id)
        health.removeValue(forKey: id)
        UserDefaults.standard.set(Array(hidden), forKey: hiddenKey)
        rebuild()
    }

    /// Shows all hidden discovered workers again.
    func unhideAll() {
        hidden.removeAll()
        UserDefaults.standard.set([String](), forKey: hiddenKey)
        rebuild()
        Task { await refreshHealth() }
    }

    private func persist() {
        if let d = try? JSONEncoder().encode(manual) { UserDefaults.standard.set(d, forKey: defaultsKey) }
    }

    /// Bonjour names this app advertises itself (SplatWork's "act as a worker", the iPhone's Compute screen):
    /// never listed as remote devices.
    var ownNames: Set<String> = [] { didSet { rebuild() } }

    private func rebuild() {
        workers = (discovered.values.filter { !hidden.contains($0.id) && !ownNames.contains($0.name) } + manual).sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
    }

    // MARK: Tokens

    func token(for id: String) -> String? { Keychain.read(account: id) }
    func setToken(_ token: String?, for id: String) {
        if let token, !token.isEmpty { Keychain.write(token, account: id) } else { Keychain.delete(account: id) }
        Task { await checkHealth(id) }
    }

    /// A ready-to-use client for a worker (with its stored token).
    func client(for id: String) -> WorkerClient? {
        guard let w = workers.first(where: { $0.id == id }) else { return nil }
        return WorkerClient(endpoint: w, token: token(for: id))
    }

    /// A client for the worker at this base URL (as recorded in a project's `RemoteStore`).
    func client(forBaseURL url: String) -> WorkerClient? {
        // also by any other address of the same worker (a discovered worker may have moved to its ethernet address)
        let host = URL(string: url)?.host
        guard let w = workers.first(where: { $0.baseURL.absoluteString == url })
                ?? workers.first(where: { w in host.map { w.alternates[$0] != nil } ?? false }) else { return nil }
        return WorkerClient(endpoint: w, token: token(for: w.id))
    }

    /// Workers that answered the last health check.
    var onlineWorkers: [WorkerEndpoint] { workers.filter { if case .online = health[$0.id]?.status { return true }; return false } }

    // MARK: Health

    func refreshHealth() async {
        await withTaskGroup(of: Void.self) { g in
            for w in workers { g.addTask { await self.checkHealth(w.id) } }
        }
    }

    func checkHealth(_ id: String) async {
        guard let w = workers.first(where: { $0.id == id }) else { return }
        var h = health[id] ?? WorkerHealth()
        if h.info == nil { h.status = .checking; health[id] = h }
        let client = WorkerClient(endpoint: w, token: token(for: id))
        let t0 = Date()
        do {
            let info = try await client.info(timeout: 4)
            h.info = info; h.status = .online
            h.latencyMs = Date().timeIntervalSince(t0) * 1000
            await learnAddresses(id, info: info)
        } catch let e as WorkerError where e.status == 401 {
            h.status = .unauthorized
        } catch {
            // A discovered worker on two networks: if this address stopped answering, try its others.
            if let moved = await switchToWorkingAddress(id, excluding: w.host) {
                h.info = moved; h.status = .online
                h.latencyMs = Date().timeIntervalSince(t0) * 1000
            } else {
                h.status = .offline(error.localizedDescription)
            }
        }
        h.checkedAt = Date()
        health[id] = h
    }

    /// Records the worker's addresses and moves a discovered worker onto ethernet when that address answers.
    /// Manually added workers keep the address the user typed.
    private func learnAddresses(_ id: String, info: WorkerInfo) async {
        guard var w = discovered[id] else { return }
        for a in info.addresses ?? [] { w.alternates[a.address] = a.kind ?? w.alternates[a.address] ?? "" }
        discovered[id] = w
        guard let target = WorkerEndpoint.preferredAddress(current: w.host, alternates: w.alternates) else { return }
        var probe = w; probe.host = target
        if (try? await WorkerClient(endpoint: probe, token: token(for: id)).info(timeout: 2)) != nil {
            discovered[id]?.host = target
            rebuild()
        }
    }

    private func switchToWorkingAddress(_ id: String, excluding bad: String) async -> WorkerInfo? {
        guard let w = discovered[id] else { return nil }
        let order = w.alternates.keys.filter { $0 != bad }.sorted { (w.alternates[$0] == "ethernet" ? 0 : 1, $0) < (w.alternates[$1] == "ethernet" ? 0 : 1, $1) }
        for a in order {
            var probe = w; probe.host = a
            if let info = try? await WorkerClient(endpoint: probe, token: token(for: id)).info(timeout: 2) {
                discovered[id]?.host = a
                rebuild()
                return info
            }
        }
        return nil
    }
}

/// Minimal generic-password Keychain wrapper for worker tokens.
enum Keychain {
    static let service = "SplatStudio.WorkerToken"

    static func write(_ value: String, account: String) {
        let base: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service, kSecAttrAccount as String: account]
        let data = Data(value.utf8)
        if SecItemUpdate(base as CFDictionary, [kSecValueData as String: data] as CFDictionary) == errSecItemNotFound {
            var add = base; add[kSecValueData as String] = data
            add[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
            SecItemAdd(add as CFDictionary, nil)
        }
    }

    static func read(account: String) -> String? {
        let q: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service, kSecAttrAccount as String: account,
                                kSecReturnData as String: true, kSecMatchLimit as String: kSecMatchLimitOne]
        var out: CFTypeRef?
        guard SecItemCopyMatching(q as CFDictionary, &out) == errSecSuccess, let d = out as? Data else { return nil }
        return String(data: d, encoding: .utf8)
    }

    static func delete(account: String) {
        let q: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service, kSecAttrAccount as String: account]
        SecItemDelete(q as CFDictionary)
    }
}
