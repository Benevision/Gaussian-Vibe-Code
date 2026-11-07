import Foundation
import Network
import Darwin
import Observation

/// A capture bundle received from an iPhone (PROTOCOL.md §5).
struct ReceivedCapture: Identifiable, Hashable, Sendable {
    var id = UUID()
    /// `<inbox>/<name>.splatcapture`
    var url: URL
    var name: String
    var receivedAt: Date
    var bytes: Int64
    var sender: String?
}

/// An upload in progress.
struct CaptureTransfer: Identifiable, Hashable, Sendable {
    var id: UUID
    var name: String
    var sender: String?
    var received: Int64
    var expected: Int64?
    var fraction: Double? { expected.map { $0 > 0 ? Double(received) / Double($0) : 0 } }
}

/// Receives `.splatcapture` uploads from Splat Capture (iOS): an HTTP/1.1 server on `NWListener`
/// advertising `_splatcapture._tcp` (TXT `name`, `version=1`). Bodies are streamed to a temp file,
/// unzipped with `ditto` into `inbox`, and published in `received`.
@MainActor
@Observable
final class CaptureReceiver {
    nonisolated static let serviceType = "_splatcapture._tcp"

    enum State: Equatable { case stopped, starting, listening(port: UInt16), failed(String) }

    private(set) var state: State = .stopped
    /// Default port (Splat Capture's manual-entry default). Falls back to a free port if taken.
    nonisolated static let defaultPort: UInt16 = 7879
    /// The bound port while listening (for "host:port" manual entry on the iPhone).
    private(set) var listeningPort: UInt16?
    /// This Mac's LAN IPv4 addresses (refreshed on start), e.g. ["192.168.1.23"].
    private(set) var lanAddresses: [String] = []
    /// Most recent first.
    private(set) var received: [ReceivedCapture] = []
    private(set) var transfers: [UUID: CaptureTransfer] = [:]
    /// Last rejected upload (e.g. wrong pairing code), for UI feedback.
    private(set) var lastError: String?

    /// Where bundles are unpacked. Default: `~/Library/Application Support/SplatStudio/Inbox`.
    var inbox: URL { didSet { server.config.inbox = inbox; loadExisting() } }
    /// When true, uploads must carry `X-Pairing-Code: <pairingCode>` (else 403).
    var requirePairingCode: Bool = true { didSet { server.config.requirePairing = requirePairingCode } }
    private(set) var pairingCode: String
    /// Name advertised in Bonjour TXT and `/v1/hello`.
    var advertisedName: String { didSet { server.config.name = advertisedName } }
    /// Called on the main actor for every saved capture (e.g. to auto-import).
    @ObservationIgnored var onCapture: ((ReceivedCapture) -> Void)?

    @ObservationIgnored private let server: CaptureHTTPServer

    init(inbox: URL? = nil, name: String? = nil) {
        let ib = inbox ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("SplatStudio/Inbox", isDirectory: true)
        let nm = name ?? Host.current().localizedName ?? "Mac"
        let code = Self.makeCode()
        self.inbox = ib
        self.advertisedName = nm
        self.pairingCode = code
        server = CaptureHTTPServer(config: .init(name: nm, inbox: ib, pairingCode: code, requirePairing: true))
        server.events = { [weak self] ev in Task { @MainActor in self?.handle(ev) } }
        loadExisting()
    }

    /// Lists bundles already in the inbox (captures received in earlier sessions), newest first.
    func loadExisting() {
        let fm = FileManager.default
        let keys: [URLResourceKey] = [.creationDateKey, .isDirectoryKey]
        guard let items = try? fm.contentsOfDirectory(at: inbox, includingPropertiesForKeys: keys, options: [.skipsHiddenFiles]) else { return }
        let known = Set(received.map { $0.url.standardizedFileURL })
        var found: [ReceivedCapture] = []
        for u in items where u.pathExtension == "splatcapture" && !known.contains(u.standardizedFileURL) {
            let date = (try? u.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? .distantPast
            found.append(ReceivedCapture(url: u, name: u.deletingPathExtension().lastPathComponent, receivedAt: date,
                                         bytes: Self.directorySize(u), sender: Self.sender(of: u)))
        }
        received = (received + found).sorted { $0.receivedAt > $1.receivedAt }
    }

    /// Removes a capture from the inbox (moves the bundle to the Trash).
    func trash(_ c: ReceivedCapture) {
        try? FileManager.default.trashItem(at: c.url, resultingItemURL: nil)
        received.removeAll { $0.id == c.id }
    }

    nonisolated static func directorySize(_ url: URL) -> Int64 {
        guard let e = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) else { return 0 }
        var total: Int64 = 0
        for case let f as URL in e { total += Int64((try? f.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0) }
        return total
    }

    /// Device name from the bundle's manifest, if it records one.
    nonisolated static func sender(of url: URL) -> String? {
        guard let d = try? Data(contentsOf: url.appendingPathComponent("manifest.json")),
              let j = try? JSONSerialization.jsonObject(with: d) as? [String: Any] else { return nil }
        return j["device"] as? String
    }

    static func makeCode() -> String { String(format: "%06d", Int.random(in: 0...999_999)) }

    /// Generates a new pairing code.
    func regeneratePairingCode() { pairingCode = Self.makeCode(); server.config.pairingCode = pairingCode }

    /// Starts listening on `port` (default 7879; if it is taken a free port is used). 0 = system-assigned.
    /// The port is advertised via Bonjour and published in `listeningPort`.
    func start(port: UInt16 = CaptureReceiver.defaultPort) {
        lanAddresses = Self.currentLANAddresses()
        guard state == .stopped || { if case .failed = state { return true }; return false }() else { return }
        state = .starting
        do {
            try FileManager.default.createDirectory(at: inbox, withIntermediateDirectories: true)
            try server.start(port: port)
        } catch { state = .failed(error.localizedDescription) }
    }

    func stop() { server.stop(); state = .stopped; listeningPort = nil; transfers.removeAll() }

    /// Non-loopback IPv4 addresses of active interfaces (en*, bridge*, …).
    nonisolated static func currentLANAddresses() -> [String] {
        var out: [String] = []
        var ifap: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&ifap) == 0, let first = ifap else { return [] }
        defer { freeifaddrs(ifap) }
        var p: UnsafeMutablePointer<ifaddrs>? = first
        while let i = p {
            defer { p = i.pointee.ifa_next }
            let flags = Int32(i.pointee.ifa_flags)
            guard let sa = i.pointee.ifa_addr, sa.pointee.sa_family == UInt8(AF_INET),
                  flags & IFF_UP != 0, flags & IFF_LOOPBACK == 0 else { continue }
            var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            if getnameinfo(sa, socklen_t(sa.pointee.sa_len), &host, socklen_t(host.count), nil, 0, NI_NUMERICHOST) == 0 {
                let s = String(cString: host)
                if !s.hasPrefix("169.254.") && !out.contains(s) { out.append(s) }
            }
        }
        return out
    }

    private func handle(_ ev: CaptureHTTPServer.Event) {
        switch ev {
        case .listening(let p): state = .listening(port: p); listeningPort = p
        case .failed(let e): state = .failed(e)
        case .transfer(let t): transfers[t.id] = t
        case .transferEnded(let id): transfers.removeValue(forKey: id)
        case .saved(let c):
            received.insert(c, at: 0)
            onCapture?(c)
        case .rejected(let msg): lastError = msg
        }
    }
}

/// The HTTP/1.1 server behind `CaptureReceiver` (thread-safe; runs on its own queue).
final class CaptureHTTPServer: @unchecked Sendable {
    struct Config: Sendable { var name: String; var inbox: URL; var pairingCode: String; var requirePairing: Bool }
    enum Event: Sendable {
        case listening(UInt16), failed(String)
        case transfer(CaptureTransfer), transferEnded(UUID)
        case saved(ReceivedCapture), rejected(String)
    }

    private let lock = NSLock()
    private var _config: Config
    var config: Config {
        get { lock.lock(); defer { lock.unlock() }; return _config }
        set { lock.lock(); _config = newValue; lock.unlock() }
    }
    var events: (@Sendable (Event) -> Void)?
    private var listener: NWListener?
    private let queue = DispatchQueue(label: "SplatStudio.CaptureHTTPServer")
    /// Maximum accepted body size (default 64 GB).
    var maxBodyBytes: Int64 = 64 << 30

    init(config: Config) { _config = config }

    /// The bound port once listening.
    var port: UInt16? { listener?.port?.rawValue }

    func start(port: UInt16) throws {
        do { try startListener(port: port) } catch where port != 0 { try startListener(port: 0) }
    }

    private func startListener(port: UInt16) throws {
        let params = NWParameters.tcp
        params.allowLocalEndpointReuse = true
        params.includePeerToPeer = true
        let l = try NWListener(using: params, on: port == 0 ? .any : NWEndpoint.Port(rawValue: port)!)
        let cfg = config
        var txt = NWTXTRecord()
        txt["name"] = cfg.name
        txt["version"] = "1"
        l.service = NWListener.Service(name: cfg.name, type: CaptureReceiver.serviceType, domain: nil, txtRecord: txt)
        l.stateUpdateHandler = { [weak self] st in
            guard let self else { return }
            switch st {
            case .ready: self.events?(.listening(self.listener?.port?.rawValue ?? 0))
            case .failed(let e):
                // Fixed port taken (EADDRINUSE): fall back to any free port.
                if port != 0 {
                    self.listener?.cancel(); self.listener = nil
                    do { try self.startListener(port: 0) } catch { self.events?(.failed(error.localizedDescription)) }
                } else { self.events?(.failed(e.localizedDescription)) }
            default: break
            }
        }
        l.newConnectionHandler = { [weak self] c in
            guard let self else { c.cancel(); return }
            HTTPConnection(connection: c, server: self).start(queue: self.queue)
        }
        l.start(queue: queue)
        listener = l
    }

    func stop() { listener?.cancel(); listener = nil }

    // MARK: Routing

    struct Request {
        var method: String
        var path: String
        var headers: [String: String]   // lower-cased keys
        var body: URL?                  // temp file
        var bodyBytes: Int64
        var peer: String?
    }
    struct Response { var status: Int; var json: [String: Any] }

    func route(_ r: Request) async -> Response {
        let cfg = config
        switch (r.method, r.path.split(separator: "?").first.map(String.init) ?? r.path) {
        case ("GET", "/v1/hello"):
            return Response(status: 200, json: ["name": cfg.name, "app": "Splat Studio", "version": 1])
        case ("POST", "/v1/captures"):
            if cfg.requirePairing && r.headers["x-pairing-code"]?.trimmingCharacters(in: .whitespaces) != cfg.pairingCode {
                events?(.rejected("Upload from \(r.peer ?? "?") rejected: wrong or missing pairing code"))
                return Response(status: 403, json: ["error": "pairing code required or incorrect"])
            }
            guard let body = r.body, r.bodyBytes > 0 else { return Response(status: 400, json: ["error": "empty body"]) }
            let rawName = r.headers["x-capture-name"].flatMap { $0.removingPercentEncoding } ?? "Capture"
            do {
                let saved = try await save(zip: body, name: rawName, inbox: cfg.inbox)
                events?(.saved(ReceivedCapture(url: saved, name: saved.deletingPathExtension().lastPathComponent, receivedAt: Date(), bytes: r.bodyBytes, sender: r.peer)))
                return Response(status: 200, json: ["ok": true, "savedAs": saved.lastPathComponent])
            } catch {
                events?(.rejected("Could not unpack capture: \(error.localizedDescription)"))
                return Response(status: 400, json: ["error": "could not unpack capture: \(error.localizedDescription)"])
            }
        case (_, "/v1/hello"), (_, "/v1/captures"):
            return Response(status: 405, json: ["error": "method not allowed"])
        default:
            return Response(status: 404, json: ["error": "not found"])
        }
    }

    /// Unzips into a staging dir, then moves the bundle to `<inbox>/<unique name>.splatcapture`.
    /// Accepts zips with the bundle directory as the single top-level entry or with contents at the root.
    func save(zip: URL, name: String, inbox: URL) async throws -> URL {
        let fm = FileManager.default
        let staging = inbox.appendingPathComponent(".incoming-\(UUID().uuidString)")
        defer { try? fm.removeItem(at: staging) }
        try await ZipUtil.unzip(zip, to: staging)
        var bundle: URL
        if fm.fileExists(atPath: staging.appendingPathComponent("manifest.json").path) {
            bundle = staging
        } else {
            let subs = try fm.contentsOfDirectory(at: staging, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
                .filter { $0.lastPathComponent != "__MACOSX" }
            guard let b = subs.first(where: { fm.fileExists(atPath: $0.appendingPathComponent("manifest.json").path) }) else {
                throw PipelineError.invalidData("zip has no manifest.json")
            }
            bundle = b
        }
        let safe = Self.sanitize(name.hasSuffix(".splatcapture") ? String(name.dropLast(13)) : name)
        var dst = inbox.appendingPathComponent("\(safe).splatcapture")
        var n = 2
        while fm.fileExists(atPath: dst.path) { dst = inbox.appendingPathComponent("\(safe) \(n).splatcapture"); n += 1 }
        try fm.moveItem(at: bundle, to: dst)
        return dst
    }

    static func sanitize(_ s: String) -> String {
        let bad = CharacterSet(charactersIn: "/\\:?%*|\"<>\0").union(.controlCharacters)
        let cleaned = s.components(separatedBy: bad).joined(separator: "_").trimmingCharacters(in: .whitespacesAndNewlines)
        let noDots = cleaned.trimmingCharacters(in: CharacterSet(charactersIn: "."))
        return noDots.isEmpty ? "Capture" : String(noDots.prefix(120))
    }
}

/// One HTTP/1.1 connection: parses the request head, streams the body (Content-Length or chunked) to a
/// temp file, routes it, writes the response, then serves further keep-alive requests.
private final class HTTPConnection: @unchecked Sendable {
    let c: NWConnection
    let server: CaptureHTTPServer
    var buffer = Data()
    var queue: DispatchQueue!
    private var selfRef: HTTPConnection?

    init(connection: NWConnection, server: CaptureHTTPServer) { c = connection; self.server = server }

    func start(queue: DispatchQueue) {
        self.queue = queue
        selfRef = self
        c.stateUpdateHandler = { [weak self] st in
            switch st {
            case .failed, .cancelled: self?.selfRef = nil
            default: break
            }
        }
        c.start(queue: queue)
        readHead()
    }

    var peer: String? {
        if case let .hostPort(h, _) = c.endpoint { return "\(h)".components(separatedBy: "%").first }
        return nil
    }

    func close() { c.cancel(); selfRef = nil }

    private func receive(_ cb: @escaping (Data?, Bool) -> Void) {
        c.receive(minimumIncompleteLength: 1, maximumLength: 1 << 20) { data, _, done, err in
            cb(data, done || err != nil)
        }
    }

    private func readHead() {
        if let r = buffer.range(of: Data("\r\n\r\n".utf8)) {
            let head = String(decoding: buffer[buffer.startIndex..<r.lowerBound], as: UTF8.self)
            buffer.removeSubrange(buffer.startIndex..<r.upperBound)
            handleHead(head)
            return
        }
        if buffer.count > 64 * 1024 { respond(431, ["error": "headers too large"], keepAlive: false); return }
        receive { [weak self] d, done in
            guard let self else { return }
            if let d { self.buffer.append(d) }
            if done && (d == nil || d!.isEmpty) && self.buffer.range(of: Data("\r\n\r\n".utf8)) == nil { self.close(); return }
            self.readHead()
        }
    }

    private func handleHead(_ head: String) {
        let lines = head.components(separatedBy: "\r\n")
        let parts = lines.first?.split(separator: " ") ?? []
        guard parts.count >= 2 else { respond(400, ["error": "bad request line"], keepAlive: false); return }
        var headers: [String: String] = [:]
        for l in lines.dropFirst() {
            guard let i = l.firstIndex(of: ":") else { continue }
            headers[l[..<i].trimmingCharacters(in: .whitespaces).lowercased()] = l[l.index(after: i)...].trimmingCharacters(in: .whitespaces)
        }
        let keepAlive = (headers["connection"]?.lowercased() ?? "keep-alive") != "close"
        let method = String(parts[0]), path = String(parts[1])
        let chunked = headers["transfer-encoding"]?.lowercased().contains("chunked") == true
        let length = Int64(headers["content-length"] ?? "") ?? 0
        if length > server.maxBodyBytes { respond(413, ["error": "body too large"], keepAlive: false); return }
        // Reject unauthorised uploads before receiving gigabytes.
        let cfg = server.config
        if method == "POST", path.hasPrefix("/v1/captures"), cfg.requirePairing,
           headers["x-pairing-code"]?.trimmingCharacters(in: .whitespaces) != cfg.pairingCode {
            server.events?(.rejected("Upload from \(peer ?? "?") rejected: wrong or missing pairing code"))
            respond(403, ["error": "pairing code required or incorrect"], keepAlive: false)
            return
        }
        if headers["expect"]?.lowercased() == "100-continue" {
            c.send(content: Data("HTTP/1.1 100 Continue\r\n\r\n".utf8), completion: .contentProcessed { _ in })
        }
        var req = CaptureHTTPServer.Request(method: method, path: path, headers: headers, body: nil, bodyBytes: 0, peer: peer)
        guard length > 0 || chunked else { dispatch(req, keepAlive: keepAlive); return }

        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("splatcapture-\(UUID().uuidString).zip")
        FileManager.default.createFile(atPath: tmp.path, contents: nil)
        guard let fh = try? FileHandle(forWritingTo: tmp) else { respond(500, ["error": "cannot create temp file"], keepAlive: false); return }
        let tid = UUID()
        let name = headers["x-capture-name"].flatMap { $0.removingPercentEncoding } ?? "Capture"
        var transfer = CaptureTransfer(id: tid, name: name, sender: peer, received: 0, expected: chunked ? nil : length)
        var lastReport = Date.distantPast
        let report: (Int64) -> Void = { [weak self] n in
            transfer.received = n
            if Date().timeIntervalSince(lastReport) > 0.25 { lastReport = Date(); self?.server.events?(.transfer(transfer)) }
        }
        let finish: (Bool, Int64) -> Void = { [weak self] ok, total in
            try? fh.close()
            guard let self else { try? FileManager.default.removeItem(at: tmp); return }
            self.server.events?(.transferEnded(tid))
            guard ok else { try? FileManager.default.removeItem(at: tmp); self.close(); return }
            req.body = tmp; req.bodyBytes = total
            self.dispatch(req, keepAlive: keepAlive, cleanup: tmp)
        }
        server.events?(.transfer(transfer))
        if chunked { readChunked(fh, total: 0, report: report, finish: finish) }
        else { readBody(fh, remaining: length, total: 0, report: report, finish: finish) }
    }

    private func readBody(_ fh: FileHandle, remaining: Int64, total: Int64, report: @escaping (Int64) -> Void, finish: @escaping (Bool, Int64) -> Void) {
        var remaining = remaining, total = total
        if !buffer.isEmpty {
            let n = Int(min(Int64(buffer.count), remaining))
            fh.write(buffer.prefix(n))
            buffer.removeFirst(n)
            remaining -= Int64(n); total += Int64(n)
            report(total)
        }
        if remaining == 0 { finish(true, total); return }
        receive { [weak self] d, done in
            guard let self else { return }
            if let d, !d.isEmpty { self.buffer.append(d) }
            if done && self.buffer.isEmpty { finish(false, total); return }
            self.readBody(fh, remaining: remaining, total: total, report: report, finish: finish)
        }
    }

    /// Chunked transfer decoding (streams chunks to the file).
    private func readChunked(_ fh: FileHandle, total: Int64, report: @escaping (Int64) -> Void, finish: @escaping (Bool, Int64) -> Void) {
        var total = total
        while true {
            guard let crlf = buffer.range(of: Data("\r\n".utf8)) else { break }
            let sizeLine = String(decoding: buffer[buffer.startIndex..<crlf.lowerBound], as: UTF8.self)
            guard let size = Int(sizeLine.split(separator: ";").first?.trimmingCharacters(in: .whitespaces) ?? "", radix: 16) else { finish(false, total); return }
            let needed = (crlf.upperBound - buffer.startIndex) + size + 2
            if size == 0 {
                // Trailer section ends with an empty line.
                if let end = buffer.range(of: Data("\r\n\r\n".utf8), in: (crlf.lowerBound)..<buffer.endIndex) {
                    buffer.removeSubrange(buffer.startIndex..<end.upperBound); finish(true, total); return
                }
                if buffer.count >= needed { buffer.removeFirst(needed); finish(true, total); return }
                break
            }
            guard buffer.count >= needed else { break }
            let start = crlf.upperBound
            fh.write(buffer[start..<(start + size)])
            buffer.removeFirst(needed)
            total += Int64(size)
            if total > server.maxBodyBytes { finish(false, total); return }
            report(total)
        }
        receive { [weak self] d, done in
            guard let self else { return }
            if let d, !d.isEmpty { self.buffer.append(d) } else if done { finish(false, total); return }
            self.readChunked(fh, total: total, report: report, finish: finish)
        }
    }

    private func dispatch(_ req: CaptureHTTPServer.Request, keepAlive: Bool, cleanup: URL? = nil) {
        let server = self.server
        Task {
            let resp = await server.route(req)
            if let cleanup { try? FileManager.default.removeItem(at: cleanup) }
            self.queue.async { self.respond(resp.status, resp.json, keepAlive: keepAlive) }
        }
    }

    private func respond(_ status: Int, _ json: [String: Any], keepAlive: Bool) {
        let body = (try? JSONSerialization.data(withJSONObject: json, options: [.sortedKeys])) ?? Data("{}".utf8)
        let reason = [200: "OK", 400: "Bad Request", 403: "Forbidden", 404: "Not Found", 405: "Method Not Allowed",
                      413: "Payload Too Large", 431: "Request Header Fields Too Large", 500: "Internal Server Error"][status] ?? "Status"
        var head = "HTTP/1.1 \(status) \(reason)\r\nContent-Type: application/json\r\nContent-Length: \(body.count)\r\n"
        head += keepAlive ? "Connection: keep-alive\r\n\r\n" : "Connection: close\r\n\r\n"
        var out = Data(head.utf8); out.append(body)
        c.send(content: out, completion: .contentProcessed { [weak self] _ in
            guard let self else { return }
            if keepAlive { self.readHead() } else { self.close() }
        })
    }
}


/// Which projects were made from which inbox capture (so the Inbox can list a capture's splats).
enum CaptureProjects {
    private static let key = "captureProjects"

    static func record(capture: URL, project: URL) {
        var m = UserDefaults.standard.dictionary(forKey: key) as? [String: [String]] ?? [:]
        var list = m[capture.standardizedFileURL.path] ?? []
        if !list.contains(project.standardizedFileURL.path) { list.append(project.standardizedFileURL.path) }
        m[capture.standardizedFileURL.path] = list
        UserDefaults.standard.set(m, forKey: key)
    }

    /// Recorded projects that still exist, plus `~/Documents/Splat Studio/<capture name>*` folders made before
    /// links were recorded.
    static func projects(for capture: ReceivedCapture) -> [URL] {
        let fm = FileManager.default
        let m = UserDefaults.standard.dictionary(forKey: key) as? [String: [String]] ?? [:]
        var out = (m[capture.url.standardizedFileURL.path] ?? []).map { URL(fileURLWithPath: $0, isDirectory: true) }
        let docs = fm.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("Splat Studio", isDirectory: true)
        if let items = try? fm.contentsOfDirectory(at: docs, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]) {
            for u in items where u.lastPathComponent == capture.name && !out.contains(where: { $0.standardizedFileURL == u.standardizedFileURL }) {
                out.append(u)
            }
        }
        return out.filter { fm.fileExists(atPath: $0.path) }
    }

    /// Trained splats in a project, newest first.
    static func splats(in project: URL) -> [URL] {
        let dir = project.appendingPathComponent("outputs", isDirectory: true)
        let items = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.contentModificationDateKey], options: [.skipsHiddenFiles])) ?? []
        func date(_ u: URL) -> Date { (try? u.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast }
        return items.filter { ["ply", "spz", "splat"].contains($0.pathExtension.lowercased()) }.sorted { date($0) > date($1) }
    }
}
