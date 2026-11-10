import Foundation
import Network
import os

/// A parsed HTTP/1.1 request. Large bodies (dataset uploads) arrive in `bodyFile` instead of `body`.
struct HTTPRequest: Sendable {
    var method: String
    /// Path without the query string, percent-decoded, e.g. `/v1/jobs/j1`.
    var path: String
    var query: [String: String] = [:]
    /// Lower-cased header names.
    var headers: [String: String] = [:]
    var body = Data()
    var bodyFile: URL? = nil
}

struct HTTPResponse: Sendable {
    enum Body: Sendable {
        case data(Data)
        /// Streams a file; `remove` deletes it after sending.
        case file(URL, remove: Bool)
    }
    var status: Int
    var contentType = "application/json"
    var headers: [String: String] = [:]
    var body: Body = .data(Data())

    static func json<T: Encodable>(_ value: T, status: Int = 200) -> HTTPResponse {
        do { return HTTPResponse(status: status, body: .data(try JSONEncoder().encode(value))) } catch {
            return .error(500, "encoding failed: \(error.localizedDescription)")
        }
    }
    /// `{"error": "message"}` (PROTOCOL.md §4).
    static func error(_ status: Int, _ message: String) -> HTTPResponse { json(["error": message], status: status) }
    static func text(_ s: String, status: Int = 200) -> HTTPResponse {
        HTTPResponse(status: status, contentType: "text/plain; charset=utf-8", body: .data(Data(s.utf8)))
    }
    static let noContent = HTTPResponse(status: 204, body: .data(Data()))

    static func reason(_ status: Int) -> String {
        switch status {
        case 100: "Continue"; case 200: "OK"; case 201: "Created"; case 204: "No Content"
        case 400: "Bad Request"; case 401: "Unauthorized"; case 403: "Forbidden"; case 404: "Not Found"
        case 405: "Method Not Allowed"; case 409: "Conflict"; case 411: "Length Required"; case 413: "Payload Too Large"
        case 415: "Unsupported Media Type"; case 431: "Request Header Fields Too Large"; case 500: "Internal Server Error"
        case 503: "Service Unavailable"; case 507: "Insufficient Storage"
        default: "Status \(status)"
        }
    }
}

/// What the server needs to know about a request before reading its body.
struct HTTPBodyPolicy: Sendable {
    /// Stream the body to a temporary file (uploads) instead of memory.
    var toFile: Bool
    /// Largest accepted body.
    var maxBytes: UInt64
    /// Refuse before reading anything (e.g. not enough disk space). nil = accept.
    var reject: HTTPResponse? = nil
}

/// Minimal HTTP/1.1 server on `NWListener` (one request per connection, `Connection: close`), plus Bonjour.
/// Handlers run on a background queue and may block (unzipping an upload, reading a file).
final class HTTPServer: @unchecked Sendable {
    typealias Handler = @Sendable (HTTPRequest) -> HTTPResponse
    /// `headers` as in `HTTPRequest.headers` (lower-cased names), e.g. to refuse an unauthenticated request before its body.
    typealias PolicyProvider = @Sendable (_ method: String, _ path: String, _ headers: [String: String], _ contentLength: UInt64?) -> HTTPBodyPolicy

    enum State: Sendable, Equatable { case stopped, starting, ready(port: UInt16), failed(String) }

    private let port: UInt16
    private let service: NWListener.Service?
    private let handler: Handler
    private let policy: PolicyProvider
    private let tempDir: URL
    private let queue = DispatchQueue(label: "splatworker.http")
    private var listener: NWListener?
    private let onState: @Sendable (State) -> Void
    static let log = Logger(subsystem: WorkerEngine.logSubsystem, category: "worker-http")

    init(port: UInt16, service: NWListener.Service?, tempDir: URL, policy: @escaping PolicyProvider,
         onState: @escaping @Sendable (State) -> Void, handler: @escaping Handler) {
        self.port = port; self.service = service; self.tempDir = tempDir
        self.policy = policy; self.onState = onState; self.handler = handler
    }

    func start() {
        queue.async { [self] in
            guard listener == nil else { return }
            onState(.starting)
            do {
                let params = NWParameters.tcp
                params.allowLocalEndpointReuse = true
                params.includePeerToPeer = false
                let l = try NWListener(using: params, on: NWEndpoint.Port(rawValue: port)!)
                l.service = service
                l.stateUpdateHandler = { [weak self] s in
                    guard let self else { return }
                    switch s {
                    case .ready: self.onState(.ready(port: self.port))
                    case .failed(let e):
                        self.onState(.failed(e.localizedDescription))
                        self.queue.async { self.listener?.cancel(); self.listener = nil }
                    case .waiting(let e): self.onState(.failed("waiting: \(e.localizedDescription)"))
                    case .cancelled: self.onState(.stopped)
                    default: break
                    }
                }
                l.newConnectionHandler = { [weak self] c in
                    guard let self else { c.cancel(); return }
                    HTTPConnection(connection: c, server: self).start()
                }
                listener = l
                l.start(queue: queue)
            } catch {
                onState(.failed(error.localizedDescription))
            }
        }
    }

    func stop() {
        queue.async { [self] in
            listener?.cancel()
            listener = nil
        }
    }

    fileprivate func bodyPolicy(_ method: String, _ path: String, _ headers: [String: String], _ len: UInt64?) -> HTTPBodyPolicy {
        policy(method, path, headers, len)
    }
    fileprivate func handle(_ r: HTTPRequest) -> HTTPResponse { handler(r) }
    fileprivate func newTempFile() -> URL {
        try? FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        return tempDir.appendingPathComponent("upload-\(UUID().uuidString).bin")
    }

    // MARK: parsing (pure, used by HTTPConnection)

    /// Parses the request line and headers (everything before the blank line).
    static func parseHead(_ head: Data) -> HTTPRequest? {
        guard let s = String(data: head, encoding: .utf8) ?? String(data: head, encoding: .isoLatin1) else { return nil }
        var lines = s.components(separatedBy: "\r\n")
        guard !lines.isEmpty else { return nil }
        let parts = lines.removeFirst().split(separator: " ", omittingEmptySubsequences: true)
        guard parts.count >= 2 else { return nil }
        let target = String(parts[1])
        var req = HTTPRequest(method: parts[0].uppercased(), path: target)
        let comps = URLComponents(string: target.hasPrefix("/") ? "http://x" + target : target)
        req.path = comps?.percentEncodedPath.removingPercentEncoding ?? target
        if req.path.isEmpty { req.path = "/" }
        for q in comps?.queryItems ?? [] { req.query[q.name] = q.value ?? "" }
        for line in lines where !line.isEmpty {
            guard let c = line.firstIndex(of: ":") else { continue }
            req.headers[line[..<c].trimmingCharacters(in: .whitespaces).lowercased()] = line[line.index(after: c)...].trimmingCharacters(in: .whitespaces)
        }
        return req
    }
}

/// One client connection: read head → read body (memory or file) → handler → write response → close.
private final class HTTPConnection: @unchecked Sendable {
    private let conn: NWConnection
    private let server: HTTPServer
    private let queue = DispatchQueue(label: "splatworker.conn")
    private var buffer = Data()
    private var request: HTTPRequest?
    private var expected: UInt64 = 0
    private var received: UInt64 = 0
    private var file: FileHandle?
    private var fileURL: URL?
    private var keep: HTTPConnection?   // self-retain until closed

    static let maxHead = 64 * 1024

    init(connection: NWConnection, server: HTTPServer) { self.conn = connection; self.server = server }

    func start() {
        keep = self
        conn.stateUpdateHandler = { [weak self] s in
            switch s {
            case .failed, .cancelled:
                guard let self else { return }
                self.queue.async { self.cleanup() }
            default: break
            }
        }
        conn.start(queue: queue)
        receiveHead()
    }

    private func cleanup() {
        try? file?.close(); file = nil
        if let fileURL { try? FileManager.default.removeItem(at: fileURL); self.fileURL = nil }
        keep = nil
    }

    private func receiveHead() {
        conn.receive(minimumIncompleteLength: 1, maximumLength: 64 * 1024) { [self] data, _, done, error in
            if let data { buffer.append(data) }
            if let r = buffer.range(of: Data("\r\n\r\n".utf8)) {
                let head = buffer.subdata(in: buffer.startIndex..<r.lowerBound)
                let rest = buffer.subdata(in: r.upperBound..<buffer.endIndex)
                buffer = Data()
                gotHead(head, rest: rest)
            } else if buffer.count > Self.maxHead {
                respond(.error(431, "request head too large"))
            } else if done || error != nil {
                conn.cancel()
            } else {
                receiveHead()
            }
        }
    }

    private func gotHead(_ head: Data, rest: Data) {
        guard var req = HTTPServer.parseHead(head) else { respond(.error(400, "malformed request")); return }
        if req.headers["transfer-encoding"]?.lowercased().contains("chunked") == true {
            respond(.error(411, "chunked bodies are not supported; send Content-Length")); return
        }
        let len = req.headers["content-length"].flatMap { UInt64($0) }
        let pol = server.bodyPolicy(req.method, req.path, req.headers, len)
        if let reject = pol.reject { respond(reject); return }
        expected = len ?? 0
        guard expected <= pol.maxBytes else { respond(.error(413, "body exceeds \(pol.maxBytes) bytes")); return }
        if pol.toFile {
            let u = server.newTempFile()
            FileManager.default.createFile(atPath: u.path, contents: nil)
            guard let fh = try? FileHandle(forWritingTo: u) else { respond(.error(507, "cannot create upload file")); return }
            file = fh; fileURL = u; req.bodyFile = u
        }
        request = req
        if req.headers["expect"]?.lowercased() == "100-continue" {
            conn.send(content: Data("HTTP/1.1 100 Continue\r\n\r\n".utf8), completion: .contentProcessed { _ in })
        }
        consume(rest)
        if received >= expected { finishBody() } else { receiveBody() }
    }

    private func consume(_ d: Data) {
        guard !d.isEmpty else { return }
        let take = d.prefix(Int(min(UInt64(d.count), expected - received)))
        received += UInt64(take.count)
        if let file {
            do { try file.write(contentsOf: take) } catch { respond(.error(507, "write failed: \(error.localizedDescription)")); return }
        } else {
            request?.body.append(take)
        }
    }

    private func receiveBody() {
        conn.receive(minimumIncompleteLength: 1, maximumLength: 1 << 20) { [self] data, _, done, error in
            if let data { consume(data) }
            if received >= expected { finishBody() }
            else if done || error != nil { cleanup(); conn.cancel() }   // client went away mid-upload
            else { receiveBody() }
        }
    }

    private func finishBody() {
        try? file?.close(); file = nil
        guard let req = request else { respond(.error(400, "no request")); return }
        fileURL = nil   // the handler owns the upload now (it moves or deletes it)
        let server = server
        DispatchQueue.global(qos: .userInitiated).async { [self] in
            let resp = server.handle(req)
            queue.async { [self] in respond(resp) }
        }
    }

    private func respond(_ r: HTTPResponse) {
        var length: UInt64 = 0
        switch r.body {
        case .data(let d): length = UInt64(d.count)
        case .file(let u, _): length = (try? FileManager.default.attributesOfItem(atPath: u.path)[.size] as? NSNumber)?.uint64Value ?? 0
        }
        var head = "HTTP/1.1 \(r.status) \(HTTPResponse.reason(r.status))\r\n"
        var headers = r.headers
        headers["Content-Type"] = r.contentType
        headers["Content-Length"] = String(length)
        headers["Connection"] = "close"
        headers["Server"] = "SplatWorker/\(WorkerEngine.kind)"
        for (k, v) in headers.sorted(by: { $0.key < $1.key }) { head += "\(k): \(v)\r\n" }
        head += "\r\n"
        switch r.body {
        case .data(let d):
            var out = Data(head.utf8); out.append(d)
            conn.send(content: out, contentContext: .finalMessage, isComplete: true, completion: .contentProcessed { [self] _ in close() })
        case .file(let u, let remove):
            guard let fh = try? FileHandle(forReadingFrom: u) else {
                respond(.error(404, "file vanished")); return
            }
            conn.send(content: Data(head.utf8), completion: .contentProcessed { [self] e in
                if e != nil { try? fh.close(); close() } else { sendChunks(fh, url: u, remove: remove) }
            })
        }
    }

    private func sendChunks(_ fh: FileHandle, url: URL, remove: Bool) {
        let chunk = (try? fh.read(upToCount: 1 << 20)) ?? nil
        if let chunk, !chunk.isEmpty {
            conn.send(content: chunk, completion: .contentProcessed { [self] e in
                if e != nil { try? fh.close(); if remove { try? FileManager.default.removeItem(at: url) }; close() }
                else { sendChunks(fh, url: url, remove: remove) }
            })
        } else {
            try? fh.close()
            if remove { try? FileManager.default.removeItem(at: url) }
            conn.send(content: nil, contentContext: .finalMessage, isComplete: true, completion: .contentProcessed { [self] _ in close() })
        }
    }

    private func close() {
        conn.cancel()
        cleanup()
    }
}
