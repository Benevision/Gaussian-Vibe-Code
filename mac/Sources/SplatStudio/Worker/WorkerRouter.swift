import Foundation

/// The subset of the SplatWork Worker API (docs/PROTOCOL.md §4, §7) this iPhone implements — what SplatWork's
/// `WorkerClient` uses for training. Pure: `match` only looks at the method and path.
enum WorkerRoute: Equatable, Sendable {
    case info                       // GET    /v1/info
    case uploadDataset              // POST   /v1/datasets            (zip body)
    case listDatasets               // GET    /v1/datasets
    case deleteDataset(String)      // DELETE /v1/datasets/{id}
    case createJob                  // POST   /v1/jobs                {datasetId, name, config}
    case listJobs                   // GET    /v1/jobs
    case job(String)                // GET    /v1/jobs/{id}
    case cancelJob(String)          // POST   /v1/jobs/{id}/cancel
    case deleteJob(String)          // DELETE /v1/jobs/{id}
    case result(String)             // GET    /v1/jobs/{id}/result.ply
    case log(String)                // GET    /v1/jobs/{id}/log
    case cameras(String)            // GET    /v1/jobs/{id}/cameras
    case preview(String)            // GET    /v1/jobs/{id}/preview.splat | preview.ply  (always 404: no live preview)
    /// Known path, wrong method.
    case methodNotAllowed
    /// Anything else (/v1/process, /v1/runtime, /v1/datasets/compose, …): 404, which WorkerClient treats as
    /// "not supported here" (e.g. compose → falls back to a full upload).
    case notFound

    static func match(method: String, path: String) -> WorkerRoute {
        var p = path
        while p.count > 1 && p.hasSuffix("/") { p.removeLast() }
        let s = p.split(separator: "/", omittingEmptySubsequences: true).map(String.init)
        guard s.count >= 2, s[0] == "v1" else { return .notFound }
        let m = method.uppercased()
        func only(_ allowed: String, _ r: WorkerRoute) -> WorkerRoute { m == allowed ? r : .methodNotAllowed }
        switch (s.count, s[1]) {
        case (2, "info"): return only("GET", .info)
        case (2, "datasets"):
            return m == "POST" ? .uploadDataset : m == "GET" ? .listDatasets : .methodNotAllowed
        case (3, "datasets"):
            return s[2] == "compose" || s[2] == "file" ? .notFound : only("DELETE", .deleteDataset(s[2]))
        case (2, "jobs"):
            return m == "POST" ? .createJob : m == "GET" ? .listJobs : .methodNotAllowed
        case (3, "jobs"):
            return m == "GET" ? .job(s[2]) : m == "DELETE" ? .deleteJob(s[2]) : .methodNotAllowed
        case (4, "jobs"):
            let id = s[2]
            switch s[3] {
            case "cancel": return only("POST", .cancelJob(id))
            case "result.ply": return only("GET", .result(id))
            case "log": return only("GET", .log(id))
            case "cameras": return only("GET", .cameras(id))
            case "preview.splat", "preview.ply": return only("GET", .preview(id))
            default: return .notFound
            }
        default: return .notFound
        }
    }
}
