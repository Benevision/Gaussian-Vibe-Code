import SwiftUI

/// "Install on <worker>" with live progress: used where a chosen model is missing on the Linux worker
/// (New Project) and in the Compute panel. The worker runs the same pip steps as its setup assistant's
/// "Processing models" option; models then appear in its /v1/info once installed.
struct WorkerModelInstallView: View {
    let workerId: String
    /// Shown above the button when set, e.g. "BensPC doesn't have SAM 2 yet."
    var reason: String? = nil
    private var browser = WorkerBrowser.shared
    @State private var status: WorkerClient.RuntimeStatus?
    @State private var error: String?
    @State private var polling = false
    @State private var showLog = false

    init(workerId: String, reason: String? = nil) { self.workerId = workerId; self.reason = reason }

    private var name: String { browser.workers.first { $0.id == workerId }?.name ?? "the worker" }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let reason, status?.state != "running" && status?.state != "done" {
                Label(reason, systemImage: "exclamationmark.triangle").font(.callout).foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
            switch status?.state {
            case "running":
                ProgressView(value: status?.fraction ?? 0) {
                    Text("Installing on \(name): \(status?.message ?? "")").font(.callout)
                }
                Text("About 300 MB; model weights (1–2 GB) download on first use. You can keep working.")
                    .font(.caption).foregroundStyle(.secondary)
            case "done":
                Label("Processing models installed on \(name).", systemImage: "checkmark.circle.fill").foregroundStyle(.green)
            case "failed":
                Label("Install on \(name) failed: \(status?.message ?? "unknown error")", systemImage: "xmark.octagon").foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
                installButton(title: "Try Again")
            default:
                installButton(title: "Install Processing Models on \(name)")
            }
            if let error { Text(error).font(.caption).foregroundStyle(.red).fixedSize(horizontal: false, vertical: true) }
            if let log = status?.log, !log.isEmpty, status?.state != "idle" {
                DisclosureGroup("Details", isExpanded: $showLog) {
                    ScrollView { Text(log.suffix(30).joined(separator: "\n")).font(.caption2.monospaced()).frame(maxWidth: .infinity, alignment: .leading).textSelection(.enabled) }
                        .frame(height: 110)
                }.font(.caption)
            }
        }
        .task(id: workerId) { await poll(once: true) }
    }

    private func installButton(title: String) -> some View {
        Button(title) {
            Task {
                guard let c = browser.client(for: workerId) else { error = "\(name) is offline."; return }
                do { status = try await c.installRuntime(); error = nil; await poll(once: false) }
                catch { self.error = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription }
            }
        }
    }

    /// Polls while an install runs (also picks up one started elsewhere, e.g. from the PC itself).
    private func poll(once: Bool) async {
        guard !polling || once else { return }
        polling = true; defer { polling = false }
        repeat {
            guard let c = browser.client(for: workerId), let s = try? await c.runtimeStatus() else { return }
            status = s
            if s.state != "running" {
                if s.state == "done" { await browser.checkHealth(workerId) }   // refresh the model list
                return
            }
            try? await Task.sleep(nanoseconds: 1_000_000_000)
        } while !Task.isCancelled
    }
}
