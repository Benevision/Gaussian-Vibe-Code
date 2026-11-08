import SwiftUI

/// "Compute": lets Splat Studio on a Mac send training blocks to this iPhone.
struct ComputeView: View {
    @Environment(ComputeWorker.self) private var worker
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Toggle("Train for Splat Studio on this iPhone", isOn: Binding(
                        get: { worker.isEnabled }, set: { worker.setEnabled($0) }))
                } footer: {
                    Text("Splat Studio on your Mac finds this iPhone under Compute › Workers and can send it a block of a large scene to train with the iPhone's GPU. Keep the app open and the phone plugged in: iOS stops GPU work as soon as the app leaves the screen, and the job then fails.")
                }

                if worker.isEnabled {
                    Section("Status") {
                        LabeledContent("Server", value: serverText)
                        if !worker.addresses.isEmpty {
                            LabeledContent("Address", value: worker.addresses.map { "\($0):\(ComputeWorker.port)" }.joined(separator: ", "))
                        }
                        LabeledContent("GPU", value: worker.gpuName)
                        LabeledContent("Memory", value: ByteCountFormatter.string(fromByteCount: Int64(ProcessInfo.processInfo.physicalMemory), countStyle: .memory))
                    }

                    if let job = worker.activeJob {
                        Section("Training") { JobProgressView(job: job, onCancel: { worker.cancel(job.jobId) }) }
                    }

                    if !worker.lastLog.isEmpty {
                        Section("Log") {
                            ForEach(Array(worker.lastLog.enumerated()), id: \.offset) { _, line in
                                Text(line).font(.caption.monospaced()).foregroundStyle(.secondary)
                            }
                        }
                    }

                    let done = worker.jobs.filter(\.isTerminal).reversed()
                    if !done.isEmpty {
                        Section("Finished") {
                            ForEach(Array(done)) { j in
                                VStack(alignment: .leading, spacing: 2) {
                                    HStack {
                                        Text(j.name).lineLimit(1)
                                        Spacer()
                                        Text(j.state.capitalized).foregroundStyle(j.state == "completed" ? .green : .orange)
                                    }
                                    if let m = j.message, j.state != "completed" { Text(m).font(.caption).foregroundStyle(.secondary) }
                                    if j.state == "completed", let n = j.numGaussians {
                                        Text("\(n.formatted()) splats · \(Self.duration(j.elapsedSeconds ?? 0))").font(.caption).foregroundStyle(.secondary)
                                    }
                                }
                            }
                        }
                    }
                }
            }
            .navigationTitle("Compute")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
        }
    }

    private var serverText: String {
        switch worker.serverState {
        case .stopped: "Stopped"
        case .starting: "Starting…"
        case .ready(let p): "Listening on port \(p)"
        case .failed(let e): "Error: \(e)"
        }
    }

    static func duration(_ s: Double) -> String {
        let t = Int(s)
        return t >= 3600 ? String(format: "%d:%02d:%02d", t / 3600, t / 60 % 60, t % 60) : String(format: "%d:%02d", t / 60, t % 60)
    }
}

struct JobProgressView: View {
    let job: WorkerJobStatus
    let onCancel: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(job.name).font(.headline).lineLimit(2)
            ProgressView(value: job.fraction)
            HStack {
                Text(job.state == "training" ? "Step \(job.step.formatted()) / \(job.totalSteps.formatted())" : (job.message ?? job.state.capitalized))
                Spacer()
                if let eta = job.etaSeconds, job.state == "training" { Text("\(ComputeView.duration(eta)) left") }
            }
            .font(.caption.monospacedDigit())
            .foregroundStyle(.secondary)
            HStack(spacing: 16) {
                if let n = job.numGaussians { Text("\(n.formatted()) splats") }
                if let p = job.psnr, p > 0 { Text(String(format: "PSNR %.2f dB", p)) }
                if let e = job.elapsedSeconds { Text(ComputeView.duration(e)) }
            }
            .font(.caption.monospacedDigit())
            Button("Cancel Job", role: .destructive, action: onCancel)
        }
        .padding(.vertical, 4)
    }
}
