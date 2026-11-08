import SplatTrainKit
import SwiftUI

/// "Make splat": options + device checkboxes → progress → the result in the viewer.
struct MakeSplatView: View {
    let item: CaptureItem
    @Environment(SplatJobs.self) private var jobs
    @Environment(ComputeWorker.self) private var computeWorker
    @Environment(\.dismiss) private var dismiss
    @State private var devices = SplatDevices.shared
    @State private var viewing: SplatOutput?
    @State private var manualHost = ""
    @State private var confirmStop = false

    private var state: SplatJobs.State { jobs.state(for: item.url) }
    private var existing: SplatOutput? { let o = SplatOutput.for(item.url); return o.exists ? o : nil }
    private var busyElsewhere: Bool { jobs.isRunning && jobs.capture?.standardizedFileURL != item.url.standardizedFileURL }

    var body: some View {
        NavigationStack {
            Form {
                switch state {
                case .running(let p): running(p)
                case .finished(let out, let n, let s): finished(out, splats: n, seconds: s)
                case .failed(let m):
                    Section { Label(m, systemImage: "exclamationmark.triangle").foregroundStyle(.orange) }
                    setup
                case .idle:
                    if let existing { Section { Button { viewing = existing } label: { Label("View the last splat", systemImage: "cube.transparent") } } }
                    setup
                }
            }
            .navigationTitle(item.name)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button(jobs.isRunning ? "Hide" : "Close") { dismiss() } }
            }
            .fullScreenCover(item: $viewing) { SplatViewerScreen(output: $0, title: item.name) }
            .confirmationDialog("Stop making this splat?", isPresented: $confirmStop, titleVisibility: .visible) {
                Button("Stop", role: .destructive) { jobs.cancel() }
            }
            .onAppear {
                devices.excludedNames = computeWorker.isEnabled ? [DeviceInfo.modelName] : []
                devices.start()
            }
        }
    }

    // MARK: setup

    @ViewBuilder private var setup: some View {
        @Bindable var jobs = jobs
        Section {
            Picker("Quality", selection: $jobs.options.quality) {
                ForEach(SplatQuality.allCases) { Text($0.title).tag($0) }
            }
            .pickerStyle(.segmented)
            Text(jobs.options.quality.subtitle).font(.caption).foregroundStyle(.secondary)
            Picker("Masks", selection: $jobs.options.masks) {
                ForEach(SplatMaskMode.allCases) { Text($0.title).tag($0) }
            }
        } header: { Text("Splat") } footer: {
            Text("Masks run on this iPhone with Apple Vision. Masked pixels are ignored while training (people walking through, or everything but the object).")
        }

        Section {
            Toggle("Refine camera poses", isOn: $jobs.options.refinePoses)
            Toggle("Learned moving-object masks", isOn: $jobs.options.learnedMovingMasks)
            Toggle("Sky masks + skybox", isOn: $jobs.options.sky)
        } header: { Text("With a SplatWork Node PC") } footer: {
            Text(devices.anyProcessor
                 ? "A SplatWork Node PC is online: these steps run there (the photos are uploaded once)."
                 : "No SplatWork Node PC is online, so these are skipped and everything runs on the devices below — this iPhone alone is enough.")
        }
        .disabled(!devices.anyProcessor)

        Section {
            ForEach(devices.all) { d in DeviceCheckRow(device: d, isOn: binding(for: d)) }
            HStack {
                TextField("Add by address (192.168.0.20)", text: $manualHost)
                    .keyboardType(.numbersAndPunctuation).textInputAutocapitalization(.never).autocorrectionDisabled()
                Button("Add") {
                    let parts = manualHost.split(separator: ":")
                    let id = devices.addManual(host: String(parts[0]), port: parts.count > 1 ? Int(parts[1]) ?? 7878 : 7878)
                    jobs.options.deviceIds.insert(id)
                    manualHost = ""
                }.disabled(manualHost.trimmingCharacters(in: .whitespaces).isEmpty)
            }
        } header: {
            HStack { Text("Train on"); Spacer(); if devices.isBrowsing { ProgressView().controlSize(.mini) } }
        } footer: {
            Text("Tick several devices to split a large scene into blocks trained in parallel (from 400 photos; smaller scenes train whole on the fastest device). Macs appear when SplatWork has Compute › “Act as a worker” on; PCs run SplatWork Node.")
        }

        Section {
            Button {
                jobs.start(item, devices: devices)
            } label: {
                Label("Make splat", systemImage: "sparkles").frame(maxWidth: .infinity).font(.headline)
            }
            .disabled(busyElsewhere || item.frameCount == 0)
        } footer: {
            if busyElsewhere { Text("Another capture is being made into a splat. Wait for it to finish or stop it first.") }
            else if trainsOnPhone { Text("Keep the app open while it trains on this iPhone (plug it in): iOS stops GPU work when the app leaves the screen.") }
        }
    }

    private var trainsOnPhone: Bool {
        let ids = jobs.options.deviceIds
        return ids.isEmpty || ids.contains(SplatDeviceInfo.localId) || !devices.remote.contains { ids.contains($0.id) && $0.isOnline }
    }

    private func binding(for d: SplatDeviceInfo) -> Binding<Bool> {
        Binding(get: { jobs.options.deviceIds.contains(d.id) },
                set: { on in
                    if on { jobs.options.deviceIds.insert(d.id) } else { jobs.options.deviceIds.remove(d.id) }
                    if jobs.options.deviceIds.isEmpty { jobs.options.deviceIds = [SplatDeviceInfo.localId] }
                })
    }

    // MARK: running / finished

    @ViewBuilder private func running(_ p: SplatMakeProgress) -> some View {
        Section {
            VStack(alignment: .leading, spacing: 8) {
                Text(p.stage).font(.headline)
                ProgressView(value: max(0, min(1, p.fraction)))
                HStack {
                    Text(p.totalSteps > 0 ? "Step \(p.step.formatted()) / \(p.totalSteps.formatted())" : p.message).lineLimit(2)
                    Spacer()
                    if let eta = p.eta, eta > 0 { Text("\(ComputeView.duration(eta)) left") }
                }
                .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                if p.totalSteps > 0 {
                    HStack(spacing: 16) {
                        if p.splats > 0 { Text("\(p.splats.formatted()) splats") }
                        if let q = p.psnr, q > 0 { Text(String(format: "PSNR %.2f dB", q)) }
                    }.font(.caption.monospacedDigit())
                }
                ForEach(p.lines, id: \.self) { Text($0).font(.caption2.monospaced()).foregroundStyle(.secondary) }
            }
            .padding(.vertical, 4)
        } footer: {
            Text("You can hide this screen and keep recording other things only if the training runs on a Mac or PC; on this iPhone keep the app open.")
        }
        Section { Button("Stop", role: .destructive) { confirmStop = true } }
    }

    @ViewBuilder private func finished(_ out: SplatOutput, splats: Int, seconds: Double) -> some View {
        Section {
            Button { viewing = out } label: { Label("View splat", systemImage: "cube.transparent").font(.headline) }
            ShareLink(item: out.plyURL) { Label("Share .ply", systemImage: "square.and.arrow.up") }
        } header: {
            Text("\(splats.formatted()) splats in \(ComputeView.duration(seconds))")
        } footer: {
            Text("Saved in Files › On My iPhone › SplatWork Capture › Splats.")
        }
        if !jobs.notes.isEmpty {
            Section("What happened") { ForEach(jobs.notes, id: \.self) { Text($0).font(.caption) } }
        }
        Section { Button("Make again with other settings") { jobs.reset() } }
    }
}

struct DeviceCheckRow: View {
    let device: SplatDeviceInfo
    @Binding var isOn: Bool

    var body: some View {
        Button { isOn.toggle() } label: {
            HStack(spacing: 12) {
                Image(systemName: isOn ? "checkmark.circle.fill" : "circle")
                    .foregroundStyle(isOn ? Color.accentColor : .secondary).font(.title3)
                Image(systemName: icon).frame(width: 24)
                VStack(alignment: .leading, spacing: 2) {
                    Text(device.name).foregroundStyle(.primary)
                    Text(device.detail).font(.caption).foregroundStyle(.secondary).lineLimit(2)
                }
                Spacer()
                if !device.isOnline { Text("offline").font(.caption).foregroundStyle(.orange) }
            }
        }
        .buttonStyle(.plain)
        .contentShape(Rectangle())
        .disabled(!device.isOnline || !device.canTrain)
        .opacity(device.isOnline && device.canTrain ? 1 : 0.5)
    }

    private var icon: String {
        switch device.kind {
        case .thisPhone, .iphone: return "iphone"
        case .mac: return "laptopcomputer"
        case .pc: return "pc"
        }
    }
}

/// Full-screen splat viewer.
struct SplatViewerScreen: View {
    let output: SplatOutput
    let title: String
    @Environment(\.dismiss) private var dismiss
    @State private var status = "Loading…"

    var body: some View {
        ZStack(alignment: .top) {
            SplatViewer(plyURL: output.plyURL, datasetURL: output.datasetURL) { r in
                switch r {
                case .success(let n): status = "\(n.formatted()) splats · drag to orbit, two fingers to pan, pinch to zoom, double-tap to reset"
                case .failure(let e): status = "Couldn't load: \(e.localizedDescription)"
                }
            }
            .ignoresSafeArea()
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(title).font(.headline)
                    Text(status).font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                ShareLink(item: output.plyURL) { Image(systemName: "square.and.arrow.up") }.padding(.trailing, 8)
                Button { dismiss() } label: { Image(systemName: "xmark.circle.fill").font(.title2) }
            }
            .padding(12)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
            .padding(.horizontal, 12)
        }
        .background(Color.black)
    }
}
