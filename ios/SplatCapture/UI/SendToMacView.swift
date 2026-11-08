import SwiftUI

struct SendToMacView: View {
    let item: CaptureItem
    @Environment(AppSettings.self) private var settings
    @Environment(\.dismiss) private var dismiss
    @State private var browser = MacBrowser()
    @State private var uploader = CaptureUploader()
    @State private var selected: DiscoveredMac?
    @State private var pairingCode = ""
    @State private var showManual = false
    @State private var manualHost = ""
    @State private var manualPort = ""

    var body: some View {
        NavigationStack {
            List {
                Section {
                    HStack(spacing: 12) {
                        ThumbnailView(url: item.thumbnailURL)
                            .frame(width: 44, height: 58)
                            .clipShape(RoundedRectangle(cornerRadius: 6))
                        VStack(alignment: .leading) {
                            Text(item.name).font(.headline)
                            Text("\(item.frameCount) frames · \(item.bytes.formattedBytes)")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }

                if uploader.phase == .idle || isFailed {
                    macsSection
                    pairingSection
                }
                statusSection
            }
            .navigationTitle("Send to Mac")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button(uploader.phase.isBusy ? "Cancel" : "Close") {
                        if uploader.phase.isBusy { uploader.cancel() } else { dismiss() }
                    }
                }
            }
            .safeAreaInset(edge: .bottom) { sendButton }
            .onAppear { browser.start() }
            .onDisappear {
                browser.stop()
                uploader.cancel()
            }
            .onChange(of: browser.macs) { _, macs in
                if selected == nil, macs.count == 1 { select(macs[0]) }
            }
            .interactiveDismissDisabled(uploader.phase.isBusy)
            .alert("Enter Mac address", isPresented: $showManual) {
                TextField("IP address, e.g. 192.168.1.20", text: $manualHost)
                    .keyboardType(.numbersAndPunctuation)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                TextField("Port", text: $manualPort).keyboardType(.numberPad)
                Button("Cancel", role: .cancel) {}
                Button("Use") {
                    let host = manualHost.trimmingCharacters(in: .whitespaces)
                    if !host.isEmpty, let port = Int(manualPort) {
                        select(DiscoveredMac(id: "manual-\(host):\(port)", name: host,
                                             target: .manual(host: host, port: port), version: nil))
                    }
                }
            } message: {
                Text("Use this if your Mac doesn't appear automatically. The address and port are shown in SplatWork › Receive from iPhone.")
            }
        }
    }

    private var isFailed: Bool {
        if case .failed = uploader.phase { return true }
        return false
    }

    private var macsSection: some View {
        Section {
            ForEach(browser.macs) { mac in
                Button { select(mac) } label: {
                    HStack {
                        Image(systemName: "desktopcomputer").foregroundStyle(.tint)
                        Text(mac.name).foregroundStyle(.primary)
                        Spacer()
                        if selected?.id == mac.id { Image(systemName: "checkmark").foregroundStyle(.tint) }
                    }
                }
            }
            if let s = selected, case .manual = s.target {
                HStack {
                    Image(systemName: "network").foregroundStyle(.tint)
                    Text(s.name)
                    Spacer()
                    Image(systemName: "checkmark").foregroundStyle(.tint)
                }
            }
            if browser.macs.isEmpty {
                HStack(spacing: 10) {
                    ProgressView()
                    Text("Looking for SplatWork…").foregroundStyle(.secondary)
                }
            }
            Button("Enter address manually…") { showManual = true }
        } header: {
            Text("Macs on this network")
        } footer: {
            switch browser.state {
            case .waiting(let m), .failed(let m):
                Text(m).foregroundStyle(.orange)
            default:
                Text("Open SplatWork on your Mac and turn on “Receive from iPhone”. Both devices must be on the same Wi-Fi network.")
            }
        }
    }

    private var pairingSection: some View {
        Section {
            TextField("6-digit code (if required)", text: $pairingCode)
                .keyboardType(.numberPad)
                .textContentType(.oneTimeCode)
                .font(.title3.monospacedDigit())
                .onChange(of: pairingCode) { _, v in
                    let digits = String(v.filter(\.isNumber).prefix(6))
                    if digits != v { pairingCode = digits }
                }
        } header: {
            Text("Pairing code")
        } footer: {
            Text("Only needed if “Require pairing code” is on in SplatWork. The code is shown in its Receive from iPhone panel and is remembered for this Mac.")
        }
    }

    @ViewBuilder
    private var statusSection: some View {
        switch uploader.phase {
        case .idle:
            EmptyView()
        case .connecting:
            Section { HStack { ProgressView(); Text("Connecting to \(selected?.name ?? "Mac")…") } }
        case .zipping(let p):
            Section {
                ProgressView(value: p) { Text("Packaging capture…") } currentValueLabel: { Text(p, format: .percent.precision(.fractionLength(0))) }
            }
        case .uploading(let p):
            Section {
                ProgressView(value: p) {
                    Text("Sending to \(uploader.macName ?? "Mac")…")
                } currentValueLabel: {
                    let sent = Int64(Double(uploader.bytesTotal) * p)
                    Text("\(sent.formattedBytes) of \(uploader.bytesTotal.formattedBytes)\(uploader.attempt > 1 ? " · attempt \(uploader.attempt)" : "")")
                }
            } footer: {
                Text("Keep SplatWork Capture open until the transfer finishes.")
            }
        case .done(let savedAs):
            Section {
                Label("Sent! Saved on \(uploader.macName ?? "the Mac") as \(savedAs).", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green)
            }
        case .failed(let message, let needsCode):
            Section {
                Label(message, systemImage: needsCode ? "lock.fill" : "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
            }
        }
    }

    @ViewBuilder
    private var sendButton: some View {
        let busy = uploader.phase.isBusy
        let done: Bool = { if case .done = uploader.phase { return true }; return false }()
        Button {
            if done { dismiss(); return }
            guard let mac = selected else { return }
            if !pairingCode.isEmpty { settings.pairingCodes[mac.name] = pairingCode }
            uploader.send(captureURL: item.url, captureName: item.name, to: mac,
                          pairingCode: pairingCode.isEmpty ? nil : pairingCode)
        } label: {
            Text(done ? "Done" : isFailed ? "Retry" : "Send")
                .font(.headline)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 6)
        }
        .buttonStyle(.borderedProminent)
        .disabled(busy || (selected == nil && !done))
        .padding(.horizontal, 16)
        .padding(.bottom, 8)
        .background(.bar)
    }

    private func select(_ mac: DiscoveredMac) {
        selected = mac
        if let code = settings.pairingCodes[mac.name], pairingCode.isEmpty { pairingCode = code }
        if isFailed { uploader.reset() }
    }
}
