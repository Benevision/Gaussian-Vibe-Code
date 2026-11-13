import SwiftUI

struct CaptureView: View {
    @Environment(AppSettings.self) private var settings
    @Environment(CaptureLibrary.self) private var library
    @Environment(\.scenePhase) private var scenePhase
    @State private var model = CaptureViewModel()
    @State private var showLibrary = false
    @State private var showSettings = false
    @State private var flash = false
    @State private var sendTarget: CaptureItem?
    @State private var makeTarget: CaptureItem?

    var body: some View {
        ZStack {
            ARCaptureView(model: model, settings: settings.capture)
                .ignoresSafeArea()

            // Keyframe flash
            RoundedRectangle(cornerRadius: 0)
                .strokeBorder(.white.opacity(flash ? 0.9 : 0), lineWidth: 6)
                .ignoresSafeArea()
                .allowsHitTesting(false)

            VStack(spacing: 10) {
                topBar
                if model.isRecording && settings.capture.showMiniMap {
                    HStack {
                        MiniMapView(keyframes: model.keyframePositions,
                                    camera: model.stats.cameraPosition,
                                    forward: model.stats.cameraForward,
                                    center: model.lookAtCentroid,
                                    sectors: model.orbitSectors)
                            .frame(width: 130, height: 130)
                            .transition(.scale(scale: 0.8, anchor: .topLeading).combined(with: .opacity))
                        Spacer()
                    }
                }
                Spacer()
                guidanceBanner
                bottomBar
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 8)
        }
        .preferredColorScheme(.dark)
        .statusBarHidden(model.isRecording)
        .animation(.spring(duration: 0.3), value: model.isRecording)
        .animation(.easeInOut(duration: 0.25), value: model.stats.guidance)
        .sensoryFeedback(trigger: model.keyframeCount) { _, new in
            (settings.capture.haptics && new > 0) ? .impact(weight: .light, intensity: 0.6) : nil
        }
        .onChange(of: model.keyframeCount) { _, new in
            guard new > 0 else { return }
            flash = true
            withAnimation(.easeOut(duration: 0.25)) { flash = false }
        }
        .onAppear { model.startSession(settings: settings.capture) }
        .onChange(of: settings.capture) { _, new in
            if !model.isRecording { model.startSession(settings: new) }
        }
        .onChange(of: scenePhase) { _, phase in
            switch phase {
            case .background:
                if model.isRecording {
                    model.autoStopMessage = "Recording stopped because the app went to the background."
                    model.stopRecording()
                }
            case .active:
                if !showLibrary && !showSettings && makeTarget == nil { model.startSession(settings: settings.capture) }
            default: break
            }
        }
        .fullScreenCover(isPresented: $showLibrary, onDismiss: { model.startSession(settings: settings.capture) }) {
            LibraryView()
        }
        .sheet(isPresented: $showSettings, onDismiss: { model.startSession(settings: settings.capture) }) {
            SettingsView()
        }
        .sheet(isPresented: $model.showFinished) {
            if let summary = model.lastSummary {
                CaptureFinishedSheet(summary: summary,
                                     autoStopMessage: model.autoStopMessage,
                                     onSend: {
                                         model.showFinished = false
                                         Task {
                                             await library.reload()
                                             // Let the summary sheet finish dismissing first.
                                             try? await Task.sleep(for: .milliseconds(500))
                                             sendTarget = library.item(at: summary.url)
                                         }
                                     },
                                     onMakeSplat: {
                                         model.showFinished = false
                                         Task {
                                             await library.reload()
                                             try? await Task.sleep(for: .milliseconds(500))
                                             if let item = library.item(at: summary.url) {
                                                 // training wants the whole GPU: no AR session underneath
                                                 model.pauseSession()
                                                 makeTarget = item
                                             }
                                         }
                                     },
                                     onLibrary: {
                                         model.showFinished = false
                                         openLibrary()
                                     },
                                     onDiscard: { model.discardLastCapture() })
                    .presentationDetents([.medium, .large])
            }
        }
        .sheet(item: $sendTarget) { item in
            SendToMacView(item: item)
        }
        .fullScreenCover(item: $makeTarget, onDismiss: { model.startSession(settings: settings.capture) }) { item in
            MakeSplatView(item: item)
        }
        .alert("SplatWork Capture", isPresented: Binding(
            get: { model.errorMessage != nil },
            set: { if !$0 { model.errorMessage = nil } })) {
            if model.errorMessage?.contains("Settings") == true {
                Button("Open Settings") {
                    if let url = URL(string: UIApplication.openSettingsURLString) { UIApplication.shared.open(url) }
                }
            }
            Button("OK", role: .cancel) {}
        } message: {
            Text(model.errorMessage ?? "")
        }
    }

    private func openLibrary() {
        model.pauseSession()
        showLibrary = true
    }

    // MARK: - Top

    private var topBar: some View {
        HStack(spacing: 8) {
            trackingPill
            Spacer()
            if model.isRecording, let start = model.recordingStart {
                TimelineView(.periodic(from: start, by: 1)) { ctx in
                    let s = Int(ctx.date.timeIntervalSince(start))
                    Label(String(format: "%d:%02d", s / 60, s % 60), systemImage: "record.circle")
                        .font(.footnote.monospacedDigit().weight(.semibold))
                        .foregroundStyle(.red)
                        .padding(.horizontal, 10).padding(.vertical, 6)
                        .background(.ultraThinMaterial, in: Capsule())
                }
            }
            Text(model.stats.hasDepth ? "LiDAR" : "No depth")
                .font(.caption.weight(.semibold))
                .padding(.horizontal, 10).padding(.vertical, 6)
                .background(.ultraThinMaterial, in: Capsule())
                .foregroundStyle(model.stats.hasDepth ? .green : .secondary)
        }
        .padding(.top, 4)
    }

    private var trackingPill: some View {
        let (text, color): (String, Color) = switch model.stats.tracking {
        case .normal: ("Tracking", .green)
        case .limited: ("Limited tracking", .yellow)
        case .notAvailable: ("No tracking", .red)
        }
        return HStack(spacing: 6) {
            Circle().fill(color).frame(width: 8, height: 8)
            Text(text).font(.caption.weight(.semibold))
        }
        .padding(.horizontal, 10).padding(.vertical, 6)
        .background(.ultraThinMaterial, in: Capsule())
        .accessibilityElement(children: .combine)
    }

    @ViewBuilder
    private var guidanceBanner: some View {
        if let msg = model.stats.guidance.message {
            Label(msg, systemImage: model.stats.guidance.isSevere ? "exclamationmark.triangle.fill" : "info.circle.fill")
                .font(.callout.weight(.semibold))
                .multilineTextAlignment(.center)
                .padding(.horizontal, 14).padding(.vertical, 10)
                .background((model.stats.guidance.isSevere ? Color.orange : Color.black).opacity(0.7),
                            in: RoundedRectangle(cornerRadius: 12))
                .foregroundStyle(.white)
                .transition(.move(edge: .bottom).combined(with: .opacity))
                .id(msg)
        } else if !model.isRecording && model.phase == .idle {
            Text("Press record, then walk slowly around your subject.")
                .font(.footnote)
                .padding(.horizontal, 12).padding(.vertical, 8)
                .background(.black.opacity(0.5), in: Capsule())
                .foregroundStyle(.white.opacity(0.9))
        }
    }

    // MARK: - Bottom

    private var bottomBar: some View {
        VStack(spacing: 12) {
            if model.isRecording || model.phase == .finishing {
                HStack(spacing: 22) {
                    stat("\(model.stats.keyframes)", "frames")
                    stat(model.stats.bytes.formattedBytes, "stored")
                    if model.stats.rejectedBlurry > 0 {
                        stat("\(model.stats.rejectedBlurry)", "skipped (blur)")
                    }
                }
                .padding(.horizontal, 16).padding(.vertical, 8)
                .background(.ultraThinMaterial, in: Capsule())
            }
            HStack {
                HStack {
                    circleButton("square.stack.3d.up", label: "Library") { openLibrary() }
                    Spacer()
                }
                .frame(maxWidth: .infinity)
                .opacity(model.phase == .idle ? 1 : 0)
                .disabled(model.phase != .idle)
                RecordButton(phase: model.phase) {
                    if model.isRecording { model.stopRecording() } else { model.startRecording(settings: settings.capture) }
                }
                HStack(spacing: 10) {
                    Spacer()
                    circleButton("gearshape", label: "Settings") { showSettings = true }
                }
                .frame(maxWidth: .infinity)
                .opacity(model.phase == .idle ? 1 : 0)
                .disabled(model.phase != .idle)
            }
        }
    }

    private func stat(_ value: String, _ label: String) -> some View {
        VStack(spacing: 0) {
            Text(value).font(.headline.monospacedDigit()).contentTransition(.numericText())
            Text(label).font(.caption2).foregroundStyle(.secondary)
        }
    }

    private func circleButton(_ icon: String, label: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: icon)
                .font(.title3.weight(.semibold))
                .frame(width: 54, height: 54)
                .background(.ultraThinMaterial, in: Circle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(label)
    }
}

struct RecordButton: View {
    let phase: CaptureViewModel.Phase
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack {
                Circle().strokeBorder(.white, lineWidth: 5).frame(width: 84, height: 84)
                if phase == .finishing {
                    ProgressView().tint(.white).controlSize(.large)
                } else {
                    RoundedRectangle(cornerRadius: phase == .recording ? 8 : 34)
                        .fill(.red)
                        .frame(width: phase == .recording ? 34 : 68, height: phase == .recording ? 34 : 68)
                }
            }
            .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .disabled(phase == .finishing)
        .animation(.spring(duration: 0.3), value: phase)
        .accessibilityLabel(phase == .recording ? "Stop recording" : "Start recording")
    }
}

struct CaptureFinishedSheet: View {
    let summary: CaptureSummary
    let autoStopMessage: String?
    let onSend: () -> Void
    let onMakeSplat: () -> Void
    let onLibrary: () -> Void
    let onDiscard: () -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var confirmDiscard = false

    var body: some View {
        NavigationStack {
            List {
                if let autoStopMessage {
                    Label(autoStopMessage, systemImage: "info.circle").foregroundStyle(.orange)
                }
                Section {
                    LabeledContent("Name", value: summary.name)
                    LabeledContent("Keyframes", value: "\(summary.frameCount)")
                    LabeledContent("Size", value: summary.bytes.formattedBytes)
                    LabeledContent("Duration", value: Duration.seconds(summary.duration).formatted(.time(pattern: .minuteSecond)))
                    if summary.pointCount > 0 {
                        LabeledContent("Point cloud", value: "\(summary.pointCount.formatted()) points")
                    }
                }
                if summary.frameCount < 30 {
                    Label("That's only a few frames. Most scenes need 100–400 keyframes covering every side.",
                          systemImage: "exclamationmark.triangle")
                        .font(.footnote).foregroundStyle(.secondary)
                }
                Section {
                    Button { onMakeSplat() } label: { Label("Make splat", systemImage: "sparkles") }
                    Button { onSend() } label: { Label("Send to Mac", systemImage: "desktopcomputer.and.arrow.down") }
                    Button { onLibrary() } label: { Label("Open Library", systemImage: "square.stack.3d.up") }
                    Button(role: .destructive) { confirmDiscard = true } label: { Label("Discard Capture", systemImage: "trash") }
                }
            }
            .navigationTitle("Capture saved")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
            }
            .confirmationDialog("Discard this capture? This deletes it from the iPhone.",
                                isPresented: $confirmDiscard, titleVisibility: .visible) {
                Button("Discard", role: .destructive) { onDiscard() }
            }
        }
    }
}
