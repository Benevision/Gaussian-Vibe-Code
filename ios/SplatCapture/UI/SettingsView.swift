import SwiftUI

struct SettingsView: View {
    @Environment(AppSettings.self) private var appSettings
    @Environment(\.dismiss) private var dismiss
    @State private var confirmReset = false

    var body: some View {
        @Bindable var app = appSettings
        let s = $app.capture
        let hasLiDAR = DeviceInfo.hasLiDAR

        NavigationStack {
            Form {
                Section {
                    Text("The defaults work well for most scenes — you can just press record. Each option below explains what it does, its default, and the trade-off.")
                        .font(.footnote).foregroundStyle(.secondary)
                }

                // MARK: Keyframes
                Section("Keyframes") {
                    Explained("Automatic saves a frame whenever you've moved or turned far enough since the last one — this gives even coverage no matter how fast you walk. Timed saves at a fixed interval. Default: Automatic.") {
                        Picker("Mode", selection: s.mode) {
                            ForEach(KeyframeMode.allCases) { Text($0.label).tag($0) }
                        }
                    }
                    if app.capture.mode == .motion {
                        Explained("A new keyframe is taken after the camera moves this far. Smaller = more overlap and more frames (better quality, bigger captures, longer training). Default: 7 cm. Use 3–5 cm for small objects, 10–20 cm for rooms.") {
                            SliderRow(title: "Min. movement", value: floatBinding(s.minTranslation, scale: 100),
                                      range: 1...30, step: 1, unit: "cm")
                        }
                        Explained("…or after it turns this much, even without moving (catches panning). Default: 6°. Lower values add frames while you look around.") {
                            SliderRow(title: "Min. rotation", value: floatBinding(s.minRotationDegrees),
                                      range: 2...20, step: 1, unit: "°")
                        }
                    } else {
                        Explained("Seconds between keyframes in timed mode. Default: 0.5 s. Walk slowly and steadily — standing still produces near-duplicate frames.") {
                            SliderRow(title: "Interval", value: s.timedInterval, range: 0.2...3, step: 0.1, unit: "s", decimals: 1)
                        }
                    }
                    Explained("Upper limit on keyframes per second, whatever the mode. Stops fast motion from flooding storage with near-duplicates and keeps the writer from falling behind. Default: 3/s.") {
                        SliderRow(title: "Max. rate", value: s.maxKeyframeRate, range: 0.5...10, step: 0.5, unit: "/s", decimals: 1)
                    }
                    Explained("Recording stops automatically at this many keyframes, to protect storage. 200–600 is typical for a room or object. Default: 1500.") {
                        Stepper(value: s.maxFrames, in: 100...10000, step: 100) {
                            LabeledContent("Frame limit", value: "\(app.capture.maxFrames)")
                        }
                    }
                }

                // MARK: Quality gates
                Section("Quality filters") {
                    Explained("Only save frames while ARKit tracking is ‘normal’. Frames captured with limited tracking often have wrong poses that ruin a splat. Default: On.") {
                        Toggle("Require good tracking", isOn: s.requireNormalTracking)
                    }
                    Explained("Skip frames that are likely blurred (the next sharp frame is taken instead). Blur is estimated from rotation speed × exposure time and from image sharpness compared with recent frames. Default: On.") {
                        Toggle("Reject blurry frames", isOn: s.rejectBlurry)
                    }
                    if app.capture.rejectBlurry {
                        Explained("Maximum predicted motion blur, in pixels. Lower = stricter (sharper images, but more frames skipped in dim light). Default: 2.5 px.") {
                            SliderRow(title: "Max. motion blur", value: floatBinding(s.maxMotionBlurPixels),
                                      range: 0.5...8, step: 0.5, unit: "px", decimals: 1)
                        }
                        Explained("A frame is rejected if its detail is below this fraction of recent sharp frames. Higher = stricter. Default: 55%.") {
                            SliderRow(title: "Min. relative sharpness", value: floatBinding(s.minRelativeSharpness, scale: 100),
                                      range: 20...90, step: 5, unit: "%")
                        }
                    }
                }

                // MARK: Camera
                Section("Camera & images") {
                    Explained("Resolution and frame rate of the ARKit camera stream. Higher resolution gives more detail but larger files and fewer fps; 4K formats are only on recent iPhones. Default: the first (ARKit-recommended) format, usually 1920×1440.") {
                        Picker("Video format", selection: s.videoFormatID) {
                            Text("Default").tag(String?.none)
                            ForEach(app.videoFormats) { f in
                                Text(f.label).tag(String?.some(f.id))
                            }
                        }
                        .disabled(app.capture.highResolutionStills)
                    }
                    Explained("Takes a full-resolution photo for every keyframe instead of using the video frame (e.g. 12 MP vs 2.7 MP). Much sharper, but slower (≈1–2 keyframes/s), much larger captures, and depth may be sampled a few ms apart from the photo. Overrides the video format. Default: Off.") {
                        Toggle("High-resolution stills", isOn: s.highResolutionStills)
                    }
                    Explained("JPEG is universally compatible. HEIC is ~40% smaller at the same quality; Splat Studio reads both. Default: JPEG.") {
                        Picker("Image format", selection: s.imageFormat) {
                            ForEach(ImageFileFormat.allCases) { Text($0.label).tag($0) }
                        }
                    }
                    Explained("Compression quality. 0.9 is visually lossless for training; 1.0 roughly doubles size for little gain; below 0.8 compression artefacts can appear in the splat. Default: 0.90.") {
                        SliderRow(title: "Quality", value: s.imageQuality, range: 0.5...1, step: 0.05, unit: "", decimals: 2)
                    }
                    Explained("Freeze exposure and white balance when recording starts, so colours stay consistent between frames (splats hate flickering brightness). Turn off if you walk between very bright and dark areas. Default: On.") {
                        Toggle("Lock exposure & white balance", isOn: s.lockExposureWhileRecording)
                    }
                    Explained("Lets the camera refocus continuously. Off fixes focus (and thus intrinsics) — only useful when the subject distance stays constant. Default: On.") {
                        Toggle("Autofocus", isOn: s.autoFocus)
                    }
                }

                // MARK: Depth
                Section {
                    Explained("Save the LiDAR depth map (256×192, float16) and use it for the point cloud. Splat Studio can use depth to supervise training and to initialise geometry. Default: On.") {
                        Toggle("Record LiDAR depth", isOn: s.recordDepth)
                    }
                    .disabled(!hasLiDAR)
                    Explained("Use ARKit's temporally smoothed depth instead of per-frame depth. Less noise and fewer holes, but edges can lag slightly behind fast motion. Default: Off.") {
                        Toggle("Smoothed depth", isOn: s.smoothedDepth)
                    }
                    .disabled(!hasLiDAR || !app.capture.recordDepth)
                    Explained("Also save the per-pixel confidence map (0 = low, 1 = medium, 2 = high) next to each depth map so the Mac can ignore unreliable depth. Tiny files. Default: On.") {
                        Toggle("Save confidence", isOn: s.saveConfidence)
                    }
                    .disabled(!hasLiDAR || !app.capture.recordDepth)
                } header: {
                    Text("LiDAR depth")
                } footer: {
                    if !hasLiDAR {
                        Text("This iPhone has no LiDAR scanner. Captures contain posed images only (hasLiDAR = false); Splat Studio will rely on images alone.")
                    }
                }

                // MARK: Point cloud
                Section("Point cloud") {
                    Explained("Fuse LiDAR depth into a coloured point cloud (points.ply) while recording. It's used to start training from real geometry, which converges faster and gives cleaner splats. Default: On.") {
                        Toggle("Build point cloud", isOn: s.buildPointCloud)
                    }
                    .disabled(!hasLiDAR || !app.capture.recordDepth)
                    if app.capture.buildPointCloud && hasLiDAR {
                        Explained("Points are merged into cubes of this size. Smaller = denser cloud and more memory; larger = sparser but lighter. Default: 1.5 cm.") {
                            SliderRow(title: "Voxel size", value: floatBinding(s.voxelSizeCm), range: 0.5...5, step: 0.5, unit: "cm", decimals: 1)
                        }
                        Explained("Ignore depth pixels below this LiDAR confidence. Medium removes most flying pixels at edges; High is cleanest but sparser. Default: Medium.") {
                            Picker("Min. confidence", selection: s.minConfidence) {
                                ForEach(ConfidenceLevel.allCases) { Text($0.label).tag($0) }
                            }
                        }
                        Explained("Ignore depth further than this. LiDAR becomes noisy beyond ~4 m. Default: 4 m.") {
                            SliderRow(title: "Max. depth", value: floatBinding(s.maxDepth), range: 1...5, step: 0.5, unit: "m", decimals: 1)
                        }
                    }
                }

                // MARK: Display
                Section("Display") {
                    Explained("Draw the captured point cloud over the camera feed so you can see what's covered. Gaps = areas to film more. Default: On.") {
                        Toggle("Coverage overlay", isOn: s.showOverlay)
                    }
                    if app.capture.showOverlay {
                        Explained("True colour shows points with their real colours; Highlight draws them in a bright tint that stands out against the camera image. Default: True colour.") {
                            Picker("Overlay style", selection: s.overlayStyle) {
                                ForEach(OverlayStyle.allCases) { Text($0.label).tag($0) }
                            }
                        }
                        Explained("Size of the overlay dots on screen. Default: 3.") {
                            SliderRow(title: "Dot size", value: floatBinding(s.overlayPointSize), range: 1...8, step: 1, unit: "")
                        }
                    }
                    Explained("Top-down map of your path while recording, with a ring that fills in green as you cover each direction around the subject. Default: On.") {
                        Toggle("Mini-map & orbit hint", isOn: s.showMiniMap)
                    }
                    Explained("A light tap each time a keyframe is saved — lets you keep your eyes on the subject. Default: On.") {
                        Toggle("Haptic tick per keyframe", isOn: s.haptics)
                    }
                }

                Section {
                    Button("Reset all settings to defaults", role: .destructive) { confirmReset = true }
                } footer: {
                    Text("Splat Capture \(DeviceInfo.appVersion) · \(DeviceInfo.modelName)")
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
            }
            .confirmationDialog("Reset all settings?", isPresented: $confirmReset, titleVisibility: .visible) {
                Button("Reset", role: .destructive) { appSettings.resetToDefaults() }
            }
        }
    }

    private func floatBinding(_ b: Binding<Float>, scale: Double = 1) -> Binding<Double> {
        Binding(get: { Double(b.wrappedValue) * scale },
                set: { b.wrappedValue = Float($0 / scale) })
    }
}

/// A control followed by an always-visible explanation.
struct Explained<Content: View>: View {
    let text: String
    let content: Content

    init(_ text: String, @ViewBuilder content: () -> Content) {
        self.text = text
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            content
            Text(text)
                .font(.footnote)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.vertical, 2)
    }
}

struct SliderRow: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let step: Double
    let unit: String
    var decimals: Int = 0

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            LabeledContent(title) {
                Text("\(value, specifier: "%.\(decimals)f")\(unit.isEmpty ? "" : " \(unit)")")
                    .monospacedDigit()
            }
            Slider(value: $value, in: range, step: step)
                .accessibilityLabel(title)
        }
    }
}
