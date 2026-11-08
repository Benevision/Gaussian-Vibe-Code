import SwiftUI

/// Easy mode shows presets and the essential controls; Advanced reveals every parameter.
enum ExperienceMode: String, CaseIterable, Identifiable {
    case easy, advanced
    var id: String { rawValue }
    var title: String { self == .easy ? "Easy" : "Advanced" }
}

private struct ExperienceModeKey: EnvironmentKey { static let defaultValue: ExperienceMode = .easy }
extension EnvironmentValues {
    var experienceMode: ExperienceMode {
        get { self[ExperienceModeKey.self] }
        set { self[ExperienceModeKey.self] = newValue }
    }
}

/// ⓘ button that opens the full documentation for a setting.
struct HelpButton: View {
    let id: String
    @State private var shown = false
    var body: some View {
        Button { shown.toggle() } label: {
            Image(systemName: "info.circle").foregroundStyle(.secondary)
        }
        .buttonStyle(.plain)
        .help(SettingsCatalog.doc(id)?.summary ?? "")
        .accessibilityLabel("Help for \(SettingsCatalog.doc(id)?.title ?? id)")
        .popover(isPresented: $shown, arrowEdge: .trailing) { SettingDocView(id: id).frame(width: 380).padding() }
    }
}

struct SettingDocView: View {
    let id: String
    var body: some View {
        if let d = SettingsCatalog.doc(id) {
            let parts = d.structuredDetail
            VStack(alignment: .leading, spacing: 14) {
                HStack(alignment: .firstTextBaseline) {
                    Text(d.title).font(.title3.bold())
                    Spacer()
                    Text(d.advanced ? "Advanced" : "Easy").font(.caption2.weight(.semibold))
                        .padding(.horizontal, 7).padding(.vertical, 2)
                        .background(d.advanced ? AnyShapeStyle(.quaternary) : AnyShapeStyle(Color.accentColor.opacity(0.18)), in: Capsule())
                }
                Text(d.summary).font(.callout)
                    .padding(10).frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.accentColor.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
                    .overlay(alignment: .leading) { RoundedRectangle(cornerRadius: 1.5).fill(Color.accentColor).frame(width: 3).padding(.vertical, 4) }
                // Quick facts
                Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 0) {
                    factRow("Default", d.defaultValue)
                    if !d.range.isEmpty { Divider(); factRow("Range", d.range) }
                    if !d.effects.isEmpty { Divider(); factRow("Effect", d.effects) }
                    Divider()
                    GridRow {
                        Text("ID").foregroundStyle(.secondary)
                        Text(d.id).font(.callout.monospaced())
                    }.padding(.vertical, 6)
                }
                .font(.callout)
                .padding(.horizontal, 10)
                .background(.quinary, in: RoundedRectangle(cornerRadius: 8))
                if !parts.overview.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("How it works").font(.headline)
                        Text(parts.overview).fixedSize(horizontal: false, vertical: true).lineSpacing(2)
                        bullets(parts.details).padding(.top, 2)
                    }
                }
                if !parts.tips.isEmpty {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("When to change it").font(.headline)
                        bullets(parts.tips)
                    }
                }
            }
            .textSelection(.enabled)
        } else {
            Text("No documentation for \(id)")
        }
    }

    private func bullets(_ items: [String]) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            ForEach(Array(items.enumerated()), id: \.offset) { _, t in
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    Text("•").foregroundStyle(.secondary)
                    Text(t).fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    @ViewBuilder private func factRow(_ label: String, _ value: String) -> some View {
        GridRow {
            Text(label).foregroundStyle(.secondary)
            Text(value).fixedSize(horizontal: false, vertical: true)
        }.padding(.vertical, 6)
    }
}

/// A labelled setting row: title, ⓘ, control, and a one-line summary underneath.
/// Hidden automatically in Easy mode when the catalog marks the setting as advanced.
struct SettingRow<Control: View>: View {
    let id: String
    var showSummary = true
    /// Always put the control under the title (lists whose width changes, e.g. the device checkboxes).
    var stacked = false
    @ViewBuilder var control: () -> Control
    @Environment(\.experienceMode) private var mode

    var body: some View {
        let d = SettingsCatalog.doc(id)
        if mode == .advanced || !(d?.advanced ?? false) {
            VStack(alignment: .leading, spacing: 3) {
                // Side by side when there is room; otherwise the control moves under the title
                // so neither the title nor the control is ever truncated.
                if stacked {
                    VStack(alignment: .leading, spacing: 5) {
                        titleView(d).fixedSize(horizontal: false, vertical: true)
                        HStack { control(); Spacer(minLength: 0) }
                    }
                } else {
                    ViewThatFits(in: .horizontal) {
                        HStack(alignment: .center, spacing: 8) {
                            titleView(d).fixedSize()
                            Spacer(minLength: 8)
                            control().fixedSize()
                        }
                        VStack(alignment: .leading, spacing: 5) {
                            titleView(d).fixedSize(horizontal: false, vertical: true)
                            HStack { control(); Spacer(minLength: 0) }
                        }
                    }
                }
                if showSummary, let s = d?.summary {
                    Text(s).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
            }
            .padding(.vertical, 2)
        }
    }

    private func titleView(_ d: SettingDoc?) -> some View {
        HStack(spacing: 4) {
            Text(d?.title ?? id)
            HelpButton(id: id)
        }
    }
}

/// Slider + numeric field, with optional log scale for learning rates.
struct NumberSlider: View {
    @Binding var value: Double
    var range: ClosedRange<Double>
    var log: Bool = false
    var format: String = "%.3g"
    var width: CGFloat = 130
    /// Round to this many decimal places (slider and text field); nil keeps 4 significant digits.
    var decimals: Int? = nil

    private func rounded(_ v: Double) -> Double {
        guard let d = decimals else { return v }
        let m = pow(10, Double(d))
        return min(range.upperBound, max(range.lowerBound, (v * m).rounded() / m))
    }

    var body: some View {
        HStack(spacing: 6) {
            if log {
                Slider(value: Binding(get: { Foundation.log10(max(value, range.lowerBound)) },
                                      set: { value = rounded(pow(10, $0)) }),
                       in: Foundation.log10(range.lowerBound)...Foundation.log10(range.upperBound))
                    .frame(width: width)
            } else {
                Slider(value: Binding(get: { value }, set: { value = rounded($0) }), in: range).frame(width: width)
            }
            if let d = decimals {
                TextField("", value: Binding(get: { value }, set: { value = rounded($0) }), format: .number.precision(.fractionLength(0...d)))
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 72)
                    .multilineTextAlignment(.trailing)
            } else {
                TextField("", value: $value, format: .number.precision(.significantDigits(4)))
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 72)
                    .multilineTextAlignment(.trailing)
            }
        }
    }
}

extension Binding where Value == Float {
    var double: Binding<Double> { Binding<Double>(get: { Double(wrappedValue) }, set: { wrappedValue = Float($0) }) }
}
extension Binding where Value == Int {
    var double: Binding<Double> { Binding<Double>(get: { Double(wrappedValue) }, set: { wrappedValue = Int($0.rounded()) }) }
}

/// Integer stepper with text field.
struct IntField: View {
    @Binding var value: Int
    var range: ClosedRange<Int>
    var step: Int = 1
    var body: some View {
        HStack(spacing: 4) {
            TextField("", value: $value, format: .number)
                .textFieldStyle(.roundedBorder).frame(width: 90).multilineTextAlignment(.trailing)
            Stepper("", value: $value, in: range, step: step).labelsHidden()
        }
    }
}

/// Section header with an optional subtitle; used across inspector panels.
struct PanelSection<Content: View>: View {
    let title: String
    var systemImage: String? = nil
    var subtitle: String? = nil
    @ViewBuilder var content: () -> Content
    @State private var expanded = true
    var body: some View {
        DisclosureGroup(isExpanded: $expanded) {
            VStack(alignment: .leading, spacing: 8) { content() }.padding(.top, 6)
        } label: {
            VStack(alignment: .leading, spacing: 1) {
                Label(title, systemImage: systemImage ?? "circle").labelStyle(TitleAndIconOrTitle(hasIcon: systemImage != nil)).font(.headline)
                if let subtitle { Text(subtitle).font(.caption).foregroundStyle(.secondary) }
            }
        }
        .padding(.vertical, 4)
    }
}

private struct TitleAndIconOrTitle: LabelStyle {
    var hasIcon: Bool
    func makeBody(configuration: Configuration) -> some View {
        if hasIcon { HStack(spacing: 6) { configuration.icon.foregroundStyle(.tint); configuration.title } } else { configuration.title }
    }
}

/// Big, friendly card used on the welcome screen.
struct ActionCard: View {
    let title: String
    let subtitle: String
    let systemImage: String
    var tint: Color = .accentColor
    let action: () -> Void
    @State private var hover = false
    var body: some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: 8) {
                Image(systemName: systemImage).font(.system(size: 26, weight: .medium)).foregroundStyle(tint)
                Text(title).font(.headline).foregroundStyle(.primary)
                Text(subtitle).font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.leading).fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 0)
            }
            .padding(14)
            .frame(width: 200, height: 132, alignment: .topLeading)
            .background(RoundedRectangle(cornerRadius: 12).fill(hover ? AnyShapeStyle(.quaternary) : AnyShapeStyle(.background.secondary)))
            .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(hover ? tint.opacity(0.6) : .secondary.opacity(0.2)))
        }
        .buttonStyle(.plain)
        .onHover { hover = $0 }
    }
}

func formatCount(_ n: Int) -> String {
    if n >= 1_000_000 { return String(format: "%.2f M", Double(n) / 1e6) }
    if n >= 10_000 { return String(format: "%.0f k", Double(n) / 1e3) }
    return "\(n)"
}

func formatDuration(_ s: Double) -> String {
    guard s.isFinite, s >= 0 else { return "–" }
    let t = Int(s)
    if t >= 3600 { return String(format: "%d h %02d min", t / 3600, (t % 3600) / 60) }
    if t >= 60 { return String(format: "%d min %02d s", t / 60, t % 60) }
    return "\(t) s"
}

func formatBytes(_ b: Int) -> String { ByteCountFormatter.string(fromByteCount: Int64(b), countStyle: .file) }
