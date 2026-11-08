import SwiftUI

/// The update prompt (AppUpdater): shown when a verified update is ready, and for "Check for Updates…".
struct UpdateSheet: View {
    @Bindable var updater = AppUpdater.shared
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .top, spacing: 14) {
                Image(nsImage: NSApp.applicationIconImage).resizable().frame(width: 56, height: 56)
                VStack(alignment: .leading, spacing: 4) {
                    Text(title).font(.title3.bold())
                    Text(subtitle).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
            }
            if case .downloading(let f) = updater.phase { ProgressView(value: f) }
            if updater.phase == .checking { ProgressView().controlSize(.small) }
            if let notes = updater.release?.notes, !notes.isEmpty, showsRelease {
                ScrollView {
                    Text(Self.markdown(notes)).frame(maxWidth: .infinity, alignment: .leading).textSelection(.enabled)
                        .padding(10)
                }
                .frame(height: 220)
                .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
            }
            if updater.phase == .ready, updater.isBusy() {
                Label("Training or preparation is running here. Relaunching stops it — or choose Later and the update installs when you quit.",
                      systemImage: "exclamationmark.triangle").font(.callout).foregroundStyle(.orange)
            }
            Toggle("Check for updates automatically", isOn: $updater.autoCheck)
            Toggle("Download updates in the background", isOn: $updater.autoDownload).disabled(!updater.autoCheck)
            HStack {
                if showsRelease, let r = updater.release {
                    Button("Skip This Version") { updater.skipThisVersion(); dismiss() }
                    Link("Release Page", destination: r.page)
                }
                Spacer()
                buttons
            }
        }
        .padding(20)
        .frame(width: 520)
    }

    @ViewBuilder private var buttons: some View {
        switch updater.phase {
        case .ready:
            Button("Later") { updater.showPrompt = false; dismiss() }.keyboardShortcut(.cancelAction)
            Button("Relaunch to Update") { updater.relaunchToUpdate() }.keyboardShortcut(.defaultAction)
        case .available:
            Button("Later") { updater.showPrompt = false; dismiss() }.keyboardShortcut(.cancelAction)
            Button("Download") { updater.downloadAvailable() }.keyboardShortcut(.defaultAction)
        case .downloading:
            Button("Hide") { updater.showPrompt = false; dismiss() }.keyboardShortcut(.defaultAction)
        case .failed:
            if let r = updater.release { Link("Download Manually", destination: r.page) }
            Button("Try Again") { Task { await updater.check(userInitiated: true) } }
            Button("Close") { updater.showPrompt = false; dismiss() }.keyboardShortcut(.defaultAction)
        default:
            Button("OK") { updater.showPrompt = false; dismiss() }.keyboardShortcut(.defaultAction)
        }
    }

    private var showsRelease: Bool {
        switch updater.phase { case .ready, .available, .downloading: return updater.release != nil; default: return false }
    }

    private var title: String {
        let v = updater.release?.version ?? ""
        switch updater.phase {
        case .checking: return "Checking for updates…"
        case .upToDate, .idle: return "SplatWork is up to date"
        case .available: return "SplatWork \(v) is available"
        case .downloading: return "Downloading SplatWork \(v)…"
        case .ready: return "SplatWork \(v) is ready to install"
        case .failed: return "Update failed"
        }
    }

    private var subtitle: String {
        let cur = updater.currentVersion ?? "?"
        switch updater.phase {
        case .checking: return "Looking at the latest release on GitHub."
        case .upToDate, .idle: return "You have version \(cur), the latest release."
        case .available: return "You have \(cur)."
        case .downloading: return "You have \(cur). The download is verified before it's offered."
        case .ready: return "You have \(cur). The download is verified. Relaunch now, or it installs the next time you quit SplatWork."
        case .failed(let m): return m
        }
    }

    static func markdown(_ s: String) -> AttributedString {
        (try? AttributedString(markdown: s, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace))) ?? AttributedString(s)
    }
}
