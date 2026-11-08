import SwiftUI
import UIKit

struct LibraryView: View {
    @Environment(CaptureLibrary.self) private var library
    @Environment(\.dismiss) private var dismiss
    @State private var renaming: CaptureItem?
    @State private var newName = ""
    @State private var deleting: CaptureItem?
    @State private var sending: CaptureItem?
    @State private var exporter = ZipExporter()
    @State private var errorMessage: String?

    var body: some View {
        NavigationStack {
            Group {
                if library.items.isEmpty && !library.isLoading {
                    ContentUnavailableView {
                        Label("No captures yet", systemImage: "camera.metering.matrix")
                    } description: {
                        Text("Close the library and press the record button to make your first capture.")
                    }
                } else {
                    list
                }
            }
            .navigationTitle("Captures")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Close") { dismiss() } }
            }
            .navigationDestination(for: CaptureItem.self) { item in
                CaptureDetailView(item: item, onSend: { sending = $0 }, onShare: { share($0) },
                                  onRename: { beginRename($0) }, onDelete: { deleting = $0 })
            }
            .task { await library.reload() }
            .refreshable { await library.reload() }
            .alert("Rename capture", isPresented: Binding(get: { renaming != nil }, set: { if !$0 { renaming = nil } })) {
                TextField("Name", text: $newName)
                Button("Cancel", role: .cancel) {}
                Button("Rename") {
                    if let r = renaming {
                        do { try library.rename(r, to: newName) } catch { errorMessage = error.localizedDescription }
                    }
                }
            }
            .confirmationDialog(deleting.map { "Delete “\($0.name)”? This can't be undone." } ?? "",
                                isPresented: Binding(get: { deleting != nil }, set: { if !$0 { deleting = nil } }),
                                titleVisibility: .visible) {
                Button("Delete", role: .destructive) {
                    if let d = deleting {
                        do { try library.delete(d) } catch { errorMessage = error.localizedDescription }
                    }
                }
            }
            .sheet(item: $sending) { SendToMacView(item: $0) }
            .sheet(item: Binding(get: { exporter.readyURL.map(ShareFile.init) },
                                 set: { if $0 == nil { exporter.finishedSharing() } })) { file in
                ActivityView(items: [file.url])
                    .ignoresSafeArea()
            }
            .overlay {
                if exporter.isZipping {
                    VStack(spacing: 12) {
                        ProgressView(value: exporter.progress) { Text("Preparing .zip…") }
                            .frame(width: 220)
                        Button("Cancel") { exporter.cancel() }
                    }
                    .padding(24)
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
                }
            }
            .alert("Error", isPresented: Binding(get: { errorMessage != nil || exporter.error != nil },
                                                 set: { if !$0 { errorMessage = nil; exporter.error = nil } })) {
                Button("OK", role: .cancel) {}
            } message: {
                Text(errorMessage ?? exporter.error ?? "")
            }
        }
    }

    private var list: some View {
        List {
            Section {
                ForEach(library.items) { item in
                    NavigationLink(value: item) { CaptureRow(item: item) }
                        .swipeActions(edge: .trailing) {
                            Button(role: .destructive) { deleting = item } label: { Label("Delete", systemImage: "trash") }
                            Button { beginRename(item) } label: { Label("Rename", systemImage: "pencil") }.tint(.blue)
                        }
                        .swipeActions(edge: .leading) {
                            Button { sending = item } label: { Label("Send", systemImage: "desktopcomputer") }.tint(.green)
                        }
                        .contextMenu {
                            Button { sending = item } label: { Label("Send to Mac", systemImage: "desktopcomputer.and.arrow.down") }
                            Button { share(item) } label: { Label("Share as .zip", systemImage: "square.and.arrow.up") }
                            Button { beginRename(item) } label: { Label("Rename", systemImage: "pencil") }
                            Button(role: .destructive) { deleting = item } label: { Label("Delete", systemImage: "trash") }
                        }
                }
            } footer: {
                Text("\(library.items.count) captures · \(library.totalBytes.formattedBytes) · Free: \(CaptureStore.freeSpace.formattedBytes). Captures are also visible in the Files app under On My iPhone › SplatWork Capture › Captures.")
            }
        }
    }

    private func beginRename(_ item: CaptureItem) {
        newName = item.name
        renaming = item
    }

    private func share(_ item: CaptureItem) {
        exporter.export(item)
    }
}

struct ShareFile: Identifiable {
    let url: URL
    var id: URL { url }
}

struct CaptureRow: View {
    let item: CaptureItem

    var body: some View {
        HStack(spacing: 12) {
            ThumbnailView(url: item.thumbnailURL)
                .frame(width: 64, height: 86)
                .clipShape(RoundedRectangle(cornerRadius: 8))
            VStack(alignment: .leading, spacing: 4) {
                Text(item.name).font(.headline).lineLimit(2)
                Text(item.createdAt.formatted(date: .abbreviated, time: .shortened))
                    .font(.subheadline).foregroundStyle(.secondary)
                HStack(spacing: 10) {
                    Label("\(item.frameCount)", systemImage: "photo.stack")
                    Label(item.bytes.formattedBytes, systemImage: "internaldrive")
                    if item.hasLiDAR { Image(systemName: "dot.radiowaves.forward").accessibilityLabel("LiDAR") }
                }
                .font(.caption).foregroundStyle(.secondary)
                .labelStyle(.titleAndIcon)
            }
        }
        .padding(.vertical, 2)
    }
}

struct ThumbnailView: View {
    let url: URL
    @State private var image: UIImage?

    var body: some View {
        ZStack {
            Rectangle().fill(.quaternary)
            if let image {
                Image(uiImage: image).resizable().scaledToFill()
            } else {
                Image(systemName: "camera.metering.matrix").foregroundStyle(.secondary)
            }
        }
        .task(id: url) {
            let u = url
            let loaded = await Task.detached(priority: .utility) { () -> UIImage? in
                guard let data = try? Data(contentsOf: u) else { return nil }
                return UIImage(data: data)?.preparingForDisplay()
            }.value
            image = loaded
        }
    }
}

struct CaptureDetailView: View {
    let item: CaptureItem
    let onSend: (CaptureItem) -> Void
    let onShare: (CaptureItem) -> Void
    let onRename: (CaptureItem) -> Void
    let onDelete: (CaptureItem) -> Void
    @Environment(CaptureLibrary.self) private var library

    /// The library item may have been renamed; always show the freshest copy.
    private var current: CaptureItem { library.items.first { $0.createdAt == item.createdAt && $0.frameCount == item.frameCount } ?? item }

    var body: some View {
        List {
            Section {
                ThumbnailView(url: current.thumbnailURL)
                    .frame(maxWidth: .infinity)
                    .frame(height: 280)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .listRowInsets(EdgeInsets())
            }
            Section("Details") {
                LabeledContent("Created", value: current.createdAt.formatted(date: .long, time: .shortened))
                LabeledContent("Keyframes", value: "\(current.frameCount)")
                LabeledContent("Size", value: current.bytes.formattedBytes)
                if let r = current.resolution { LabeledContent("Resolution", value: r) }
                LabeledContent("LiDAR depth", value: current.hasLiDAR ? "Yes" : "No")
                LabeledContent("Point cloud", value: current.hasPointCloud ? "Yes" : "No")
                if let d = current.device { LabeledContent("Device", value: d) }
            }
            Section {
                Button { onSend(current) } label: { Label("Send to Mac", systemImage: "desktopcomputer.and.arrow.down") }
                Button { onShare(current) } label: { Label("Share as .zip (AirDrop, Files…)", systemImage: "square.and.arrow.up") }
                Button { onRename(current) } label: { Label("Rename", systemImage: "pencil") }
                Button(role: .destructive) { onDelete(current) } label: { Label("Delete", systemImage: "trash") }
            }
        }
        .navigationTitle(current.name)
        .navigationBarTitleDisplayMode(.inline)
    }
}

/// Builds a temporary .zip for the share sheet.
@MainActor
@Observable
final class ZipExporter {
    private(set) var isZipping = false
    private(set) var progress = 0.0
    private(set) var readyURL: URL?
    var error: String?
    private var task: Task<Void, Never>?

    func export(_ item: CaptureItem) {
        guard !isZipping else { return }
        isZipping = true
        progress = 0
        let dest = CaptureStore.exportsDirectory.appendingPathComponent("\(item.url.deletingPathExtension().lastPathComponent).zip")
        task = Task { [weak self] in
            do {
                try await CaptureUploader.zip(item.url, to: dest) { p in
                    Task { @MainActor in self?.progress = p }
                }
                try Task.checkCancellation()
                self?.readyURL = dest
            } catch {
                try? FileManager.default.removeItem(at: dest)
                if !(error is CancellationError) && !Task.isCancelled {
                    self?.error = error.localizedDescription
                }
            }
            self?.isZipping = false
        }
    }

    func cancel() {
        task?.cancel()
    }

    func finishedSharing() {
        if let u = readyURL { try? FileManager.default.removeItem(at: u) }
        readyURL = nil
    }
}

struct ActivityView: UIViewControllerRepresentable {
    let items: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }

    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
}
