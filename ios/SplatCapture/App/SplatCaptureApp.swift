import SwiftUI

@main
struct SplatCaptureApp: App {
    @State private var settings = AppSettings()
    @State private var library = CaptureLibrary()
    @State private var computeWorker = ComputeWorker()
    @State private var splatJobs = SplatJobs()
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(settings)
                .environment(library)
                .environment(computeWorker)
                .environment(splatJobs)
        }
        .onChange(of: scenePhase) { _, phase in
            switch phase {
            case .background: computeWorker.appDidEnterBackground(); splatJobs.appDidEnterBackground()
            case .active: computeWorker.appDidBecomeActive()
            default: break
            }
        }
    }
}

struct RootView: View {
    @State private var showCompute = false
    @State private var selfTestResult: SplatOutput?
    @State private var makeSplatItem: CaptureItem?
    @Environment(CaptureLibrary.self) private var library

    var body: some View {
        content
            #if DEBUG
            .task {
                if SplatSelfTest.isRequested { await SplatSelfTest.run(library: library) { selfTestResult = $0 } }
                if ProcessInfo.processInfo.arguments.contains("-showMakeSplat") { await library.reload(); makeSplatItem = library.items.first }
            }
            .fullScreenCover(item: $selfTestResult) { SplatViewerScreen(output: $0, title: "Self test") }
            .fullScreenCover(item: $makeSplatItem) { MakeSplatView(item: $0) }
            #endif
    }

    @ViewBuilder private var content: some View {
        if DeviceInfo.isARSupported {
            CaptureView()
        } else {
            ContentUnavailableView {
                Label("ARKit not available", systemImage: "arkit")
            } description: {
                Text("SplatWork Capture needs an iPhone with ARKit world tracking. It can't record in the Simulator.")
            } actions: {
                Button("Compute…") { showCompute = true }
            }
            .sheet(isPresented: $showCompute) { ComputeView() }
        }
    }
}
