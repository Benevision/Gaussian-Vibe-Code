import SwiftUI

@main
struct SplatCaptureApp: App {
    @State private var settings = AppSettings()
    @State private var library = CaptureLibrary()
    @State private var computeWorker = ComputeWorker()
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(settings)
                .environment(library)
                .environment(computeWorker)
        }
        .onChange(of: scenePhase) { _, phase in
            switch phase {
            case .background: computeWorker.appDidEnterBackground()
            case .active: computeWorker.appDidBecomeActive()
            default: break
            }
        }
    }
}

struct RootView: View {
    @State private var showCompute = false

    var body: some View {
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
