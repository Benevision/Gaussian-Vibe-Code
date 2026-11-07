import SwiftUI

@main
struct SplatCaptureApp: App {
    @State private var settings = AppSettings()
    @State private var library = CaptureLibrary()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(settings)
                .environment(library)
        }
    }
}

struct RootView: View {
    var body: some View {
        if DeviceInfo.isARSupported {
            CaptureView()
        } else {
            ContentUnavailableView(
                "ARKit not available",
                systemImage: "arkit",
                description: Text("Splat Capture needs an iPhone with ARKit world tracking. It can't record in the Simulator."))
        }
    }
}
