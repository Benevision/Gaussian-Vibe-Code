// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "SplatStudio",
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "SplatStudio", targets: ["SplatStudio"]),
    ],
    targets: [
        .executableTarget(
            name: "SplatStudio",
            path: "Sources/SplatStudio",
            swiftSettings: [
                .swiftLanguageMode(.v5),
                .unsafeFlags(["-Ounchecked", "-wmo"], .when(configuration: .release)),
            ],
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("MetalKit"),
                .linkedFramework("RealityKit"),
                .linkedFramework("AVFoundation"),
                .linkedFramework("Network"),
                .linkedFramework("Accelerate"),
            ]
        ),
    ]
)
