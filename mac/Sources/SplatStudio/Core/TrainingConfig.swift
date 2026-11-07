import Foundation

/// Training hyper-parameters, shared verbatim with the Linux worker (PROTOCOL.md §3).
/// Every field is documented for users in `SettingsCatalog` (and docs/SETTINGS.md).
struct TrainingConfig: Codable, Equatable {
    enum Strategy: String, Codable, CaseIterable { case mcmc, adaptive }
    enum Background: String, Codable, CaseIterable { case black, white, random }
    struct Region: Codable, Equatable { var min: [Float]; var max: [Float]; var margin: Float = 0.15 }

    var iterations = 30_000
    var maxImageSize = 1600
    var shDegree = 3
    var shDegreeInterval = 1000
    var strategy: Strategy = .mcmc
    var maxGaussians = 1_500_000
    var densifyStart = 500
    var densifyStop = 15_000
    var densifyInterval = 100
    var densifyGradThreshold: Float = 0.0002
    var absGrad = false
    var opacityResetInterval = 3000
    var pruneOpacity: Float = 0.005
    var lrPosition: Float = 0.00016
    var lrPositionFinal: Float = 0.0000016
    var lrColor: Float = 0.0025
    var lrSHRest: Float = 0.000125
    var lrOpacity: Float = 0.05
    var lrScale: Float = 0.005
    var lrRotation: Float = 0.001
    var ssimWeight: Float = 0.2
    var opacityReg: Float = 0.01
    var scaleReg: Float = 0.01
    var noiseLR: Float = 500_000
    var antialiased = true
    var depthWeight: Float = 0.1
    var background: Background = .random
    var initialOpacity: Float = 0.1
    var randomInitCount = 100_000
    var seed = 42
    var previewInterval = 500
    var evalInterval = 2000
    /// Ignore moving / transient objects with robust (RobustNeRF-style) loss masking. Off by default: on real
    /// captures it kept mistaking hard-to-learn detail for movement (held-out PSNR +1–2 dB with it off on two
    /// bedroom captures); turn on for scenes with people or pets moving through.
    var transientMasking = false
    /// A pixel is a transient candidate when its error exceeds this multiple of the image's median error.
    var transientSensitivity: Float = 3
    /// Step at which transient masking starts (residuals are meaningless before the scene forms).
    var transientStart = 1000
    /// Moving-object masking switches itself off when photos are this far apart (median gap to the nearest
    /// view, degrees): with few views per surface, real scenery disagrees between views and gets ignored.
    var transientMaxViewGap: Float = 15
    /// Learn a per-photo colour correction (3×3 colour matrix + offset: exposure, white balance, tint),
    /// so photos taken with different auto-exposure don't fight each other. Seeded from recorded exposure.
    var appearanceCorrection = true
    /// Each photo's correction is only updated when that photo is drawn (once per pass over the dataset),
    /// so this is much higher than the per-splat rates.
    var appearanceLR: Float = 0.02
    /// Pulls each photo's correction towards its starting value (higher = corrections stay smaller).
    var appearanceReg: Float = 0.001
    /// Down-weight LiDAR / stereo depth at image edges, where depth sensors are least reliable.
    var depthEdgeAware = true
    /// Penalty on splats whose longest axis exceeds `anisotropyMaxRatio` × their shortest (needle splats
    /// that streak along view rays in sparsely covered areas). 0 = off.
    var anisotropyReg: Float = 0.1
    var anisotropyMaxRatio: Float = 10
    /// MCMC growth/relocation places copies along the parent's longest axis (ImprovedGS / IGS+) instead of
    /// stacking them on top of it. +2.2 dB held-out on a 649-photo bedroom capture (+3.4 dB together with
    /// moving-object masking off).
    var longAxisSplit = true
    var region: Region? = nil
    var cameraIds: [Int]? = nil

    init() {}

    // Tolerant decoding: missing keys keep their defaults.
    init(from decoder: Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func d<T: Decodable>(_ k: CodingKeys, _ v: inout T) { if let x = try? c.decode(T.self, forKey: k) { v = x } }
        d(.iterations, &iterations); d(.maxImageSize, &maxImageSize); d(.shDegree, &shDegree)
        d(.shDegreeInterval, &shDegreeInterval); d(.strategy, &strategy); d(.maxGaussians, &maxGaussians)
        d(.densifyStart, &densifyStart); d(.densifyStop, &densifyStop); d(.densifyInterval, &densifyInterval)
        d(.densifyGradThreshold, &densifyGradThreshold); d(.absGrad, &absGrad); d(.opacityResetInterval, &opacityResetInterval)
        d(.pruneOpacity, &pruneOpacity); d(.lrPosition, &lrPosition); d(.lrPositionFinal, &lrPositionFinal)
        d(.lrColor, &lrColor); d(.lrSHRest, &lrSHRest); d(.lrOpacity, &lrOpacity); d(.lrScale, &lrScale)
        d(.lrRotation, &lrRotation); d(.ssimWeight, &ssimWeight); d(.opacityReg, &opacityReg); d(.scaleReg, &scaleReg)
        d(.noiseLR, &noiseLR); d(.antialiased, &antialiased); d(.depthWeight, &depthWeight); d(.background, &background)
        d(.initialOpacity, &initialOpacity); d(.randomInitCount, &randomInitCount); d(.seed, &seed)
        d(.previewInterval, &previewInterval); d(.evalInterval, &evalInterval)
        d(.transientMasking, &transientMasking); d(.transientSensitivity, &transientSensitivity); d(.transientStart, &transientStart)
        d(.transientMaxViewGap, &transientMaxViewGap)
        d(.depthEdgeAware, &depthEdgeAware); d(.anisotropyReg, &anisotropyReg); d(.anisotropyMaxRatio, &anisotropyMaxRatio); d(.longAxisSplit, &longAxisSplit)
        d(.appearanceCorrection, &appearanceCorrection); d(.appearanceLR, &appearanceLR); d(.appearanceReg, &appearanceReg)
        region = try? c.decode(Region.self, forKey: .region)
        cameraIds = try? c.decode([Int].self, forKey: .cameraIds)
    }

    /// Easy-mode presets. They only touch the handful of settings that matter most.
    enum Preset: String, CaseIterable, Identifiable, Codable {
        case preview, balanced, high, ultra
        var id: String { rawValue }
        var title: String {
            switch self {
            case .preview: return "Quick preview"
            case .balanced: return "Balanced"
            case .high: return "High quality"
            case .ultra: return "Maximum"
            }
        }
        var subtitle: String {
            switch self {
            case .preview: return "≈2–4 min on M4 Pro. Check coverage before a long run."
            case .balanced: return "≈10–15 min. Good for sharing and most scenes."
            case .high: return "≈25–40 min. Sharp detail, 1.5 M splats."
            case .ultra: return "≈1 h+. Full resolution, 3 M splats, needs lots of RAM."
            }
        }
        var config: TrainingConfig {
            var c = TrainingConfig()
            switch self {
            case .preview:
                c.iterations = 5000; c.maxImageSize = 1000; c.maxGaussians = 300_000; c.shDegree = 1; c.densifyStop = 4000
            case .balanced:
                c.iterations = 15_000; c.maxImageSize = 1400; c.maxGaussians = 800_000; c.densifyStop = 12_000
            case .high:
                c.iterations = 30_000; c.maxImageSize = 1600; c.maxGaussians = 1_500_000; c.densifyStop = 25_000
            case .ultra:
                c.iterations = 40_000; c.maxImageSize = 2400; c.maxGaussians = 3_000_000; c.densifyStop = 30_000
            }
            return c
        }
    }
}
