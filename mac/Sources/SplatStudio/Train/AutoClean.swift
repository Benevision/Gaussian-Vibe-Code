import Foundation

/// What Auto clean proposes to remove, from a contribution census over every photo
/// (`MetalTrainer.contributionCensus`).
///
/// - Invisible: never adds `peakThreshold` (2 %) to any pixel of any photo. Removing these leaves every
///   photo's view unchanged — on MCMC results that is often most of the file.
/// - Seen by too few photos: clearly visible (≥ `peakThreshold`) in fewer than `fewViewsFraction` of the
///   photos a typical visible splat is. A real surface is drawn by most photos that look its way; a blob hanging
///   just in front of a few cameras — which fits those photos and blocks the view from anywhere else — only by
///   those few. On a 1,258-photo drone survey (typical 198 photos) a quarter caught every floater more than
///   25 m above the ground and two thirds of those at 20–25 m, while touching 0.2 % of ground and tree splats.
enum AutoClean {
    static let peakThreshold: Float = 0.02
    static let fewViewsFraction: Float = 0.25
    /// Below this typical count (small captures) the few-photos rule is skipped: every splat is "few".
    static let minTypicalViews: Float = 12

    struct Plan {
        var remove: [Bool]
        var invisible: Int
        var fewViews: Int
        /// Photos a typical clearly visible splat contributes to (0 = few-photos rule not applied).
        var typicalViews: Int
    }

    static func plan(peak: [Float], views: [UInt32], opacityLogits: [Float],
                     fewViewsFraction f: Float = fewViewsFraction) -> Plan {
        let n = min(peak.count, views.count)
        var sample: [UInt32] = []
        let stride = max(1, n / 300_000)
        for i in Swift.stride(from: 0, to: n, by: stride) where peak[i] >= peakThreshold {
            let o = i < opacityLogits.count ? 1 / (1 + exp(-opacityLogits[i])) : 1
            if o > 0.2 { sample.append(views[i]) }
        }
        sample.sort()
        let typical = sample.isEmpty ? 0 : Float(sample[sample.count / 2])
        let cut = typical >= minTypicalViews && f > 0 ? f * typical : 0
        var remove = [Bool](repeating: false, count: n)
        var inv = 0, few = 0
        for i in 0..<n {
            if peak[i] < peakThreshold { remove[i] = true; inv += 1 }
            else if Float(views[i]) < cut { remove[i] = true; few += 1 }
        }
        return Plan(remove: remove, invisible: inv, fewViews: few, typicalViews: cut > 0 ? Int(typical) : 0)
    }
}
