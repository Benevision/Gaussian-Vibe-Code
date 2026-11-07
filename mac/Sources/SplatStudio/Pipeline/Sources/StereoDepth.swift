import Foundation
import CoreGraphics

/// Options for stereo depth estimation on rectified left/right pairs.
struct StereoDepthOptions: Codable, Hashable, Sendable {
    /// Width of the images used for matching (depth map resolution). Height follows the aspect ratio.
    var matchWidth: Int = 640
    /// Search range in pixels *at matchWidth*. Objects closer than f·B/maxDisparity are not recovered.
    var maxDisparity: Int = 96
    /// Half-size of the aggregation window (window = 2r+1 square).
    var blockRadius: Int = 4
    /// Left-right consistency tolerance in pixels (larger = more, but noisier, depth).
    var lrTolerance: Float = 1
    /// Reject pixels whose best cost is not clearly better than the second-best (0 = off, 0.1 = 10 %).
    var uniqueness: Float = 0.08
    /// Clamp depth to this range in metres; outside → 0 (invalid).
    var minDepth: Float = 0.1
    var maxDepth: Float = 50

    init() {}
    init(from decoder: Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        matchWidth = c(.matchWidth, matchWidth); maxDisparity = c(.maxDisparity, maxDisparity); blockRadius = c(.blockRadius, blockRadius)
        lrTolerance = c(.lrTolerance, lrTolerance); uniqueness = c(.uniqueness, uniqueness)
        minDepth = c(.minDepth, minDepth); maxDepth = c(.maxDepth, maxDepth)
    }
}

/// Census-transform block matching with box aggregation, sub-pixel refinement, uniqueness test and
/// left-right consistency check (multi-core CPU).
///
/// Quality notes: this is a classic local stereo method — robust to exposure differences between the eyes
/// (census), sharp at textured surfaces, but it leaves holes (0 = invalid) in untextured regions, at
/// occlusions and on reflective surfaces, and fattens foreground edges by ~`blockRadius` pixels. Depth is
/// meant as *supervision with a small weight* for splat training, not as a dense ground truth. It assumes
/// the pair is rectified (epipolar lines horizontal), which holds for SBS/MV-HEVC footage from parallel rigs.
/// Runtime ≈ 40–80 ms per 640×360 pair on M-series CPUs.
enum StereoDepth {
    /// Returns a disparity map in pixels at the input resolution (0 = invalid).
    static func disparity(left L: [UInt8], right R: [UInt8], width w: Int, height h: Int, options o: StereoDepthOptions) -> [Float] {
        let D = max(4, min(o.maxDisparity, w / 2))
        let cl = census(L, w, h), cr = census(R, w, h)
        let n = w * h
        let r = o.blockRadius
        // Aggregated cost volume, d-major: cost[d * n + i]
        var cost = [UInt16](repeating: UInt16.max, count: D * n)
        cost.withUnsafeMutableBufferPointer { cp in
            let cbase = cp.baseAddress!
            cl.withUnsafeBufferPointer { lp in cr.withUnsafeBufferPointer { rp in
                let lb = lp.baseAddress!, rb = rp.baseAddress!
                DispatchQueue.concurrentPerform(iterations: D) { d in
                    // raw Hamming cost (invalid where x-d < 0 → max cost 24)
                    var raw = [UInt16](repeating: 24, count: n)
                    for y in 0..<h {
                        let row = y * w
                        for x in d..<w { raw[row + x] = UInt16((lb[row + x] ^ rb[row + x - d]).nonzeroBitCount) }
                    }
                    // separable box sum (horizontal then vertical)
                    var tmp = [UInt16](repeating: 0, count: n)
                    for y in 0..<h {
                        let row = y * w
                        var s = 0
                        for x in 0...min(r, w - 1) { s += Int(raw[row + x]) }
                        for x in 0..<w {
                            tmp[row + x] = UInt16(min(s, Int(UInt16.max)))
                            let add = x + r + 1, sub = x - r
                            if add < w { s += Int(raw[row + add]) }
                            if sub >= 0 { s -= Int(raw[row + sub]) }
                        }
                    }
                    let out = cbase + d * n
                    for x in 0..<w {
                        var s = 0
                        for y in 0...min(r, h - 1) { s += Int(tmp[y * w + x]) }
                        for y in 0..<h {
                            out[y * w + x] = UInt16(min(s, Int(UInt16.max)))
                            let add = y + r + 1, sub = y - r
                            if add < h { s += Int(tmp[add * w + x]) }
                            if sub >= 0 { s -= Int(tmp[sub * w + x]) }
                        }
                    }
                }
            } }
        }
        // Winner-take-all for left and right views.
        var dispL = [Float](repeating: 0, count: n)
        var dispR = [Int](repeating: -1, count: n)
        cost.withUnsafeBufferPointer { cp in
            let c = cp.baseAddress!
            dispL.withUnsafeMutableBufferPointer { dl in dispR.withUnsafeMutableBufferPointer { dr in
                let dlb = dl.baseAddress!, drb = dr.baseAddress!
                DispatchQueue.concurrentPerform(iterations: h) { y in
                    for x in 0..<w {
                        let i = y * w + x
                        var best = Int.max, second = Int.max, bd = 0
                        let maxD = min(D - 1, x)
                        if maxD < 1 { dlb[i] = 0; continue }
                        for d in 0...maxD {
                            let v = Int(c[d * n + i])
                            if v < best { second = best; best = v; bd = d } else if v < second { second = v }
                        }
                        if o.uniqueness > 0, second != Int.max, Float(second - best) < o.uniqueness * Float(best) { dlb[i] = 0; continue }
                        var sub = Float(bd)
                        if bd > 0 && bd < maxD {
                            let c0 = Float(c[(bd - 1) * n + i]), c1 = Float(best), c2 = Float(c[(bd + 1) * n + i])
                            let den = c0 - 2 * c1 + c2
                            if den > 0 { sub += 0.5 * (c0 - c2) / den }
                        }
                        dlb[i] = max(sub, 0.0001)
                    }
                    // Right view: pixel xr matches left xl = xr + d.
                    for xr in 0..<w {
                        var best = Int.max, bd = -1
                        let maxD = min(D - 1, w - 1 - xr)
                        if maxD < 0 { continue }
                        for d in 0...maxD {
                            let v = Int(c[d * n + y * w + xr + d])
                            if v < best { best = v; bd = d }
                        }
                        drb[y * w + xr] = bd
                    }
                }
            } }
        }
        // Left-right consistency.
        for y in 0..<h {
            for x in 0..<w {
                let i = y * w + x
                let d = dispL[i]
                guard d > 0 else { continue }
                let xr = x - Int(d.rounded())
                if xr < 0 || dispR[y * w + xr] < 0 || abs(Float(dispR[y * w + xr]) - d) > o.lrTolerance { dispL[i] = 0 }
            }
        }
        return median3(dispL, w, h)
    }

    /// Depth (metres, Z) for the left image at matching resolution. `focal` must be in pixels at `width`.
    static func depth(left: CGImage, right: CGImage, focal: Float, baseline: Float, options o: StereoDepthOptions)
        -> (depth: [Float], width: Int, height: Int) {
        let w = min(o.matchWidth, left.width)
        let h = max(8, Int((Double(left.height) * Double(w) / Double(left.width)).rounded()))
        let L = ImageUtil.gray8(left, width: w, height: h), R = ImageUtil.gray8(right, width: w, height: h)
        let disp = disparity(left: L, right: R, width: w, height: h, options: o)
        let fb = focal * Float(w) / Float(left.width) * baseline
        let depth = disp.map { d -> Float in
            guard d > 0 else { return 0 }
            let z = fb / d
            return (z >= o.minDepth && z <= o.maxDepth) ? z : 0
        }
        return (depth, w, h)
    }

    /// 5×5 census transform (24 bits).
    static func census(_ img: [UInt8], _ w: Int, _ h: Int) -> [UInt32] {
        var out = [UInt32](repeating: 0, count: w * h)
        img.withUnsafeBufferPointer { ip in out.withUnsafeMutableBufferPointer { op in
            let s = ip.baseAddress!, o = op.baseAddress!
            DispatchQueue.concurrentPerform(iterations: h) { y in
                for x in 0..<w {
                    let c = s[y * w + x]
                    var bits: UInt32 = 0
                    for dy in -2...2 {
                        let yy = min(max(y + dy, 0), h - 1)
                        for dx in -2...2 where dx != 0 || dy != 0 {
                            let xx = min(max(x + dx, 0), w - 1)
                            bits = (bits << 1) | (s[yy * w + xx] < c ? 1 : 0)
                        }
                    }
                    o[y * w + x] = bits
                }
            }
        } }
        return out
    }

    /// 3×3 median over valid (non-zero) values; isolated pixels are removed.
    static func median3(_ d: [Float], _ w: Int, _ h: Int) -> [Float] {
        var out = d
        guard w > 2, h > 2 else { return out }
        for y in 1..<(h - 1) {
            for x in 1..<(w - 1) {
                var v: [Float] = []
                v.reserveCapacity(9)
                for dy in -1...1 { for dx in -1...1 { let s = d[(y + dy) * w + x + dx]; if s > 0 { v.append(s) } } }
                if v.count < 4 { out[y * w + x] = 0; continue }
                if d[y * w + x] == 0 && v.count < 7 { continue }
                v.sort(); out[y * w + x] = v[v.count / 2]
            }
        }
        return out
    }
}
