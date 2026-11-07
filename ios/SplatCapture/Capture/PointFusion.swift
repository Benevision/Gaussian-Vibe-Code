import CoreVideo
import Foundation
import simd

/// Points prepared for the live overlay (positions & colours as packed Float32 xyz / rgb 0..1).
struct PointSnapshot: Sendable {
    let positions: Data
    let colors: Data
    let count: Int
}

/// Voxel-downsampled, confidence-filtered, coloured point accumulation.
/// Not thread-safe: owned and used by the capture writer queue only.
final class PointFusion {
    private var voxel: Float
    private let minConfidence: UInt8
    private let maxDepth: Float
    private let maxVoxels: Int

    private var index: [Int64: Int32] = [:]
    /// xyz sums + count in w.
    private var sums: [SIMD4<Float>] = []
    private var colorSums: [SIMD3<Float>] = []

    var count: Int { sums.count }

    init(voxelSize: Float, minConfidence: UInt8, maxDepth: Float, maxVoxels: Int = 1_500_000) {
        self.voxel = max(0.003, voxelSize)
        self.minConfidence = minConfidence
        self.maxDepth = maxDepth
        self.maxVoxels = maxVoxels
        index.reserveCapacity(200_000)
    }

    /// - Parameters:
    ///   - intrinsics: fx, fy, cx, cy for an image of `imageSize` that is pixel-aligned with the depth map.
    ///   - color: the YCbCr colour buffer (may be a different resolution than `imageSize`; it is rescaled).
    func integrate(depth: DepthSample, intrinsics: SIMD4<Float>, imageSize: SIMD2<Float>,
                   cameraToWorld: simd_float4x4, color: CVPixelBuffer) {
        CVPixelBufferLockBaseAddress(color, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(color, .readOnly) }
        guard let sampler = YCbCrSampler(lockedBuffer: color) else { return }

        let sx = imageSize.x / Float(depth.width), sy = imageSize.y / Float(depth.height)
        let cxs = Float(sampler.width) / imageSize.x, cys = Float(sampler.height) / imageSize.y
        let fx = intrinsics.x, fy = intrinsics.y, cx = intrinsics.z, cy = intrinsics.w
        // Budget full: merge into voxels twice the size and keep going, so newly scanned areas still get
        // points (previously fusion simply stopped at maxVoxels: no dots, and no training points there).
        if sums.count >= maxVoxels { coarsen() }
        let inv = 1 / voxel
        let full = sums.count >= maxVoxels

        for v in 0..<depth.height {
            for u in 0..<depth.width {
                let i = v * depth.width + u
                let d = depth.depth[i]
                guard d > 0.1, d < maxDepth else { continue }
                if let conf = depth.confidence, conf[i] < minConfidence { continue }
                let px = (Float(u) + 0.5) * sx, py = (Float(v) + 0.5) * sy
                // ARKit camera: +X right, +Y up, looking down -Z. Image y grows downward.
                let pc = SIMD4<Float>((px - cx) / fx * d, -(py - cy) / fy * d, -d, 1)
                let pw = cameraToWorld * pc
                let ix = Int64((pw.x * inv).rounded(.down))
                let iy = Int64((pw.y * inv).rounded(.down))
                let iz = Int64((pw.z * inv).rounded(.down))
                let key = Self.key(ix, iy, iz)
                let rgb = sampler.rgb(x: Int(px * cxs), y: Int(py * cys))
                if let slot = index[key] {
                    let s = Int(slot)
                    // Cap per-voxel weight so late observations can still refine colour a bit.
                    if sums[s].w < 64 {
                        sums[s] += SIMD4(pw.x, pw.y, pw.z, 1)
                        colorSums[s] += rgb
                    }
                } else if !full {
                    index[key] = Int32(sums.count)
                    sums.append(SIMD4(pw.x, pw.y, pw.z, 1))
                    colorSums.append(rgb)
                }
            }
        }
    }

    /// Doubles the voxel size, merging existing voxels (sums and colour sums add up).
    private func coarsen() {
        voxel *= 2
        let inv = 1 / voxel
        var newIndex: [Int64: Int32] = [:]
        newIndex.reserveCapacity(sums.count / 4)
        var newSums: [SIMD4<Float>] = [], newColors: [SIMD3<Float>] = []
        newSums.reserveCapacity(sums.count / 4); newColors.reserveCapacity(sums.count / 4)
        for (k, s) in sums.enumerated() {
            let c = SIMD3(s.x, s.y, s.z) / max(s.w, 1)
            let key = Self.key(Int64((c.x * inv).rounded(.down)), Int64((c.y * inv).rounded(.down)), Int64((c.z * inv).rounded(.down)))
            if let slot = newIndex[key] {
                newSums[Int(slot)] += s; newColors[Int(slot)] += colorSums[k]
            } else {
                newIndex[key] = Int32(newSums.count); newSums.append(s); newColors.append(colorSums[k])
            }
        }
        index = newIndex; sums = newSums; colorSums = newColors
    }

    private static func key(_ ix: Int64, _ iy: Int64, _ iz: Int64) -> Int64 {
        ((ix & 0x1FFFFF) << 42) | ((iy & 0x1FFFFF) << 21) | (iz & 0x1FFFFF)
    }

    /// A strided subset for display.
    func snapshot(maxPoints: Int) -> PointSnapshot {
        let n = sums.count
        let stride = max(1, (n + maxPoints - 1) / max(1, maxPoints))
        let m = (n + stride - 1) / stride
        var pos = Data(count: m * 12)
        var col = Data(count: m * 12)
        pos.withUnsafeMutableBytes { pr in
            col.withUnsafeMutableBytes { cr in
                let p = pr.bindMemory(to: Float.self)
                let c = cr.bindMemory(to: Float.self)
                var j = 0
                var i = 0
                while i < n && j < m {
                    let s = sums[i], w = s.w
                    p[j * 3] = s.x / w; p[j * 3 + 1] = s.y / w; p[j * 3 + 2] = s.z / w
                    let cc = colorSums[i] / (w * 255)
                    c[j * 3] = cc.x; c[j * 3 + 1] = cc.y; c[j * 3 + 2] = cc.z
                    j += 1
                    i += stride
                }
            }
        }
        return PointSnapshot(positions: pos, colors: col, count: m)
    }

    /// Writes `binary_little_endian` PLY: x y z float, red green blue uchar.
    func writePLY(to url: URL) throws {
        let n = sums.count
        let header = """
        ply
        format binary_little_endian 1.0
        comment Splat Capture fused LiDAR point cloud (ARKit world, metres)
        element vertex \(n)
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header

        """
        FileManager.default.createFile(atPath: url.path, contents: nil)
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        try handle.write(contentsOf: Data(header.utf8))

        let chunk = 65_536
        var buffer = Data(capacity: chunk * 15)
        var start = 0
        while start < n {
            let end = min(n, start + chunk)
            buffer.removeAll(keepingCapacity: true)
            for i in start..<end {
                let s = sums[i], w = s.w
                appendFloat(&buffer, s.x / w)
                appendFloat(&buffer, s.y / w)
                appendFloat(&buffer, s.z / w)
                let c = colorSums[i] / w
                buffer.append(UInt8(min(255, max(0, c.x.rounded()))))
                buffer.append(UInt8(min(255, max(0, c.y.rounded()))))
                buffer.append(UInt8(min(255, max(0, c.z.rounded()))))
            }
            try handle.write(contentsOf: buffer)
            start = end
        }
    }

    private func appendFloat(_ data: inout Data, _ v: Float) {
        var le = v.bitPattern.littleEndian
        withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
    }
}
