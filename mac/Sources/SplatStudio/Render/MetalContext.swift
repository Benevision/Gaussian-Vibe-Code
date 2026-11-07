import Foundation
import Metal

/// Process-wide Metal state: device, queue and the compiled shader library.
final class MetalContext: @unchecked Sendable {
    static let shared: MetalContext = {
        do { return try MetalContext() } catch { fatalError("Metal initialisation failed: \(error)") }
    }()

    let device: MTLDevice
    let queue: MTLCommandQueue
    let library: MTLLibrary
    private var pipelines: [String: MTLComputePipelineState] = [:]
    private let lock = NSLock()

    init() throws {
        guard let d = MTLCreateSystemDefaultDevice() else { throw FormatError.unsupported("no Metal device") }
        device = d
        guard let q = d.makeCommandQueue() else { throw FormatError.unsupported("no command queue") }
        queue = q
        let opts = MTLCompileOptions()
        opts.mathMode = .fast
        opts.languageVersion = .version3_1
        library = try d.makeLibrary(source: ShaderSource.all, options: opts)
    }

    func pipeline(_ name: String) -> MTLComputePipelineState {
        lock.lock(); defer { lock.unlock() }
        if let p = pipelines[name] { return p }
        guard let f = library.makeFunction(name: name) else { fatalError("missing kernel \(name)") }
        let p = try! device.makeComputePipelineState(function: f)
        pipelines[name] = p
        return p
    }

    func buffer(_ length: Int, label: String? = nil) -> MTLBuffer {
        let b = device.makeBuffer(length: max(length, 16), options: .storageModeShared)!
        b.label = label
        return b
    }

    func buffer<T>(_ array: [T], label: String? = nil) -> MTLBuffer {
        let b = array.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: max($0.count, 16), options: .storageModeShared)! }
        b.label = label
        return b
    }

    /// Memory the GPU may use before it starts to page (≈ 75 % of RAM on Apple silicon).
    var recommendedWorkingSet: UInt64 { device.recommendedMaxWorkingSetSize }
}

extension MTLComputeCommandEncoder {
    func dispatch1D(_ pso: MTLComputePipelineState, count: Int, threadsPerGroup: Int = 256) {
        setComputePipelineState(pso)
        guard count > 0 else { return }
        let tg = min(threadsPerGroup, pso.maxTotalThreadsPerThreadgroup)
        dispatchThreads(MTLSize(width: count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
    }
    func dispatchGroups(_ pso: MTLComputePipelineState, groups: Int, threadsPerGroup: Int) {
        setComputePipelineState(pso)
        guard groups > 0 else { return }
        dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))
    }
    func set<T>(_ value: T, _ index: Int) {
        var v = value
        setBytes(&v, length: MemoryLayout<T>.stride, index: index)
    }
}

/// GPU exclusive prefix scan over uint32 (multi-level).
final class GPUScan {
    private let ctx: MetalContext
    private var levels: [(sums: MTLBuffer, scanned: MTLBuffer, capacity: Int)] = []
    static let block = 1024

    init(ctx: MetalContext = .shared) { self.ctx = ctx }

    /// `output[i] = sum(input[0..<i])`. In-place (input === output) is allowed.
    func encode(_ enc: MTLComputeCommandEncoder, input: MTLBuffer, output: MTLBuffer, count: Int, inputOffset: Int = 0, level: Int = 0) {
        guard count > 0 else { return }
        let blocks = (count + Self.block - 1) / Self.block
        if levels.count <= level {
            levels.append((ctx.buffer(blocks * 4), ctx.buffer(blocks * 4), blocks))
        } else if levels[level].capacity < blocks {
            levels[level] = (ctx.buffer(blocks * 4), ctx.buffer(blocks * 4), blocks)
        }
        let lv = levels[level]
        enc.setComputePipelineState(ctx.pipeline("scan_block"))
        enc.setBuffer(input, offset: inputOffset, index: 0)
        enc.setBuffer(output, offset: 0, index: 1)
        enc.setBuffer(lv.sums, offset: 0, index: 2)
        enc.set(UInt32(count), 3)
        enc.dispatchThreadgroups(MTLSize(width: blocks, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        if blocks > 1 {
            enc.memoryBarrier(scope: .buffers)
            encode(enc, input: lv.sums, output: lv.scanned, count: blocks, level: level + 1)
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(ctx.pipeline("scan_add"))
            enc.setBuffer(output, offset: 0, index: 0)
            enc.setBuffer(lv.scanned, offset: 0, index: 1)
            enc.set(UInt32(count), 2)
            enc.dispatch1D(ctx.pipeline("scan_add"), count: count)
        }
    }
}

/// Stable GPU LSD radix sort of (uint32 key, uint32 value) pairs.
final class GPURadixSort {
    private let ctx: MetalContext
    private let scan: GPUScan
    private var tmpKeys: MTLBuffer?
    private var tmpVals: MTLBuffer?
    private var blockHist: MTLBuffer?
    private var capacity = 0

    init(ctx: MetalContext = .shared) { self.ctx = ctx; self.scan = GPUScan(ctx: ctx) }

    private func ensure(_ n: Int) {
        guard n > capacity else { return }
        let cap = Int(Double(n) * 1.25) + 1024
        tmpKeys = ctx.buffer(cap * 4, label: "sortTmpKeys")
        tmpVals = ctx.buffer(cap * 4, label: "sortTmpVals")
        let blocks = (cap + 1023) / 1024
        blockHist = ctx.buffer(blocks * 256 * 4, label: "sortHist")
        capacity = cap
    }

    /// Sorts `keys`/`vals` in place by the low `bits` bits of the key (rounded up to an even
    /// number of 8-bit passes so the result lands back in the input buffers).
    func encode(_ enc: MTLComputeCommandEncoder, keys: MTLBuffer, vals: MTLBuffer, count: Int, bits: Int = 32) {
        guard count > 1 else { return }
        ensure(count)
        var passes = (bits + 7) / 8
        if passes % 2 == 1 { passes += 1 }
        passes = min(passes, 4)
        let blocks = (count + 1023) / 1024
        var srcK = keys, srcV = vals, dstK = tmpKeys!, dstV = tmpVals!
        let hist = ctx.pipeline("radix_hist"), scat = ctx.pipeline("radix_scatter")
        for p in 0..<passes {
            let shift = UInt32(p * 8)
            enc.setComputePipelineState(hist)
            enc.setBuffer(srcK, offset: 0, index: 0)
            enc.setBuffer(blockHist!, offset: 0, index: 1)
            enc.set(UInt32(count), 2); enc.set(shift, 3); enc.set(UInt32(blocks), 4)
            enc.dispatchThreadgroups(MTLSize(width: blocks, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)
            scan.encode(enc, input: blockHist!, output: blockHist!, count: blocks * 256)
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(scat)
            enc.setBuffer(srcK, offset: 0, index: 0); enc.setBuffer(srcV, offset: 0, index: 1)
            enc.setBuffer(dstK, offset: 0, index: 2); enc.setBuffer(dstV, offset: 0, index: 3)
            enc.setBuffer(blockHist!, offset: 0, index: 4)
            enc.set(UInt32(count), 5); enc.set(shift, 6); enc.set(UInt32(blocks), 7)
            enc.dispatchThreadgroups(MTLSize(width: blocks, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)
            swap(&srcK, &dstK); swap(&srcV, &dstV)
        }
    }
}
