extension ShaderSource {
    /// Exclusive prefix scan + stable LSD radix sort (8-bit digits, key/value uint32).
    /// Ranking inside a SIMD-group uses ballot-based multisplit, which makes the scatter
    /// stable without any serial loops — this is what lets us sort tile keys that were
    /// emitted in depth order and keep per-tile depth order intact.
    static let sort = #"""
#define SCAN_TG 256
#define SCAN_PER_THREAD 4
#define SCAN_BLOCK (SCAN_TG * SCAN_PER_THREAD)

kernel void scan_block(device const uint* input [[buffer(0)]],
                       device uint* output [[buffer(1)]],
                       device uint* blockSums [[buffer(2)]],
                       constant uint& n [[buffer(3)]],
                       uint tid [[thread_index_in_threadgroup]],
                       uint bid [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint sgTotals[SCAN_TG / 32];
    uint base = bid * SCAN_BLOCK + tid * SCAN_PER_THREAD;
    uint v[SCAN_PER_THREAD];
    uint sum = 0;
    for (uint i = 0; i < SCAN_PER_THREAD; i++) {
        uint idx = base + i;
        v[i] = idx < n ? input[idx] : 0;
        uint t = v[i]; v[i] = sum; sum += t;   // thread-local exclusive
    }
    uint pre = simd_prefix_exclusive_sum(sum);
    if (lane == 31) sgTotals[sg] = pre + sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        uint t = lane < (SCAN_TG / 32) ? sgTotals[lane] : 0;
        uint e = simd_prefix_exclusive_sum(t);
        if (lane < (SCAN_TG / 32)) sgTotals[lane] = e;
        if (lane == (SCAN_TG / 32) - 1) blockSums[bid] = e + t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint off = pre + sgTotals[sg];
    for (uint i = 0; i < SCAN_PER_THREAD; i++) {
        uint idx = base + i;
        if (idx < n) output[idx] = v[i] + off;
    }
}

kernel void scan_add(device uint* data [[buffer(0)]],
                     device const uint* blockOffsets [[buffer(1)]],
                     constant uint& n [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    data[gid] += blockOffsets[gid / SCAN_BLOCK];
}

#define RADIX_TG 256
#define RADIX_ROUNDS 4
#define RADIX_BLOCK (RADIX_TG * RADIX_ROUNDS)

kernel void radix_hist(device const uint* keys [[buffer(0)]],
                       device uint* blockHist [[buffer(1)]],
                       constant uint& n [[buffer(2)]],
                       constant uint& shift [[buffer(3)]],
                       constant uint& numBlocks [[buffer(4)]],
                       uint tid [[thread_index_in_threadgroup]],
                       uint bid [[threadgroup_position_in_grid]]) {
    threadgroup atomic_uint hist[256];
    atomic_store_explicit(&hist[tid], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base = bid * RADIX_BLOCK;
    for (uint r = 0; r < RADIX_ROUNDS; r++) {
        uint idx = base + r * RADIX_TG + tid;
        if (idx < n) atomic_fetch_add_explicit(&hist[(keys[idx] >> shift) & 0xFF], 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    blockHist[tid * numBlocks + bid] = atomic_load_explicit(&hist[tid], memory_order_relaxed);
}

kernel void radix_scatter(device const uint* keysIn [[buffer(0)]],
                          device const uint* valsIn [[buffer(1)]],
                          device uint* keysOut [[buffer(2)]],
                          device uint* valsOut [[buffer(3)]],
                          device const uint* blockOffsets [[buffer(4)]],
                          constant uint& n [[buffer(5)]],
                          constant uint& shift [[buffer(6)]],
                          constant uint& numBlocks [[buffer(7)]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint bid [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint sgCounts[RADIX_TG / 32][256];
    threadgroup uint running[256];
    running[tid] = blockOffsets[tid * numBlocks + bid];
    uint base = bid * RADIX_BLOCK;
    for (uint r = 0; r < RADIX_ROUNDS; r++) {
        for (uint s = 0; s < RADIX_TG / 32; s++) sgCounts[s][tid] = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint idx = base + r * RADIX_TG + tid;
        bool valid = idx < n;
        uint key = valid ? keysIn[idx] : 0;
        uint digit = (key >> shift) & 0xFF;
        ulong peers = (ulong)(simd_vote::vote_t)simd_ballot(valid);
        for (uint b = 0; b < 8; b++) {
            bool bit = (digit >> b) & 1;
            ulong vb = (ulong)(simd_vote::vote_t)simd_ballot(bit);
            peers &= bit ? vb : ~vb;
        }
        ulong ltMask = (1ul << lane) - 1ul;
        uint rank = popcount(peers & ltMask);
        uint cnt = popcount(peers);
        if (valid && rank == 0) sgCounts[sg][digit] = cnt;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Thread `tid` owns digit `tid`: exclusive prefix over SIMD-groups.
        uint total = 0;
        for (uint s = 0; s < RADIX_TG / 32; s++) { uint t = sgCounts[s][tid]; sgCounts[s][tid] = total; total += t; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (valid) {
            uint pos = running[digit] + sgCounts[sg][digit] + rank;
            keysOut[pos] = key;
            valsOut[pos] = valsIn[idx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        running[tid] += total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void fill_iota(device uint* vals [[buffer(0)]], constant uint& n [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid < n) vals[gid] = gid;
}

kernel void fill_uint(device uint* vals [[buffer(0)]], constant uint& n [[buffer(1)]], constant uint& value [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid < n) vals[gid] = value;
}
"""#
}
