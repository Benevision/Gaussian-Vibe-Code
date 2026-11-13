"""Patches a gsplat checkout so it builds for AMD GPUs with ROCm PyTorch (run by build-gsplat-rocm-wheel.sh inside the
build container, in the source folder). Fails loudly when setup.py no longer looks as expected.

  upstream  nerfstudio gsplat converted by PyTorch's hipify: hipcc gets clang flags instead of nvcc's
            (--use_fast_math, -diag-suppress), glm is copied whole into gsplat/hip/ (hipify copies its headers
            but not the .inl files they include), and the CUDA cooperative-groups calls HIP lacks are replaced
            (HIP_SHIMS below): cg::reduce, cg::labeled_partition, cg::this_grid().thread_rank(). The shims
            assume 32-wide wavefronts, i.e. Radeon (RDNA) — the only targets this wheel is built for.
  amd       AMD's port: its setup.py always appends +<git hash> to the version and compiles for the one target
            rocminfo reports; use GSPLAT_LOCAL_TAG and PYTORCH_ROCM_ARCH instead
"""
import glob
import os
import re
import shutil
import sys

HIP_SHIMS = r"""
// ---- SplatWork ROCm build: cooperative-groups pieces HIP doesn't have (32-wide wavefronts, RDNA) ----
struct GsLabeledGroup {
    unsigned int mask;   // lanes (of this 32-lane wavefront) with the same label
    unsigned int base;   // first lane of the 32-lane group in the hardware wavefront
    unsigned int rank;
    __device__ unsigned int thread_rank() const { return rank; }
};
template <class W, class L>
__device__ inline GsLabeledGroup gs_labeled_partition(W &, L label) {
    const unsigned int lane = __lane_id() & 31u, base = __lane_id() & ~31u;
    const unsigned int active = (unsigned int)(__ballot(1) >> base);   // lanes that returned early don't count
    unsigned int mask = 0;
    for (int i = 0; i < 32; ++i) {
        L other = __shfl(label, (int)(base + i));
        if (((active >> i) & 1u) && other == label) mask |= 1u << i;
    }
    return GsLabeledGroup{mask, base, (unsigned int)__popc(mask & ((1u << lane) - 1u))};
}
template <class T> __device__ inline T gs_reduce_sum(GsLabeledGroup &g, T v) {
    T t = T(0);
    for (int i = 0; i < 32; ++i) { T x = __shfl(v, (int)(g.base + i)); if ((g.mask >> i) & 1u) t += x; }
    return t;
}
template <class T> __device__ inline T gs_reduce_max(GsLabeledGroup &g, T v) {
    T t = v;
    for (int i = 0; i < 32; ++i) { T x = __shfl(v, (int)(g.base + i)); if (((g.mask >> i) & 1u) && x > t) t = x; }
    return t;
}
template <class G, class T> __device__ inline T gs_reduce_sum(G &, T v) {          // a full 32-lane tile
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    return v;
}
template <class G, class T> __device__ inline T gs_reduce_max(G &, T v) {
    for (int o = 16; o > 0; o >>= 1) { T x = __shfl_xor(v, o, 32); v = x > v ? x : v; }
    return v;
}
// ---- end of SplatWork shims ----
"""

flavor = sys.argv[1]
s = open("setup.py").read()
if flavor == "upstream":
    anchor = 'extra_compile_args["nvcc"] = nvcc_flags'
    assert anchor in s, "setup.py changed upstream (nvcc flags)"
    s = s.replace(anchor, anchor + '\n    extra_compile_args["nvcc"] = ["-O3", "-std=c++17", "-ffp-contract=fast"]'
                  '  # ROCm: clang flags (SplatWork build)', 1)
    # cooperative-groups calls HIP lacks
    utils = os.path.join("gsplat", "cuda", "include", "Utils.cuh")
    u = open(utils).read()
    assert "#include <cooperative_groups/reduce.h>" in u and "namespace cg = cooperative_groups;" in u, \
        "Utils.cuh changed upstream"
    u = u.replace("#include <cooperative_groups/reduce.h>\n", "")
    u = u.replace("namespace cg = cooperative_groups;", "namespace cg = cooperative_groups;\n" + HIP_SHIMS, 1)
    open(utils, "w").write(u)
    counts = {"reduce": 0, "labeled": 0, "grid": 0}
    for f in glob.glob(os.path.join("gsplat", "cuda", "csrc", "*.cu")) + [utils]:
        t = open(f).read()
        t, n1 = re.subn(r"cg::reduce\(\s*(\w+)\s*,\s*([^,]+?)\s*,\s*cg::plus<[^>]+>\(\)\s*\)", r"gs_reduce_sum(\1, \2)", t)
        t, n2 = re.subn(r"cg::reduce\(\s*(\w+)\s*,\s*([^,]+?)\s*,\s*cg::greater<[^>]+>\(\)\s*\)", r"gs_reduce_max(\1, \2)", t)
        t, n3 = re.subn(r"cg::labeled_partition\(", "gs_labeled_partition(", t)
        t, n4 = re.subn(r"cg::this_grid\(\)\.thread_rank\(\)", "(blockIdx.x * blockDim.x + threadIdx.x)", t)
        counts["reduce"] += n1 + n2
        counts["labeled"] += n3
        counts["grid"] += n4
        open(f, "w").write(t)
    print("replaced", counts)
    assert counts["reduce"] >= 10 and counts["labeled"] >= 8, counts
    left = [f for f in glob.glob(os.path.join("gsplat", "cuda", "**", "*.cu*"), recursive=True)
            if "third_party" not in f and re.search(r"cg::reduce|cg::labeled_partition|cg::this_grid", open(f).read())]
    assert not left, f"unreplaced cooperative-groups calls in {left}"
    dst = os.path.join("gsplat", "hip", "csrc", "third_party", "glm")
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(os.path.join("gsplat", "cuda", "csrc", "third_party", "glm"), dst)
elif flavor == "amd":
    pat = r'__version__ \+= f"\+\{git_rev\}"'
    assert len(re.findall(pat, s)) == 1, "setup.py changed upstream (version)"
    s = re.sub(pat, '__version__ += "+" + os.environ["GSPLAT_LOCAL_TAG"]', s)
    arch = 'f"--offload-arch={gpu_arch}"'
    assert arch in s, "setup.py changed upstream (offload-arch)"
    s = s.replace(arch, '*[f"--offload-arch={a}" for a in os.environ.get("PYTORCH_ROCM_ARCH", gpu_arch).split(";") if a]')
    if not re.search(r"^import os", s, re.M):
        s = "import os\n" + s
else:
    raise SystemExit(f"unknown flavor {flavor}")
open("setup.py", "w").write(s)
print(f"patched setup.py for {flavor}")
