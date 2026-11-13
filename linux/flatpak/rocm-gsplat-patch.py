"""Patches a gsplat checkout so it builds for AMD GPUs with ROCm PyTorch (run by build-gsplat-rocm-wheel.sh inside the
build container, in the source folder). Fails loudly when setup.py no longer looks as expected.

  upstream  nerfstudio gsplat converted by PyTorch's hipify: hipcc gets clang flags instead of nvcc's
            (--use_fast_math, -diag-suppress), and glm is copied whole into gsplat/hip/ (hipify copies its headers
            but not the .inl files they include)
  amd       AMD's port: its setup.py always appends +<git hash> to the version and compiles for the one target
            rocminfo reports; use GSPLAT_LOCAL_TAG and PYTORCH_ROCM_ARCH instead
"""
import os
import re
import shutil
import sys

flavor = sys.argv[1]
s = open("setup.py").read()
if flavor == "upstream":
    anchor = 'extra_compile_args["nvcc"] = nvcc_flags'
    assert anchor in s, "setup.py changed upstream (nvcc flags)"
    s = s.replace(anchor, anchor + '\n    extra_compile_args["nvcc"] = ["-O3", "-std=c++17", "-ffp-contract=fast"]'
                  '  # ROCm: clang flags (SplatWork build)', 1)
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
