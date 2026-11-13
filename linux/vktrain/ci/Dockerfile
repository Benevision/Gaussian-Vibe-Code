# Build and test image for the Vulkan trainer: Debian trixie with Mesa lavapipe (software Vulkan), so the
# kernels and their numerical checks run without a GPU. Works on x86_64 (CI) and aarch64 (colima on the Mac).
#   docker build -t splat-vk-test -f linux/vktrain/ci/Dockerfile linux/vktrain/ci
#   docker run --rm -v "$PWD":/w -w /w/linux/vktrain splat-vk-test \
#     sh -c 'cmake -S . -B build -G Ninja && cmake --build build && ./build/vktrain-selftest'
FROM debian:trixie
RUN apt-get update -q && DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
    g++ cmake ninja-build make pkg-config libvulkan-dev vulkan-tools glslang-tools spirv-tools \
    mesa-vulkan-drivers python3 python3-numpy python3-pytest python3-pil ca-certificates git && rm -rf /var/lib/apt/lists/*
# Select lavapipe. Its ICD manifest is lvp_icd.json on trixie, lvp_icd.<arch>.json on some other releases:
# link whichever exists to a fixed name so the image is the same on every architecture.
RUN set -e; f=$(ls /usr/share/vulkan/icd.d/lvp_icd*.json | head -n 1); test -n "$f"; \
    ln -sf "$f" /usr/share/vulkan/icd.d/selected_lavapipe.json
ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/selected_lavapipe.json
