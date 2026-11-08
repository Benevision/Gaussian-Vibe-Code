# SplatWork Node - CUDA training server
#
#   docker build -t splatworker .
#   docker run --gpus all --network host \
#     -v splatworker-data:/data -e SPLATWORKER_TOKEN=$(cat token) splatworker
#
# With the splatprep processing toolkit (named build context, BuildKit):
#   docker build --build-context tools=../tools --target worker-splatprep -t splatworker .
#
# --network host is needed for Bonjour/mDNS advertisement to reach the LAN.
# The devel image is used because gsplat / fused-ssim compile CUDA code.

ARG CUDA_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04
FROM ${CUDA_IMAGE} AS worker

ARG TORCH_CUDA=cu124
# Compile CUDA kernels for common GPUs (Turing .. Blackwell); override to speed up builds.
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    TORCH_EXTENSIONS_DIR=/data/torch_extensions \
    SPLATWORKER_DATA_DIR=/data

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev python3-pip git build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --upgrade pip wheel setuptools ninja \
 && pip install torch --index-url https://download.pytorch.org/whl/${TORCH_CUDA}

# gsplat: prebuilt wheel if available for this torch/CUDA, else build from source
# (ahead of time, so the first job does not wait for a JIT compile).
RUN TORCH_MM=$(python -c 'import torch;print("".join(torch.__version__.split("+")[0].split(".")[:2]))') \
 && (pip install "gsplat>=1.4,<2" --index-url https://docs.gsplat.studio/whl/pt${TORCH_MM}${TORCH_CUDA} \
        --extra-index-url https://pypi.org/simple \
     || pip install --no-build-isolation "git+https://github.com/nerfstudio-project/gsplat.git@v1.5.3") \
 && (pip install --no-build-isolation "git+https://github.com/rahul-goel/fused-ssim" \
     || echo "fused-ssim unavailable; using torch SSIM fallback")

WORKDIR /opt/splatworker
COPY pyproject.toml README.md ./
COPY splatworker ./splatworker
RUN pip install ".[train]"

RUN useradd --system --create-home --home-dir /home/splatworker splatworker \
 && mkdir -p /data && chown splatworker:splatworker /data
USER splatworker
VOLUME ["/data"]
EXPOSE 7878
ENTRYPOINT ["splatworker", "serve"]
CMD ["--port", "7878", "--data-dir", "/data"]

# ----------------------------------------------------------------------------
FROM worker AS worker-splatprep
USER root
COPY --from=tools splatprep /opt/tools/splatprep
RUN if [ -f /opt/tools/splatprep/pyproject.toml ] || [ -f /opt/tools/splatprep/setup.py ]; then \
        pip install /opt/tools/splatprep; \
    else \
        echo /opt/tools > $(python -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')/splatprep-local.pth; \
    fi
USER splatworker

# Default target = plain worker (does not need the 'tools' build context).
FROM worker AS final
