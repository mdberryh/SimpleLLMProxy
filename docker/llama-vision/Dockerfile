# Optimized Vulkan llama.cpp server for RDNA4 (gfx1201)
# Based on community findings from https://github.com/ggml-org/llama.cpp/discussions/21043
#
# REQUIRES HOST FIXES FIRST (see NK-01 GPU Upgrade Plan):
#   1. Mesa >= 25.2.8 (ideally 26.0+)
#   2. Kernel >= 6.19.8
#   3. PCIe ASPM performance mode
#
# Build: docker build -f Dockerfile.llama-vulkan -t llama-vulkan .
# Run: docker run --device=/dev/dri/card0 --device=/dev/dri/renderD128 \
#        -p 8080:8080 -v /mnt/data/models:/models \
#        -e VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
#        -e GGML_VK_VISIBLE_DEVICES=0 \
#        llama-vulkan --model /models/Qwen3.6-27B-UD-Q4_K_XL.gguf ...

# ---- Build stage ----
FROM ubuntu:24.04 AS builder

# Install build deps + Vulkan SDK (minimal — just loader + headers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake g++ make ninja-build \
    libvulkan-dev vulkan-tools \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git .

# Vulkan build flags for RDNA4:
# GGML_VULKAN=ON           : Vulkan backend
# GGML_VULKAN_CHECK_RESULTS : disabled for perf (use only for debugging)
# GGML_VULKAN_DEBUG        : disabled for production
# GGML_FLASH_ATTN=ON       : Flash attention (critical for RDNA4 prefill)
# GGML_K_QUANTS=ON         : K-quants support (Q4_K_XL, Q6_K, etc.)
RUN cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_VULKAN=ON \
    -DGGML_FLASH_ATTN=ON \
    -DGGML_K_QUANTS=ON \
    -DGGML_AVX2=ON \
    -DGGML_FMA=ON \
    -DGGML_F16C=ON \
    -DLLAMA_CURL=ON \
    -DLLAVA_SERVER=ON \
    -DBUILD_SHARED_LIBS=OFF \
    && cmake --build build -j$(nproc)

# ---- Runtime stage ----
FROM ubuntu:24.04

# Install minimal runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libvulkan1 mesa-vulkan-drivers \
    vulkan-tools ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy built binary
COPY --from=builder /src/build/bin/llama-server /usr/local/bin/llama-server

# RDNA4 Vulkan environment:
# VK_ICD_FILENAMES: Force RADV explicitly (avoids loading 7 ICDs)
# GGML_VK_VISIBLE_DEVICES: Pin to specific GPU (0 = card0)
# RADV_PERFTEST=transfer_queue: Use dedicated SDma for buffer transfers
#   (experimental but helps llama.cpp buffer-heavy workloads)
ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json
ENV GGML_VK_VISIBLE_DEVICES=0
ENV RADV_PERFTEST=transfer_queue

# NOTE: DO NOT set GGML_VK_ALLOW_GRAPHICS_QUEUE=1 for dense models
# It hurts dense decode by ~8% on RDNA4 (only helps MoE +5%)

WORKDIR /app
EXPOSE 8080

ENTRYPOINT ["llama-server"]
CMD ["--help"]

