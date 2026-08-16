# ROCm-based llama.cpp server — native HIP backend (no Vulkan/RADV)
# Build: docker build -f Dockerfile.llama-rocm -t llama-rocm .

# ---- Build stage ----
FROM rocm/dev-ubuntu-24.04:7.2.4-complete AS builder

ENV HIP_PLATFORM=amd
ENV HIP_PATH=/opt/rocm
ENV PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/llvm/lib
ENV CMAKE_PREFIX_PATH=/opt/rocm
ENV AMDGPU_TARGETS="gfx1201"

RUN apt-get update -o Acquire::AllowInsecureRepositories=true && \
    apt-get install -y --no-install-recommends --allow-unauthenticated \
    git cmake g++ make ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git .

RUN cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_HIP=ON \
    -DGGML_AVX2=ON \
    -DGGML_FMA=ON \
    -DGGML_F16C=ON \
    -DGGML_FLASH_ATTN=ON \
    -DGGML_K_QUANTS=ON \
    -DLLAMA_CURL=ON \
    -DLLAVA_SERVER=ON \
    && cmake --build build -j$(nproc)

# ---- Runtime stage ----
FROM rocm/dev-ubuntu-24.04:7.2.4-complete

ENV HIP_PLATFORM=amd
ENV HIP_PATH=/opt/rocm
ENV PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/llvm/lib
ENV AMDGPU_TARGETS="gfx1201"

RUN apt-get update -o Acquire::AllowInsecureRepositories=true && \
    apt-get install -y --no-install-recommends --allow-unauthenticated \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/build/bin/llama-server /usr/local/bin/llama-server

WORKDIR /app
EXPOSE 8080

ENTRYPOINT ["llama-server"]
CMD ["--help"]
