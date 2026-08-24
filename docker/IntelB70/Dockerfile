FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Base build dependencies
RUN apt-get update && apt-get install -y \
  git cmake build-essential wget curl \
  python3-pip software-properties-common gnupg && \
  rm -rf /var/lib/apt/lists/*

# Add Intel graphics PPA for Battlemage-compatible Level Zero driver
# Ubuntu 24.04's default libze-intel-gpu1 (23.43) is too old for B70
RUN add-apt-repository -y ppa:kobuk-team/intel-graphics && \
  apt-get update && \
  apt-get install -y \
    libze-dev libze1 libze-intel-gpu1 \
    intel-metrics-discovery intel-opencl-icd \
    clinfo && \
  rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y xpu-manager


# Intel Deep Learning Essentials (icx/icpx, oneMKL, oneCCL, DPC++ runtime)
COPY intel-deep-learning-essentials-*.sh /tmp/
RUN sh /tmp/intel-deep-learning-essentials-*.sh -a -s --eula accept \
  --install-dir /opt/intel/oneapi && \
  rm /tmp/intel-deep-learning-essentials-*.sh

# oneDNN (separate as of 2026.0 — required for XMX hardware units)
COPY intel-onednn-*.sh /tmp/
RUN sh /tmp/intel-onednn-*.sh -a -s --eula accept \
  --install-dir /opt/intel/oneapi && \
  rm /tmp/intel-onednn-*.sh

# Clone and build llama.cpp with SYCL for Battlemage
RUN git clone https://github.com/ggml-org/llama.cpp /opt/llama.cpp

WORKDIR /opt/llama.cpp

RUN bash -c '\
  source /opt/intel/oneapi/setvars.sh && \
  cmake --fresh -S . -B build \
    -DCMAKE_C_COMPILER=icx \
    -DCMAKE_CXX_COMPILER=icpx \
    -DGGML_SYCL=ON \
    -DGGML_SYCL_TARGET=INTEL \
    -DGGML_SYCL_DNN=ON \
    -DGGML_SYCL_F16=ON \
    -DGGML_SYCL_GRAPH=ON \
    -DGGML_SYCL_HOST_MEM_FALLBACK=ON \
    -DCMAKE_BUILD_TYPE=Release && \
  cmake --build build --config Release --target llama-server llama-bench test-backend-ops -j$(nproc)'

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8080
ENTRYPOINT ["/entrypoint.sh"]
