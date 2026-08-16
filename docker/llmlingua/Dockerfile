# ==============================================
# LLMLingua-2 Compression Server — ROCm/HIP backend (gfx1201 / RDNA4)
# Shares the same GPU as llama-agents via the ROCm 7.2.4 stack.

FROM rocm/dev-ubuntu-24.04:7.2.4-complete AS base

ENV HIP_PLATFORM=amd
ENV HIP_PATH=/opt/rocm
ENV PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/llvm/lib:$LD_LIBRARY_PATH
ENV AMDGPU_TARGETS="gfx1201"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev curl git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install torch with ROCm support — must come before requirements.txt
# so pip does not override this with the CPU build from PyPI.
# Using rocm6.2 wheel which is compatible with ROCm 7.2.4 host/base.
RUN pip3 install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2

# Install remaining deps (torch already installed above)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
