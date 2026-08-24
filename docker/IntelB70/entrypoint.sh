#!/bin/bash

source /opt/intel/oneapi/setvars.sh > /dev/null

# SIGSEGV bug in oneAPI 2026.0 on Xe2 (Battlemage) without this
export SYCL_CACHE_PERSISTENT=0

# Target B70 via Level Zero (not OpenCL fallback)
export ONEAPI_DEVICE_SELECTOR=level_zero:0

exec /opt/llama.cpp/build/bin/llama-server "$@"
