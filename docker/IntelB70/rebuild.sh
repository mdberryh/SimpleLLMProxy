#!/bin/bash
set -euo pipefail

BUILD_DIR="/mnt/user/Scripts/Docker/IntelB70"
IMAGE_NAME="llama-sycl-b70"
CONTAINER_NAME="qwen38-sycl"
MODEL_DIR="/mnt/user/Models/qwen38-27b"

cd "$BUILD_DIR"

echo "=== Building $IMAGE_NAME (latest llama.cpp master) ==="
docker build --no-cache -t "$IMAGE_NAME" .

echo "=== Stopping old container ==="
docker stop "$CONTAINER_NAME" 2>/dev/null || true
docker rm "$CONTAINER_NAME" 2>/dev/null || true

echo "=== Starting new container ==="
docker run -d --name "$CONTAINER_NAME" \
  --device /dev/dri \
  -v /mnt/user/Models:/models:ro \
  -p 8087:8080 \
  --restart unless-stopped \
  "$IMAGE_NAME" \
  -m /models/qwen38-27b/Qwen3.8-27B-Q4_K_M.gguf \
  --mmproj /models/qwen38-27b/mmproj-Qwen3.8-27B-f16.gguf \
  --device SYCL0 \
  -c 112640 \
  -ngl 999 \
  -fa on \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --spec-type draft-mtp \
  --spec-draft-n-max 1 \
  --parallel 1 \
  --jinja \
  --reasoning-budget 8192 \
  --reasoning-preserve \
  --presence-penalty 0 \
  --host 0.0.0.0 \
  --port 8080

echo "=== Waiting for model to load ==="
sleep 30

echo "=== Health check ==="
if curl -s http://localhost:8087/v1/models | grep -q "qwen"; then
  echo "✅ Server is up and serving"
else
  echo "⚠️  Server not responding yet — check logs:"
  echo "    docker logs $CONTAINER_NAME"
fi

echo "=== Done ==="
echo "Model:    Qwen3.8-27B Q4_K_S + MTP n-max 1"
echo "Endpoint: http://localhost:8087/v1"
echo "Logs:     docker logs -f $CONTAINER_NAME"
