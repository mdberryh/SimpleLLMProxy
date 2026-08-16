docker build -f docker/proxy/Dockerfile -t llm-proxy:latest .

docker stop llm-proxy && docker rm llm-proxy

docker run -d \
  --name llm-proxy \
  -p 8082:8081 \
  -e APP_CONFIG_PATH=/app/app.yml \
  -v $(pwd)/docker/proxy/app.yml:/app/app.yml \
  -v /var/run/docker.sock:/var/run/docker.sock \
  llm-proxy:latest
