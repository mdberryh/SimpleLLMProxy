"""Proxy relay HTTP server — main entry point."""

# Change Log
# ==============================================
# 2026-06-15 - AI - /health now returns actual uptime duration instead of epoch time
# 2026-06-15 - AI - Tagged temporary auto-wake call sites with [AUTOWAKE-TEMP] for clean future removal
# 2026-07-06 - AI - Added bypassNonStreamingChatCompletions option: stream=False chat requests get a synthetic response, never touch GPUs
# 2026-07-06 - AI - Strip all hop-by-hop headers, force upstream Connection: close, readline()-based SSE relay with [DONE] break
# 2026-07-06 - AI - Removed manual socket shutdowns, reduced JSON upstream timeout to 120s, added per-request ID debug logs
# 2026-07-06 - AI - Track _response_started to prevent writing a second set of headers after an upstream failure mid-response
# 2026-07-06 - AI - _forward_raw now sends an explicit Content-Length and flushes the client socket
# ==============================================

import json
import logging
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlparse

from .compression import CircuitBreaker, compress_via_llmlingua
from .router import (
    is_multimodal,
    reassemble_messages,
    rewrite_model,
    should_compress,
    split_messages_for_compression,
)
from ..gpu_managers import GpuWakelockManager  # [AUTOWAKE-TEMP] see wakelock.py removal guide
from ..shared import load_config

logger = logging.getLogger("proxy.server")

START_TIME = time.time()

# RFC 2616 hop-by-hop headers — must never be forwarded in either direction.
HOP_BY_HOP_HEADERS = frozenset({
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "proxy-connection",
    "te",
    "trailer",
    "trailers",
    "transfer-encoding",
    "upgrade",
})


# ---------------------------------------------------------------------------
# Metrics counters (lightweight in-process state — no disk persistence)
# ---------------------------------------------------------------------------

class Metrics:
    def __init__(self):
        self.tokens_saved_total = 0
        self.compression_calls = 0
        self.compression_input_tokens_total = 0
        self.compression_output_tokens_total = 0
        self.gpu_wake_events_total = 0
        self._lock = __import__("threading").Lock()

    def record_compression(self, original: int, compressed: int) -> None:
        with self._lock:
            self.compression_calls += 1
            self.compression_input_tokens_total += original
            self.compression_output_tokens_total += compressed
            saved = max(0, original - compressed)
            self.tokens_saved_total += saved

    def record_gpu_wake(self) -> None:
        with self._lock:
            self.gpu_wake_events_total += 1

    def render_prometheus(self, queue_depth: int, circuit_open: int, gpu_statuses: dict = None) -> str:
        with self._lock:
            avg_ratio = (
                self.compression_output_tokens_total
                / self.compression_input_tokens_total
                if self.compression_input_tokens_total
                else 0.0
            )
            lines = [
                "# HELP llmlingua_tokens_saved_total Total tokens saved via compression",
                "# TYPE llmlingua_tokens_saved_total counter",
                f"llmlingua_tokens_saved_total {self.tokens_saved_total}",
                "# HELP llmlingua_compression_ratio Average compression ratio achieved",
                "# TYPE llmlingua_compression_ratio gauge",
                f"llmlingua_compression_ratio {avg_ratio:.4f}",
                "# HELP gpu_wake_events_total Total GPU container wake events",
                "# TYPE gpu_wake_events_total counter",
                f"gpu_wake_events_total {self.gpu_wake_events_total}",
                "# HELP proxy_queue_depth Current request queue depth",
                "# TYPE proxy_queue_depth gauge",
                f"proxy_queue_depth {queue_depth}",
                "# HELP llmlingua_circuit_breaker_state 1 if open, 0 if closed",
                "# TYPE llmlingua_circuit_breaker_state gauge",
                f"llmlingua_circuit_breaker_state {circuit_open}",
            ]
            
            if gpu_statuses:
                lines.extend([
                    "# HELP gpu_heartbeat_status Last heartbeat status for GPU (1 = healthy, 0 = unhealthy)",
                    "# TYPE gpu_heartbeat_status gauge",
                ])
                for gpu_name, gpu_info in gpu_statuses.items():
                    is_healthy = 1 if gpu_info.get("healthy", False) else 0
                    lines.append(f'gpu_heartbeat_status{{gpu="{gpu_name}"}} {is_healthy}')
                
                lines.extend([
                    "# HELP gpu_state Current state of the GPU (1 = ready, 2 = busy, 3 = starting, 4 = stopped, 0 = unknown)",
                    "# TYPE gpu_state gauge",
                ])
                state_map = {"ready": 1, "busy": 2, "starting": 3, "stopped": 4, "unknown": 0}
                for gpu_name, gpu_info in gpu_statuses.items():
                    state_val = state_map.get(gpu_info.get("state", "unknown"), 0)
                    lines.append(f'gpu_state{{gpu="{gpu_name}"}} {state_val}')
                    
        return "\n".join(lines) + "\n"


METRICS = Metrics()

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

CONFIG = load_config()
CIRCUIT = CircuitBreaker(
    max_failures=CONFIG.get("circuitBreakerFailures", 2),
    cooldown_minutes=CONFIG.get("circuitBreakerCooldownMinutes", 10),
)
# [AUTOWAKE-TEMP] Temporary GPU power manager. Post-removal, replace with a tiny
# passthrough router (see REMOVAL GUIDE at top of src/gpu_managers/wakelock.py).
WAKELOCK = GpuWakelockManager(CONFIG)


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    _response_started = False  # guards against double response headers

    def log_message(self, format, *args):
        logger.info("Proxy: " + (format % args))

    # ------------------------------------------------------------------
    # GET endpoints
    # ------------------------------------------------------------------

    def do_GET(self):
        client_ip = self.client_address[0]
        if self.path != "/health":
            logger.info("GET %s from %s", self.path, client_ip)
        if self.path == "/health":
            self._json(200, {
                "status": "healthy",
                "uptime_seconds": round(time.time() - START_TIME, 1),
            })
        elif self.path == "/api/status":
            self._json(200, WAKELOCK.status)  # [AUTOWAKE-TEMP] wake/queue fields
        elif self.path == "/metrics":
            status = WAKELOCK.status  # [AUTOWAKE-TEMP] queue depth only
            body = METRICS.render_prometheus(
                queue_depth=status.get("queue_length", 0),
                circuit_open=CIRCUIT.get_state_for_metrics(),
                gpu_statuses=status.get("gpus", {}),
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.end_headers()
            self.wfile.write(body.encode())
        elif self.path.startswith("/v1/models") or self.path.startswith("/models"):
            # Return a static list of supported models based on config.
            # We do this statically to avoid waking up GPUs just for health checks.
            models = []
            model_map = CONFIG.get("modelMap", {})
            for model_id in model_map.keys():
                models.append({
                    "id": model_id,
                    "object": "model",
                    "created": int(START_TIME),
                    "owned_by": "local"
                })
            
            default_model = CONFIG.get("_defaultFallback", "")
            if default_model and default_model not in model_map:
                models.append({
                    "id": default_model,
                    "object": "model",
                    "created": int(START_TIME),
                    "owned_by": "local"
                })

            self._json(200, {
                "object": "list",
                "data": models
            })
        else:
            self._json(404, {"error": "not found"})

    # ------------------------------------------------------------------
    # POST endpoints
    # ------------------------------------------------------------------

    def do_POST(self):
        req_id = uuid.uuid4().hex[:8]
        self._response_started = False
        client_ip = self.client_address[0]
        user_agent = self.headers.get("User-Agent", "")
        auth_header = self.headers.get("Authorization", "")
        logger.info("[%s] Request from %s | UA: %s | Auth: %s", req_id, client_ip, user_agent, auth_header[:20] + "..." if len(auth_header) > 20 else auth_header)

        if "/v1/chat/completions" not in self.path and "/chat/completions" not in self.path:
            self._forward_raw()
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid JSON"})
            return

        stream = data.get("stream", False)
        messages = data.get("messages", [])
        model_name = data.get("model", "")
        logger.info("[%s] Request model='%s' stream=%s", req_id, model_name, stream)

        if not messages:
            self._json(400, {"error": "missing messages array"})
            return

        # Non-stream bypass: background stream=False completions (UI title
        # generation etc.) must never occupy a GPU slot — real stream=True
        # requests would otherwise starve at the wakelock queue. Return a
        # minimal OpenAI-compatible synthetic response without routing.
        if not stream and CONFIG.get("bypassNonStreamingChatCompletions", False):
            logger.info(
                "[%s] Non-streaming chat completion bypassed (no GPU routing) for model='%s'",
                req_id, model_name,
            )
            self._json(200, {
                "id": f"chatcmpl-bypass-{req_id}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            })
            return

        # 1. Multi-modal bypass
        multimodal = is_multimodal(messages)
        if multimodal:
            logger.info(
                "Multi-modal request for '%s'. Bypassing LLMLingua compression.",
                model_name,
            )
        # 2. Compression step
        elif should_compress(model_name, CONFIG):
            # Circuit breaker gate
            if not CIRCUIT.can_execute():
                logger.info("[Proxy] Circuit breaker OPEN. Bypassing compression.")
            else:
                # Smart compression with recency bias
                system_msgs, middle_msgs, recent_msgs = split_messages_for_compression(
                    messages, keep_recent_count=4
                )
                if middle_msgs:
                    compressed, orig_tok, comp_tok, errored = compress_via_llmlingua(
                        middle_msgs, CONFIG
                    )
                    if errored:
                        # Real compression error — LLMLingua failed
                        CIRCUIT.record_failure()
                    elif orig_tok > 0:
                        # Reassemble + record metrics
                        data["messages"] = reassemble_messages(
                            system_msgs, compressed, recent_msgs
                        )
                        METRICS.record_compression(orig_tok, comp_tok)
                        CIRCUIT.record_success()
                        logger.info(
                            "Compression active for '%s': %d -> %d tokens",
                            model_name, orig_tok, comp_tok,
                        )
                    else:
                        # LLMLingua succeeded with zero tokens (empty input)
                        # — not a failure, just nothing to compress
                        logger.info(
                            "Compression skipped for '%s': no tokens to compress",
                            model_name,
                        )

        # 3. Model rewrite
        rewrite_model(data, CONFIG)

        # 4. Route
        WAKELOCK._record_request()  # [AUTOWAKE-TEMP] idle/wake bookkeeping
        try:
            # [AUTOWAKE-TEMP] post-removal: replace with passthrough lookup of the
            # highest-priority localModel URL + self.path (used_external -> False).
            upstream_url, used_external, gpu_name = WAKELOCK.route(self.path, multimodal=multimodal)
        except RuntimeError as e:
            if str(e) == "QUEUE_OVERFLOW":
                self._json(503, {
                    "error": "service unavailable",
                    "message": "GPU queue overflow — too many pending requests",
                })
                return
            raise

        if not upstream_url:
            self._json(503, {"error": "no backend available"})
            return

        # 5. Forward request
        auth = self.headers.get("Authorization", "")
        if used_external:  # [AUTOWAKE-TEMP] cloud fallback only happens during warmup
            ext_auth = WAKELOCK.external_auth()
            if ext_auth:
                auth = ext_auth

        headers = {
            "Content-Type": "application/json",
            "Authorization": auth,
            "Connection": "close",  # force upstream to close, never keep-alive
        }
        forward_body = json.dumps(data).encode()

        try:
            if stream:
                self._forward_stream(upstream_url, forward_body, headers, gpu_name, req_id)
            else:
                self._forward_json(upstream_url, forward_body, headers, gpu_name, req_id)
        except Exception:
            logger.exception("[%s] Forwarding error", req_id)
            if not self._response_started:
                self._json(502, {"error": "upstream forwarding failed"})

    # ------------------------------------------------------------------
    # Forwarding helpers
    # ------------------------------------------------------------------

    def _forward_json(self, url, body, headers, gpu_name, req_id=""):
        self.close_connection = True
        req = urllib_request.Request(url, data=body, headers=headers, method="POST")
        try:
            # 120s upstream timeout: JSON completions should never hold a GPU
            # slot for the previous 300s worst case.
            with urllib_request.urlopen(req, timeout=120) as resp:
                payload = resp.read()
                logger.debug("[%s] _forward_json: received %d bytes from upstream", req_id, len(payload))
                self.send_response(resp.status)
                self._response_started = True
                resp_headers = []
                for k, v in resp.headers.items():
                    if k.lower() in HOP_BY_HOP_HEADERS or k.lower() in ("content-encoding", "content-length"):
                        continue
                    resp_headers.append((k, v))
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Connection", "close")
                resp_headers.append(("Content-Length", str(len(payload))))
                resp_headers.append(("Connection", "close"))
                logger.debug("[%s] _forward_json: sending headers: %s", req_id, resp_headers)
                self.end_headers()
                self.wfile.write(payload)
                self.wfile.flush()
                logger.debug("[%s] _forward_json: wrote %d bytes to client, closing connection", req_id, len(payload))
        except urllib_error.HTTPError as e:
            logger.error("[%s] _forward_json: HTTPError %d: %s", req_id, e.code, e.reason)
            self._json(e.code, e.read().decode())
        except Exception as e:
            logger.error("[%s] _forward_json: exception: %s", req_id, e, exc_info=True)
            raise
        finally:
            WAKELOCK.release(gpu_name)

    def _forward_stream(self, url, body, headers, gpu_name, req_id=""):
        import http.client
        from urllib.parse import urlparse

        self.close_connection = True
        line_count = 0
        logger.debug("[%s] _forward_stream: connecting to %s", req_id, url)

        parsed = urlparse(url)
        conn = None
        try:
            conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=120)
            conn.request("POST", parsed.path + (f"?{parsed.query}" if parsed.query else ""), body=body, headers=headers)
            resp = conn.getresponse()

            logger.debug("[%s] _forward_stream: connected, status=%d", req_id, resp.status)
            self.send_response(resp.status)
            self._response_started = True

            has_content_type = False
            resp_headers = []
            for k, v in resp.getheaders():
                if k.lower() in HOP_BY_HOP_HEADERS or k.lower() in ("content-encoding", "content-length"):
                    continue
                if k.lower() == "content-type":
                    has_content_type = True
                resp_headers.append((k, v))
                self.send_header(k, v)
            if not has_content_type:
                self.send_header("Content-Type", "text/event-stream")
                resp_headers.append(("Content-Type", "text/event-stream"))
            self.send_header("Connection", "close")
            resp_headers.append(("Connection", "close"))
            logger.debug("[%s] _forward_stream: sending headers: %s", req_id, resp_headers)
            self.end_headers()

            # Relay SSE line-by-line so we can stop exactly at [DONE] instead
            # of blocking in a fixed-size read waiting for upstream to close.
            sent_done = False
            while True:
                try:
                    line = resp.readline()
                    if not line:
                        break
                    self.wfile.write(line)
                    self.wfile.flush()
                    line_count += 1
                    if b"[DONE]" in line:
                        sent_done = True
                        try:
                            self.wfile.write(b"\n")
                            self.wfile.flush()
                        except BrokenPipeError:
                            pass
                        break
                except BrokenPipeError:
                    logger.warning("[%s] _forward_stream: client disconnected after %d lines", req_id, line_count)
                    break
                except TimeoutError:
                    logger.error("[%s] _forward_stream: upstream read timeout after %d lines", req_id, line_count)
                    break

            if sent_done:
                logger.info("[%s] _forward_stream: completed %d lines", req_id, line_count)
            elif line_count > 0:
                logger.warning("[%s] _forward_stream: incomplete - %d lines, no [DONE]", req_id, line_count)
                try:
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except BrokenPipeError:
                    pass
            else:
                logger.warning("[%s] _forward_stream: no data received from upstream", req_id)

        except http.client.HTTPException as e:
            logger.error("[%s] _forward_stream: HTTPException after %d lines: %s", req_id, line_count, e)
            if line_count == 0 and not self._response_started:
                self._json(502, {"error": "upstream error", "message": str(e)})
        except TimeoutError:
            logger.error("[%s] _forward_stream: socket timeout after %d lines", req_id, line_count)
        except Exception as e:
            logger.error("[%s] _forward_stream: exception after %d lines: %s", req_id, line_count, e, exc_info=True)
            if line_count == 0:
                raise
        finally:
            if conn:
                conn.close()
            WAKELOCK.release(gpu_name)

    def _forward_raw(self):
        """Forward non-chat endpoints without modification."""
        self.close_connection = True
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        url, _, gpu_name = WAKELOCK.route(self.path)  # [AUTOWAKE-TEMP] -> passthrough lookup
        if not url:
            self._json(503, {"error": "no backend available"})
            return

        parsed_url = urlparse(url)
        if not parsed_url.path or parsed_url.path.rstrip("/") == "":
            url = url.rstrip("/") + self.path

        hdrs = {
            k: v for k, v in self.headers.items()
            if k.lower() not in HOP_BY_HOP_HEADERS
        }
        hdrs.pop("Host", None)
        hdrs.pop("Content-Length", None)
        hdrs["Content-Length"] = str(len(body))
        hdrs["Connection"] = "close"  # force upstream to close, never keep-alive

        req = urllib_request.Request(url, data=body, headers=hdrs, method="POST")
        try:
            with urllib_request.urlopen(req, timeout=120) as resp:
                payload = resp.read()
                logger.debug("_forward_raw: received %d bytes from upstream", len(payload))
                self.send_response(resp.status)
                self._response_started = True
                resp_headers = []
                for k, v in resp.headers.items():
                    if k.lower() in HOP_BY_HOP_HEADERS or k.lower() in ("content-encoding", "content-length"):
                        continue
                    resp_headers.append((k, v))
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Connection", "close")
                resp_headers.append(("Content-Length", str(len(payload))))
                resp_headers.append(("Connection", "close"))
                logger.debug("_forward_raw: sending headers: %s", resp_headers)
                self.end_headers()
                self.wfile.write(payload)
                self.wfile.flush()
                logger.debug("_forward_raw: wrote %d bytes to client", len(payload))
        except urllib_error.HTTPError as e:
            logger.error("_forward_raw: HTTPError %d: %s", e.code, e.reason)
            self._json(e.code, e.read().decode())
        except Exception as e:
            logger.error("_forward_raw: exception: %s", e, exc_info=True)
            raise
        finally:
            WAKELOCK.release(gpu_name)

    # ------------------------------------------------------------------
    # JSON helper
    # ------------------------------------------------------------------

    def _json(self, code, body):
        if isinstance(body, dict):
            body = json.dumps(body)
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self._response_started = True
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()
        self.close_connection = True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    port = CONFIG.get("proxyPort", 8087)
    logger.info("Starting proxy on port %d", port)
    logger.info(
        "Config: %d GPUs, llmlingua=%s, wakelock=%s",
        len(CONFIG.get("localModels", [])),
        CONFIG.get("llmlinguaUrl", "N/A"),
        "enabled" if WAKELOCK.enabled else "disabled",
    )

    server = ThreadingHTTPServer(("0.0.0.0", port), ProxyHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
