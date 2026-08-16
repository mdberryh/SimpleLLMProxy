"""LLMLingua compression client — calls external container via OpenAI API."""

# Change Log
# ==============================================
# 2026-06-15 - AI - Retry compression once after 5s on unreachable/503 before fallback
# ==============================================

import json
import logging
import threading
import time

import requests

logger = logging.getLogger("proxy.compression")


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """Opens after consecutive LLMLingua failures; bypasses compression during cooldown."""

    def __init__(self, max_failures: int = 2, cooldown_minutes: int = 10):
        self.max_failures = max_failures
        self.cooldown_seconds = cooldown_minutes * 60
        self.failures = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"
        self._lock = threading.Lock()

    def can_execute(self) -> bool:
        with self._lock:
            if self.state == "CLOSED":
                return True
            if time.time() - self.last_failure_time >= self.cooldown_seconds:
                logger.info("[CircuitBreaker] Cooldown expired. Resetting to CLOSED.")
                self.state = "CLOSED"
                self.failures = 0
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            if self.state == "OPEN":
                self.state = "CLOSED"
            self.failures = 0

    def record_failure(self) -> None:
        with self._lock:
            self.failures += 1
            self.last_failure_time = time.time()
            if self.failures >= self.max_failures:
                self.state = "OPEN"
                logger.warning(
                    "[CircuitBreaker] OPENED after %d consecutive failures. "
                    "Bypassing compression for %ds.",
                    self.failures,
                    self.cooldown_seconds,
                )

    def get_state_for_metrics(self) -> int:
        """1 if OPEN, 0 if CLOSED — for Prometheus gauge."""
        with self._lock:
            return 1 if self.state == "OPEN" else 0


# ---------------------------------------------------------------------------
# Compression client
# ---------------------------------------------------------------------------

def compress_via_llmlingua(messages: list, config: dict) -> tuple:
    """
    Send messages to the external LLMLingua container for compression.

    Returns (compressed_messages, original_tokens, compressed_tokens, errored).
    On failure returns (messages, 0, 0, True) so the caller knows to record
    a circuit breaker failure and fall back to passthrough.
    On empty input returns (messages, 0, 0, False) — not an error.
    """
    llmlingua_url = config.get("llmlinguaUrl", "")
    if not llmlingua_url:
        logger.info("No llmlinguaUrl configured, skipping compression")
        return messages, 0, 0, False

    # Approximate token count
    original_tokens = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            original_tokens += len(content) // 4
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    original_tokens += len(part.get("text", "")) // 4

    if original_tokens == 0:
        # Nothing to compress — not an error
        return messages, 0, 0, False

    payload = {
        "model": "llmlingua",
        "messages": messages,
        "max_tokens": 1,
    }

    compression_rate = config.get("compressionRate", 0.6)
    if compression_rate is not None:
        payload["compression_rate"] = compression_rate

    url = llmlingua_url.rstrip("/") + "/v1/chat/completions"

    # Per NK-09 edge cases: if LLMLingua is unreachable or still loading (503),
    # retry once after 5s before falling back to passthrough.
    response = None
    last_error = None
    for attempt in (1, 2):
        try:
            response = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
        except requests.exceptions.RequestException as e:
            response = None
            last_error = e
        if response is not None and response.status_code != 503:
            break
        if attempt == 1:
            reason = "unreachable" if response is None else "still loading (503)"
            logger.info("LLMLingua %s, retrying once in 5s...", reason)
            time.sleep(5)

    if response is None:
        logger.warning("LLMLingua unavailable (%s), falling back to passthrough", last_error)
        return messages, 0, 0, True
    if response.status_code != 200:
        logger.warning("LLMLingua returned %s, skipping compression", response.status_code)
        return messages, 0, 0, True

    try:
        result = response.json()
        compressed_messages = result.get("choices", [{}])[0].get(
            "message", {}
        ).get("content", messages)

        # Handle various response shapes
        if isinstance(compressed_messages, str):
            try:
                parsed = json.loads(compressed_messages)
                if isinstance(parsed, list):
                    compressed_messages = parsed
                elif isinstance(parsed, dict):
                    compressed_messages = [parsed]
                else:
                    compressed_messages = [{"role": "system", "content": compressed_messages}]
            except (json.JSONDecodeError, TypeError):
                compressed_messages = [{"role": "system", "content": compressed_messages}]

        compressed_tokens = 0
        for msg in compressed_messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                compressed_tokens += len(content) // 4
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        compressed_tokens += len(part.get("text", "")) // 4

        saved = original_tokens - compressed_tokens
        if saved > 0:
            logger.info(
                "LLMLingua compression: %d -> %d tokens (%d saved)",
                original_tokens,
                compressed_tokens,
                saved,
            )
        return compressed_messages, original_tokens, compressed_tokens, False

    except (ValueError, KeyError, TypeError) as e:
        logger.warning("LLMLingua response parse failed (%s), falling back to passthrough", e)
        return messages, 0, 0, True
