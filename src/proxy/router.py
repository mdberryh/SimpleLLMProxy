"""Routing helpers — model rewriting, token counting, compression scoping."""

import logging

logger = logging.getLogger("proxy.router")


def should_compress(model_name: str, config: dict) -> bool:
    """Check if model matches any pattern in compressionModels."""
    patterns = config.get("compressionModels", [])
    if not patterns:
        return False
    for pattern in patterns:
        if pattern == model_name:
            return True
        if pattern.startswith("*") and model_name.endswith(pattern[1:]):
            return True
        if pattern.endswith("*") and model_name.startswith(pattern[:-1]):
            return True
    return False


def rewrite_model(request_data: dict, config: dict) -> None:
    """Rewrite model name via modelMap: OpenClaw internal ID → upstream name."""
    model_map = config.get("modelMap", {})
    model_name = request_data.get("model", "")
    if not model_name:
        return

    mapped = model_map.get(model_name)
    if mapped:
        logger.info("Model rewrite: '%s' -> '%s'", model_name, mapped)
        request_data["model"] = mapped
    else:
        default = config.get("_defaultFallback", "")
        if default:
            logger.info("No mapping for '%s', using default '%s'", model_name, default)
            request_data["model"] = default


def count_tokens(messages: list) -> int:
    """Approximate total token count for a message list."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(content) // 4
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += len(part.get("text", "")) // 4
    return total


def is_multimodal(messages: list) -> bool:
    """Return True if any message contains image_url blocks."""
    return any(
        isinstance(msg.get("content"), list)
        and any(
            isinstance(block, dict) and block.get("type") == "image_url"
            for block in msg["content"]
        )
        for msg in messages
    )


# ---------------------------------------------------------------------------
# Smart Compression Scope (Recency Bias)
# ---------------------------------------------------------------------------

def split_messages_for_compression(
    messages: list, keep_recent_count: int = 4
) -> tuple:
    """
    Split OpenAI-style messages into three parts:
      1. system_msgs  — all system prompts (never compressed)
      2. middle_msgs  — older conversation history (sent to LLMLingua)
      3. recent_msgs  — last N turns (kept raw for immediate context fidelity)

    Returns (system_msgs, middle_msgs, recent_msgs).
    """
    if not messages:
        return [], [], []

    system_msgs = [m for m in messages if m.get("role") == "system"]
    conversational_msgs = [m for m in messages if m.get("role") != "system"]

    if len(conversational_msgs) > keep_recent_count:
        middle_msgs = conversational_msgs[:-keep_recent_count]
        recent_msgs = conversational_msgs[-keep_recent_count:]
    else:
        middle_msgs = []
        recent_msgs = conversational_msgs

    return system_msgs, middle_msgs, recent_msgs


def reassemble_messages(
    system_msgs: list, compressed_middle: list, recent_msgs: list
) -> list:
    """Reassemble messages in chronological order: [system] + [compressed] + [recent]."""
    return system_msgs + compressed_middle + recent_msgs

