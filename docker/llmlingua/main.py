"""LLMLingua-2 Compression & Tokenization Server.

Provides:
  - POST /v1/chat/completions  — compress messages via LLMLingua-2
  - POST /v1/tokenize         — count tokens using Qwen3-compatible encoding
  - GET  /health               — liveness check
  - GET  /ready                — readiness check (model loaded)

The compression model (microsoft/llmlingua-2-bert-base-uncased) is loaded
on startup. Tokenization uses tiktoken o200k_base (GPT-4o encoding) which
closely approximates Qwen3's tokenizer vocabulary.
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import tiktoken
import torch
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("llmlingua.server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def load_llmlingua_config(path: str | None = None) -> dict:
    config_path = path or os.environ.get("LLMLINGUA_CONFIG_PATH")
    if not config_path:
        if Path("/app/app.yml").exists():
            config_path = "/app/app.yml"
        else:
            config_path = "app.yml"
    try:
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

compressor = None
tokenizer = None
START_TIME = time.time()
CONFIG = load_llmlingua_config()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class MessageContent(BaseModel):
    """OpenAI-style content block."""
    type: str
    text: Optional[str] = None
    image_url: Optional[dict] = None


class Message(BaseModel):
    """OpenAI-style chat message."""
    role: str
    content: str | list | None = None


class TokenizeRequest(BaseModel):
    messages: list[Message]
    model: str = "qwen3"


class TokenizeResponse(BaseModel):
    token_count: int
    model: str


class ChatCompletionRequest(BaseModel):
    model: str = "llmlingua"
    messages: list[Message]
    compression_rate: float = Field(default=None, ge=0.1, le=1.0)
    max_tokens: int = 1  # Unused but required for OpenAI compatibility


class ChatCompletionResponse(BaseModel):
    model: str
    choices: list[dict]
    usage: dict


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global compressor, tokenizer

    # Load tokenizer first (fast)
    logger.info("Loading tokenizer (tiktoken o200k_base for Qwen3 approximation)...")
    tokenizer = tiktoken.get_encoding("o200k_base")
    logger.info("Tokenizer loaded: o200k_base (~200K vocab, Qwen3-compatible)")

    # Load LLMLingua-2 compression model (slow — downloads on first run)
    # Use the ROCm/HIP GPU when available (PyTorch ROCm exposes it as "cuda");
    # fall back to CPU if no GPU is visible to the container.
    device = CONFIG.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    model_name = CONFIG.get("modelName", "microsoft/llmlingua-2-xlm-roberta-large-meetingbank")

    logger.info(
        "Loading LLMLingua-2 compression model '%s' on device=%s (this may take a few minutes on first run)...",
        model_name,
        device,
    )
    try:
        from llmlingua import PromptCompressor
        compressor = PromptCompressor(
            model_name=model_name,
            use_llmlingua2=True,
            device_map=device,
        )
        logger.info("LLMLingua-2 model loaded successfully on %s", device)
    except Exception as e:
        logger.error("Failed to load LLMLingua-2 model: %s", e)
        # Don't crash — tokenization will still work, compression will 503

    yield


app = FastAPI(title="LLMLingua-2 Server", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Liveness probe — container is running."""
    return {
        "status": "ok",
        "model_loaded": compressor is not None,
        "tokenizer_loaded": tokenizer is not None,
        "uptime_seconds": int(time.time() - START_TIME),
    }


@app.get("/ready")
async def ready():
    """Readiness probe — service is ready to handle requests."""
    if compressor is None:
        raise HTTPException(status_code=503, detail="LLMLingua-2 model not loaded")
    if tokenizer is None:
        raise HTTPException(status_code=503, detail="Tokenizer not loaded")
    return {"status": "ready"}


# ---------------------------------------------------------------------------
# Token counting helpers
# ---------------------------------------------------------------------------

def _count_message_tokens(msg: Message) -> int:
    """Count tokens in a single OpenAI-style message."""
    if tokenizer is None:
        # Fallback: rough approximation
        content = msg.content
        if isinstance(content, str):
            return max(1, len(content) // 4)
        return 0

    total = 0
    content = msg.content
    if isinstance(content, str):
        total += len(tokenizer.encode(content))
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    total += len(tokenizer.encode(part.get("text", "")))
                elif part.get("type") == "image_url":
                    # Image tokens vary by resolution; conservative estimate
                    total += 256
    elif content is None:
        pass  # Empty message
    return total


def _extract_text(msg: Message) -> str:
    """Extract plain text content from a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return " ".join(parts)
    return ""


# ---------------------------------------------------------------------------
# POST /v1/tokenize
# ---------------------------------------------------------------------------

@app.post("/v1/tokenize", response_model=TokenizeResponse)
async def tokenize(request: TokenizeRequest):
    """Count tokens for a list of OpenAI-style messages.

    Uses tiktoken o200k_base encoding, which closely approximates Qwen3's
    tokenizer (both are large-vocabulary BPE with ~150K-200K tokens).
    """
    if tokenizer is None:
        raise HTTPException(status_code=503, detail="Tokenizer not loaded")

    total_tokens = 0
    for msg in request.messages:
        total_tokens += _count_message_tokens(msg)

    return TokenizeResponse(token_count=total_tokens, model=request.model)


# ---------------------------------------------------------------------------
# POST /v1/chat/completions  — LLMLingua-2 compression
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """Compress messages using LLMLingua-2.

    Input: OpenAI-style messages array + compression_rate (0.1–1.0).
    Output: Compressed text in OpenAI chat completion format.

    The compression concatenates all input messages (with role labels)
    into a single context, compresses via LLMLingua-2 token classification,
    and returns the result as a single assistant message.
    """
    if compressor is None:
        raise HTTPException(status_code=503, detail="LLMLingua-2 model not loaded")

    # Use request compression_rate if provided, otherwise fall back to config default
    compression_rate = request.compression_rate
    if compression_rate is None:
        compression_rate = CONFIG.get("compressionRate", 0.6)

    # Count original tokens
    original_tokens = sum(_count_message_tokens(msg) for msg in request.messages)

    if original_tokens == 0:
        return ChatCompletionResponse(
            model="llmlingua-2",
            choices=[{
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "stop",
            }],
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    # Build context string with role labels for cross-turn awareness
    context_parts = []
    for msg in request.messages:
        text = _extract_text(msg)
        if text:
            context_parts.append(f"[{msg.role}]: {text}")

    if not context_parts:
        return ChatCompletionResponse(
            model="llmlingua-2",
            choices=[{
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "stop",
            }],
            usage={"prompt_tokens": original_tokens, "completion_tokens": 0, "total_tokens": original_tokens},
        )

    # Compress using LLMLingua-2
    try:
        context = "\n".join(context_parts)
        result = compressor.compress_prompt(
            context=context,
            rate=compression_rate,
            force_tokens=["\n", ".", "?", "!", ":", ";"],
        )
    except Exception as e:
        logger.error("LLMLingua-2 compression failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Compression failed: {e}")

    compressed_text = result.get("compressed_prompt", "")
    compressed_tokens = result.get("compressed_tokens", 0)

    # If LLMLingua didn't return token counts, estimate from the tokenizer
    if compressed_tokens == 0 and compressed_text and tokenizer:
        compressed_tokens = len(tokenizer.encode(compressed_text))

    logger.info(
        "Compression: %d -> %d tokens (%.1f%% reduction, rate=%.2f)",
        original_tokens,
        compressed_tokens,
        (1 - compressed_tokens / max(original_tokens, 1)) * 100,
        compression_rate,
    )

    return ChatCompletionResponse(
        model="llmlingua-2",
        choices=[{
            "index": 0,
            "message": {"role": "assistant", "content": compressed_text},
            "finish_reason": "stop",
        }],
        usage={
            "prompt_tokens": original_tokens,
            "completion_tokens": compressed_tokens,
            "total_tokens": original_tokens + compressed_tokens,
        },
    )


# ---------------------------------------------------------------------------
# Main entry point (for direct execution)
# ---------------------------------------------------------------------------

def main():
    """Run the server directly with uvicorn."""
    import uvicorn

    port = CONFIG.get("port", 8000)
    host = CONFIG.get("host", "0.0.0.0")

    logger.info("Starting LLMLingua-2 server on %s:%d", host, port)
    logger.info("Config: model=%s, device=%s, compressionRate=%.2f",
                CONFIG.get("modelName", "microsoft/llmlingua-2-bert-base-uncased"),
                CONFIG.get("device", "auto"),
                CONFIG.get("compressionRate", 0.6))

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
