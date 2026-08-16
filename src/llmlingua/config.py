"""Configuration loader for LLMLingua-2 server — reads app.yml and returns a dict."""

import logging
import os
from pathlib import Path

import yaml

logger = logging.getLogger("llmlingua.config")


def load_llmlingua_config(path: str | None = None) -> dict:
    """Load and return llmlingua app.yml config. Returns empty dict on failure.
    
    Path resolution order:
      1. `path` argument (if provided)
      2. $LLMLINGUA_CONFIG_PATH env var (if set)
      3. /app/app.yml (Docker — Dockerfile COPYs it here)
      4. docker/llmlingua/app.yml (local dev — relative to project root)
    """
    config_path = path or os.environ.get("LLMLINGUA_CONFIG_PATH")
    
    if not config_path:
        # Try Docker path first
        if Path("/app/app.yml").exists():
            config_path = "/app/app.yml"
        else:
            # Local dev: find project root by walking up from this file
            here = Path(__file__).resolve().parent  # src/llmlingua/
            for parent in (here, *here.parents):
                candidate = parent / "docker" / "llmlingua" / "app.yml"
                if candidate.exists():
                    config_path = str(candidate)
                    break
            else:
                # Last resort: cwd-relative
                config_path = "docker/llmlingua/app.yml"
    
    try:
        with open(config_path) as f:
            logger.info("LLMLingua config loaded from %s", config_path)
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning("LLMLingua config load failed for %s (%s), using defaults", config_path, e)
        return {}

