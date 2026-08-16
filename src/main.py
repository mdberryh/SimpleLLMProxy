#!/usr/bin/env python3
"""
LLMLingua-2 Proxy — main entry point.

Lightweight relay with zero ML dependencies. All config from app.yml.

Architecture (4 containers):
  OpenClaw → Proxy (this) → LLMLingua → llama-vision / llama-agents → Venice.AI

Usage:
  python -m src.main
  (or via Docker ENTRYPOINT)
"""


from src.proxy.server import main

if __name__ == "__main__":
    main()
