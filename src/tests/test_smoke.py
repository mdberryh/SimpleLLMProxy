"""
Author:      AI
Date:        2026-06-15
Description: Smoke tests for proxy packaging + core routing helpers - NK-09

Run from the llm-proxy/ directory:
    python -m unittest discover -s tests

These tests guard against import/packaging regressions (the kind that broke
`python -m src.main`) and verify the pure routing/compression helpers.
"""

import importlib
import unittest

from src.proxy.compression import CircuitBreaker
from src.proxy.router import (
    count_tokens,
    is_multimodal,
    reassemble_messages,
    rewrite_model,
    should_compress,
    split_messages_for_compression,
)


class TestPackaging(unittest.TestCase):
    def test_main_module_imports(self):
        """`python -m src.main` must be able to resolve `main` (regression guard)."""
        mod = importlib.import_module("src.main")
        self.assertTrue(callable(mod.main))
        self.assertEqual(mod.main.__module__, "src.proxy.server")


class TestShouldCompress(unittest.TestCase):
    CONFIG = {"compressionModels": ["qwen3.6-llmlingua", "*-llmlingua", "vision-*"]}

    def test_exact_match(self):
        self.assertTrue(should_compress("qwen3.6-llmlingua", self.CONFIG))

    def test_suffix_wildcard(self):
        self.assertTrue(should_compress("anything-llmlingua", self.CONFIG))

    def test_prefix_wildcard(self):
        self.assertTrue(should_compress("vision-7b", self.CONFIG))

    def test_no_match(self):
        self.assertFalse(should_compress("qwen3.6-agents", self.CONFIG))

    def test_empty_patterns(self):
        self.assertFalse(should_compress("anything", {}))


class TestSplitAndReassemble(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(split_messages_for_compression([]), ([], [], []))

    def test_system_isolated(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ]
        system, middle, recent = split_messages_for_compression(msgs, keep_recent_count=4)
        self.assertEqual(system, [{"role": "system", "content": "sys"}])
        self.assertEqual(middle, [])  # not enough history to compress
        self.assertEqual(len(recent), 2)

    def test_middle_recent_split(self):
        convo = [{"role": "user", "content": str(i)} for i in range(6)]
        system, middle, recent = split_messages_for_compression(convo, keep_recent_count=4)
        self.assertEqual(system, [])
        self.assertEqual(len(middle), 2)
        self.assertEqual(len(recent), 4)

    def test_reassemble_order(self):
        system = [{"role": "system", "content": "s"}]
        compressed = [{"role": "system", "content": "c"}]
        recent = [{"role": "user", "content": "r"}]
        result = reassemble_messages(system, compressed, recent)
        self.assertEqual([m["content"] for m in result], ["s", "c", "r"])


class TestMultimodal(unittest.TestCase):
    def test_detects_image(self):
        msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]
        self.assertTrue(is_multimodal(msgs))

    def test_text_only(self):
        msgs = [{"role": "user", "content": "hi"}]
        self.assertFalse(is_multimodal(msgs))

    def test_text_block_list(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        self.assertFalse(is_multimodal(msgs))


class TestCountTokens(unittest.TestCase):
    def test_string_content(self):
        self.assertEqual(count_tokens([{"role": "user", "content": "a" * 8}]), 2)

    def test_text_blocks(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "a" * 4}]}]
        self.assertEqual(count_tokens(msgs), 1)


class TestRewriteModel(unittest.TestCase):
    def test_mapped(self):
        data = {"model": "qwen3.6-llmlingua"}
        rewrite_model(data, {"modelMap": {"qwen3.6-llmlingua": "Qwen.gguf"}})
        self.assertEqual(data["model"], "Qwen.gguf")

    def test_default_fallback(self):
        data = {"model": "unknown"}
        rewrite_model(data, {"modelMap": {}, "_defaultFallback": "Fallback.gguf"})
        self.assertEqual(data["model"], "Fallback.gguf")

    def test_empty_model_untouched(self):
        data = {"model": ""}
        rewrite_model(data, {"_defaultFallback": "Fallback.gguf"})
        self.assertEqual(data["model"], "")


class TestCircuitBreaker(unittest.TestCase):
    def test_opens_after_failures(self):
        cb = CircuitBreaker(max_failures=2, cooldown_minutes=10)
        self.assertTrue(cb.can_execute())
        cb.record_failure()
        self.assertTrue(cb.can_execute())
        cb.record_failure()
        self.assertFalse(cb.can_execute())
        self.assertEqual(cb.get_state_for_metrics(), 1)

    def test_success_resets(self):
        cb = CircuitBreaker(max_failures=2, cooldown_minutes=10)
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        self.assertTrue(cb.can_execute())

    def test_cooldown_reopens(self):
        cb = CircuitBreaker(max_failures=1, cooldown_minutes=10)
        cb.record_failure()
        self.assertFalse(cb.can_execute())
        cb.cooldown_seconds = 0  # simulate elapsed cooldown
        self.assertTrue(cb.can_execute())


if __name__ == "__main__":
    unittest.main()
