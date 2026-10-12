"""Cliente LLM configurable (anthropic | openai | ollama | none) con caché en disco (D-007).

Si falta la API key o la librería, el cliente cae a `none` y los módulos que lo usan
aplican plantillas deterministas; así el pipeline completo es reproducible sin LLM.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from src.common.io import read_json, stable_hash, write_json
from src.common.logging_utils import get_logger

log = get_logger(__name__)


class LLMClient:
    def __init__(self, provider: str, model: str | None, temperature: float = 0.0, max_tokens: int = 600,
                 cache_dir: Path | None = None, ollama_url: str = "http://localhost:11434"):
        self.provider, self.model = provider, model
        self.temperature, self.max_tokens = temperature, max_tokens
        self.cache_dir, self.ollama_url = cache_dir, ollama_url
        self._client = None
        if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
            log.warning("ANTHROPIC_API_KEY no definida → LLM desactivado (provider=none)")
            self.provider = "none"
        if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
            log.warning("OPENAI_API_KEY no definida → LLM desactivado (provider=none)")
            self.provider = "none"

    @classmethod
    def from_config(cls, cfg) -> "LLMClient":
        c = cfg.context
        prov = c["provider"]
        return cls(prov, c["model"].get(prov), c["temperature"], c["max_tokens"],
                   cfg.path("cache") / "llm", c["ollama_url"])

    @property
    def enabled(self) -> bool:
        return self.provider != "none"

    def complete(self, system: str, user: str) -> str | None:
        if not self.enabled:
            return None
        key = stable_hash(self.provider, self.model, self.temperature, system, user)
        cache = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if cache and cache.exists():
            return read_json(cache)["text"]
        try:
            text = getattr(self, f"_{self.provider}")(system, user)
        except Exception as exc:
            log.warning("llamada LLM falló (%s): %s", self.provider, exc)
            return None
        if cache:
            write_json(cache, {"provider": self.provider, "model": self.model, "text": text})
        return text

    def _anthropic(self, system: str, user: str) -> str:
        import anthropic

        self._client = self._client or anthropic.Anthropic()
        resp = self._client.messages.create(model=self.model, max_tokens=self.max_tokens, temperature=self.temperature,
                                            system=system, messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    def _openai(self, system: str, user: str) -> str:
        from openai import OpenAI

        self._client = self._client or OpenAI()
        resp = self._client.chat.completions.create(model=self.model, temperature=self.temperature,
                                                    max_tokens=self.max_tokens,
                                                    messages=[{"role": "system", "content": system},
                                                              {"role": "user", "content": user}])
        return resp.choices[0].message.content or ""

    def _ollama(self, system: str, user: str) -> str:
        import urllib.request

        body = json.dumps({"model": self.model, "stream": False, "options": {"temperature": self.temperature},
                           "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        req = urllib.request.Request(f"{self.ollama_url}/api/chat", data=body.encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())["message"]["content"]


def parse_json(text: str | None) -> dict | list | None:
    if not text:
        return None
    m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
