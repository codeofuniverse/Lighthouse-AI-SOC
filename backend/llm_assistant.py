"""LLM-generated security explanations.

Provider priority (auto-detected from env):
  1. Groq         (LLM_PROVIDER=groq,         LLM_API_KEY=gsk_...)
  2. OpenAI       (LLM_PROVIDER=openai,       LLM_API_KEY=sk-...)
  3. Ollama Cloud (LLM_PROVIDER=ollama_cloud, LLM_API_KEY=<hex.b64> from api.ollama.com)
  4. Ollama local (LLM_PROVIDER=ollama,       no key needed)
  5. Rule-based fallback (always works, no dependencies)

Set in .env:
  LLM_PROVIDER=ollama_cloud
  LLM_API_KEY=85588afa28d04bcab08bcc5bb2f72aab.N6MAhy_vb4fvTJ-5__SB-BgZ
  LLM_MODEL=gemma3:4b
"""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
_API_KEY  = os.getenv("LLM_API_KEY", "")
_TIMEOUT  = 15.0

_PROVIDER_DEFAULTS = {
    "groq":         {"url": "https://api.groq.com/openai/v1/chat/completions", "model": "llama-3.1-8b-instant"},
    "openai":       {"url": "https://api.openai.com/v1/chat/completions",      "model": "gpt-4o-mini"},
    "ollama_cloud": {"url": "https://api.ollama.com/api/chat",                 "model": "gemma3:4b"},
    "ollama":       {"url": "http://localhost:11434/api/chat",                 "model": "qwen3:4b"},
}

_defaults = _PROVIDER_DEFAULTS.get(_PROVIDER, _PROVIDER_DEFAULTS["ollama"])
_MODEL    = os.getenv("LLM_MODEL", _defaults["model"])
_API_URL  = _defaults["url"]

_SYSTEM_PROMPT = (
    "You are a concise SOC analyst assistant. "
    "Explain network detections in 1-2 sentences. "
    "Be technical. No markdown. No bullet points. No preamble."
)

_USER_TEMPLATE = """\
Attack type: {prediction}
Source IP: {src_ip} (AbuseIPDB abuse score: {abuse_score}/100)
ML confidence: {confidence}%
Risk score: {risk_score}/100
Explain this detection for a security analyst in 1-2 sentences."""


class LLMAssistant:

    def __init__(self) -> None:
        self._consecutive_failures = 0

    async def is_available(self) -> bool:
        """Quick probe to check if the LLM endpoint is reachable."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                if _PROVIDER in ("ollama", "ollama_cloud"):
                    resp = await client.get(_API_URL.replace("/api/chat", "/api/tags"))
                else:
                    resp = await client.get(_API_URL.split("/chat")[0])
                return resp.status_code < 500
        except Exception:
            return False

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    async def explain(
        self,
        prediction: str,
        src_ip: str,
        confidence: float,
        risk_score: float,
        abuse_score: int = 0,
    ) -> str:
        user_msg = _USER_TEMPLATE.format(
            prediction=prediction,
            src_ip=src_ip,
            abuse_score=abuse_score,
            confidence=round(confidence * 100, 1),
            risk_score=round(risk_score, 1),
        )
        if _PROVIDER in ("ollama", "ollama_cloud"):
            return await self._call_ollama(user_msg, prediction, src_ip, confidence, risk_score)
        return await self._call_openai_compat(user_msg, prediction, src_ip, confidence, risk_score)

    async def _call_openai_compat(self, user_msg: str, *fallback_args) -> str:
        if not _API_KEY:
            logger.warning("LLM_API_KEY not set for provider=%s, using fallback", _PROVIDER)
            return self._fallback(*fallback_args)
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.post(
                    _API_URL,
                    headers={"Authorization": f"Bearer {_API_KEY}", "Content-Type": "application/json"},
                    json={
                        "model": _MODEL,
                        "messages": [
                            {"role": "system", "content": _SYSTEM_PROMPT},
                            {"role": "user",   "content": user_msg},
                        ],
                        "max_tokens": 120,
                        "temperature": 0.3,
                    },
                )
                resp.raise_for_status()
                self._consecutive_failures = 0
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            self._consecutive_failures += 1
            if self._consecutive_failures >= 3:
                logger.warning("LLM consecutive failures=%d (%s): %s", self._consecutive_failures, _PROVIDER, exc)
            else:
                logger.debug("LLM call failed (%s): %s — using fallback", _PROVIDER, exc)
            return self._fallback(*fallback_args)

    async def _call_ollama(self, user_msg: str, *fallback_args) -> str:
        """Ollama native chat format — works for local Ollama and ollama.com cloud."""
        headers: dict[str, str] = {}
        if _API_KEY:
            headers["Authorization"] = f"Bearer {_API_KEY}"
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.post(
                    _API_URL,
                    headers=headers,
                    json={
                        "model": _MODEL,
                        "messages": [
                            {"role": "system", "content": _SYSTEM_PROMPT},
                            {"role": "user",   "content": user_msg},
                        ],
                        "stream": False,
                        "options": {"num_predict": 120, "temperature": 0.3},
                        "think": False,
                    },
                )
                resp.raise_for_status()
                self._consecutive_failures = 0
                return resp.json()["message"]["content"].strip()
        except Exception as exc:
            self._consecutive_failures += 1
            if self._consecutive_failures >= 3:
                logger.warning("LLM consecutive failures=%d (%s): %s", self._consecutive_failures, _PROVIDER, exc)
            else:
                logger.debug("Ollama call failed (%s): %s — using fallback", _PROVIDER, exc)
            return self._fallback(*fallback_args)

    def _fallback(self, prediction: str, src_ip: str, confidence: float, risk_score: float) -> str:
        return (
            f"{prediction} detected from {src_ip} with {confidence:.0%} ML confidence. "
            f"Risk score: {risk_score:.0f}/100."
        )


llm = LLMAssistant()
