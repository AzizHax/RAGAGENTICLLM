# phenorag/utils/llm_client.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests


@dataclass
class LLMResponseMeta:
    model: str
    latency_ms: float
    status_code: int
    error: Optional[str] = None
    raw_text: Optional[str] = None


class LLMClient:
    """
    Small client for Ollama-like /api/generate.

    - Handles timeouts, retries
    - Can enforce JSON output via `format="json"`
    - Returns (parsed_json, meta)
    """

    def __init__(self, base_url: str, default_timeout_s: int = 120, max_retries: int = 2):
        self.base_url = base_url.rstrip("/")
        self.default_timeout_s = default_timeout_s
        self.max_retries = max_retries

    def generate(
        self,
        *,
        prompt: str,
        model: str,
        temperature: float = 0.1,
        timeout_s: Optional[int] = None,
        response_format: Optional[str] = "json",  # "json" or None
        extra: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], LLMResponseMeta]:
        url = f"{self.base_url}/api/generate"
        timeout_s = timeout_s or self.default_timeout_s
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "temperature": temperature,
        }
        if response_format:
            payload["format"] = response_format
        if extra:
            payload.update(extra)

        last_error = None
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                resp = requests.post(url, json=payload, timeout=timeout_s)
                latency_ms = (time.perf_counter() - t0) * 1000

                if resp.status_code != 200:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:500]}"
                    continue

                data = resp.json()
                raw = data.get("response", "")

                # If format=json, Ollama returns a JSON string inside "response"
                try:
                    parsed = json.loads(raw) if response_format == "json" else {"text": raw}
                    return parsed, LLMResponseMeta(
                        model=model,
                        latency_ms=latency_ms,
                        status_code=resp.status_code,
                        raw_text=raw,
                    )
                except Exception as e:
                    last_error = f"JSON parse error: {e}; raw={raw[:500]}"
                    continue

            except Exception as e:
                latency_ms = (time.perf_counter() - t0) * 1000
                last_error = str(e)

            # retry

        return None, LLMResponseMeta(
            model=model,
            latency_ms=0.0,
            status_code=-1,
            error=last_error,
        )