"""
Minimal LLM client using unified settings.

- Reads provider, model, API key, endpoint, temperature, etc. from settings.py
- Simple `chat(messages)` method
- Returns text or None if failed
"""

import logging
import time
import requests
from typing import List, Dict, Optional

from ..config.settings import settings

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(h)
logger.setLevel(getattr(settings, "log_level", logging.INFO))


class LLMClient:
    def __init__(self):
        self.provider = settings.llm_provider
        self.endpoint = settings.llm_api_endpoint
        self.api_key = settings.llm_api_key
        self.model = settings.llm_model
        self.temperature = float(getattr(settings, "temperature", 0.3))
        # self.max_tokens = 1024  # could be env-driven later

    def chat(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Send a chat completion request and return text content or None."""
        if not settings.validate_llm_config():
            logger.error("LLM config invalid — missing key(s).")
            return None

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        t0 = time.time()
        try:
            resp = requests.post(self.endpoint, json=payload, headers=headers, timeout=300)
            latency_ms = int((time.time() - t0) * 1000)

            if resp.status_code == 200:
                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content")
                logger.info("LLM OK (provider=%s, model=%s, %d ms)", self.provider, self.model, latency_ms)
                return content

            logger.error("LLM HTTP error %s", resp.status_code)
            return None

        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return None


def main():
    """Quick CLI test."""
    client = LLMClient()
    result = client.chat([{"role": "user", "content": "Say hello in one word"}])
    print("Response:", result or "FAIL")


if __name__ == "__main__":
    main()
