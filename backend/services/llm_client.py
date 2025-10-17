"""
Minimal LLM client using unified settings (Azure-aware).

- Reads provider, model/deploy, api key, endpoint, api_version, temperature from settings.py
- For provider='azure', uses the Azure Chat Completions path and 'api-key' header
- For provider='openai' (or others), uses the raw endpoint + Bearer header
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
        self.provider = (settings.llm_provider or "").lower().strip()
        self.endpoint = settings.llm_api_endpoint or ""
        self.api_key = settings.llm_api_key
        self.model = settings.llm_model
        # Azure specifics
        self.deployment = getattr(settings, "llm_deploy", self.model)
        self.api_version = getattr(settings, "llm_api_version", "2024-12-01-preview")
        # Misc
        self.temperature = float(getattr(settings, "temperature", 0.3))
        self.timeout = int(getattr(settings, "llm_timeout_secs", 60))

    def _url(self) -> str:
        if self.provider == "azure":
            base = self.endpoint.rstrip("/")  # do not mutate settings; just for safe join
            return f"{base}/openai/deployments/{self.deployment}/chat/completions?api-version={self.api_version}"
        # OpenAI-style (expects a full /v1/chat/completions endpoint in settings)
        return self.endpoint

    def _headers(self) -> dict:
        if self.provider == "azure":
            return {"api-key": self.api_key, "Content-Type": "application/json"}
        # Default to OpenAI-compatible header
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def chat(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Send a chat completion request and return text content or None."""
        if not settings.validate_llm_config():
            logger.error("LLM config invalid — missing key(s).")
            return None

        url = self._url()
        headers = self._headers()
        # For Azure, the "model" field should be the deployment name; harmless for others
        payload = {
            "model": self.deployment if self.provider == "azure" else self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

        t0 = time.time()
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            latency_ms = int((time.time() - t0) * 1000)

            if resp.status_code == 200:
                data = resp.json()
                content = (data.get("choices") or [{}])[0].get("message", {}).get("content")
                if content is None:
                    logger.error("LLM 200 OK but missing choices[0].message.content")
                    return None
                logger.info("LLM OK (provider=%s, model=%s, %d ms)", self.provider, self.model, latency_ms)
                return content

            # Log a short snippet of body for debugging
            body_snip = resp.text[:300].replace("\n", " ")
            logger.error("LLM HTTP %s at %s — %s", resp.status_code, url, body_snip)
            return None

        except requests.Timeout:
            logger.error("LLM request timed out at %s", url)
            return None
        except requests.ConnectionError as e:
            logger.error("LLM connection error at %s: %s", url, e)
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
