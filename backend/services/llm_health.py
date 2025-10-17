"""
Simple LLM health check (AzureOpenAI SDK version).

- Uses settings.llm_provider / llm_api_key / llm_api_version / llm_api_endpoint / llm_deploy
- Does NOT modify or normalize the endpoint; it is passed to the SDK as-is.
- Sends a 1-token ping via Chat Completions and reports success/latency.

Run:
    python -m backend.services.llm_health
"""

import logging
import sys
import time
from typing import Dict

from ..config.settings import settings

# Azure OpenAI SDK
from openai import AzureOpenAI

# Logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(h)
logger.setLevel(getattr(settings, "log_level", logging.INFO))


def check_llm() -> Dict[str, object]:
    """Perform a minimal health check against the configured LLM endpoint."""
    if not settings.validate_llm_config():
        logger.error("LLM config invalid — missing key(s).")
        return {"status": "fail", "reason": "invalid_config"}

    provider = (settings.llm_provider or "").lower().strip()
    deploy = getattr(settings, "llm_deploy", settings.llm_model)

    logger.info("LLM Provider: %s", settings.llm_provider)
    logger.info("LLM Model: %s", settings.llm_model)
    logger.info("LLM Deploy: %s", deploy)
    logger.info("LLM Endpoint: %s", settings.llm_api_endpoint)
    logger.info("LLM API Version: %s", getattr(settings, "llm_api_version", ""))

    try:
        if provider != "azure":
            return {"status": "fail", "reason": "unsupported_provider"}

        # Create Azure client using settings *as-is* (no endpoint modification here)
        client = AzureOpenAI(
            api_key=settings.llm_api_key,
            azure_endpoint=settings.llm_api_endpoint,
            api_version=getattr(settings, "llm_api_version", "2024-12-01-preview"),
        )

        t0 = time.time()
        resp = client.chat.completions.create(
            model=deploy,  # deployment name
            messages=[
                {"role": "system", "content": "ping"},
                {"role": "user", "content": "ping"},
            ],
            temperature=0.0,
            max_tokens=1,
        )
        latency_ms = int((time.time() - t0) * 1000)

        # Defensive parse
        choice = (getattr(resp, "choices", None) or [None])[0]
        content = getattr(getattr(choice, "message", None), "content", None)
        if content is None:
            logger.error("200 OK but no content in response")
            return {"status": "fail", "reason": "malformed_response"}

        logger.info("LLM connectivity OK (%d ms)", latency_ms)
        return {"status": "success", "latency_ms": latency_ms}

    # Common failure buckets
    except Exception as e:
        # Keep this generic to avoid SDK-version-specific imports
        msg = str(e)
        logger.error("LLM request failed: %s", msg)
        # Try to include a short reason key for your status dashboard
        reason = "unexpected_error:" + type(e).__name__
        return {"status": "fail", "reason": reason, "detail": msg[:300]}


def main():
    result = check_llm()
    logger.info("Health result: %s", result)
    sys.exit(0 if result.get("status") == "success" else 1)


if __name__ == "__main__":
    main()
