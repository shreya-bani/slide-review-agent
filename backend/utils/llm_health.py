"""
Simple LLM health check.

Uses settings.llm_provider, llm_api_key, llm_model, llm_api_endpoint.
Tries a 1-token ping and logs success/failure.

Run:
    python -m backend.utils.llm_health
"""

import logging
import sys
import time
import requests

from ..config.settings import settings

# Logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(h)
logger.setLevel(getattr(settings, "log_level", logging.INFO))


def check_llm() -> dict:
    """Perform a minimal health check against the configured LLM endpoint."""
    if not settings.validate_llm_config():
        logger.error("LLM config invalid — missing key(s).")
        return {"status": "fail", "reason": "invalid_config"}

    # Log what we are about to use (without exposing secrets)
    logger.info("LLM Provider: %s", settings.llm_provider)
    logger.info("LLM Model: %s", settings.llm_model)
    logger.info("LLM Endpoint: %s", settings.llm_api_endpoint)
    logger.info("LLM API key: %s", "set" if settings.llm_api_key else "missing")

    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": "ping"},
            {"role": "user", "content": "ping"},
        ],
        "temperature": 0.0,
        "max_tokens": 1,
    }
    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }

    t0 = time.time()
    try:
        resp = requests.post(settings.llm_api_endpoint, json=payload,
                             headers=headers, timeout=10)
        latency_ms = int((time.time() - t0) * 1000)

        if resp.status_code == 200:
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content")
            if content is not None:
                logger.info("LLM connectivity OK (provider=%s, model=%s, %d ms)",
                            settings.llm_provider, settings.llm_model, latency_ms)
                return {"status": "success", "latency_ms": latency_ms}
            logger.error("LLM responded 200 but no content field.")
            return {"status": "fail", "reason": "malformed_response"}

        logger.error("LLM responded with HTTP %s", resp.status_code)
        return {"status": "fail", "reason": f"http_{resp.status_code}"}

    except requests.Timeout:
        logger.error("LLM request timed out")
        return {"status": "fail", "reason": "timeout"}
    except requests.ConnectionError:
        logger.error("LLM connection error")
        return {"status": "fail", "reason": "connection_error"}
    except Exception as e:
        logger.exception("Unexpected error")
        return {"status": "fail", "reason": f"unexpected_error:{type(e).__name__}"}


def main():
    result = check_llm()
    logger.info("Health result: %s", result)
    sys.exit(0 if result["status"] == "success" else 1)


if __name__ == "__main__":
    main()
