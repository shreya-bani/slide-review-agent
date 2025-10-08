"""
Provider-agnostic settings with ONE LLM config:
  LLM_PROVIDER, LLM_API_KEY, LLM_MODEL, LLM_API_ENDPOINT

CLI:
  python -m backend.config.settings --verbose
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# logging bootstrap
logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"
if ENV_EXAMPLE.exists():
    load_dotenv(ENV_EXAMPLE, override=False)
    logger.info("Loaded defaults from %s (override=False)", ENV_EXAMPLE)

class Settings(BaseSettings):
    # App
    app_name: str = Field(default="slide-review-agent", alias="APP_NAME")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    debug: bool = Field(default=True, alias="DEBUG")
    
    max_file_size_mb: int = Field(default=50, alias="MAX_FILE_SIZE_MB")

    # DB
    database_url: str = Field(default=f"sqlite:///{PROJECT_ROOT}/data/slide_review.db",
                              alias="DATABASE_URL")
    log_dir: str = Field(default=f"{PROJECT_ROOT}/logs", alias="LOG_DIR")
    output_dir: str = Field(default=f"{PROJECT_ROOT}/outputs", alias="OUTPUT_DIR")
    upload_dir: str = Field(default=f"{PROJECT_ROOT}/uploads", alias="UPLOAD_DIR")
    
    # SINGLE LLM CONFIG
    llm_provider: str = Field(default="huggingface", alias="LLM_PROVIDER")
    llm_api_key: Optional[str] = Field(default=None, alias="LLM_API_KEY")
    llm_model: str = Field(default="google/gemma-2-2b-it", alias="LLM_MODEL")
    llm_api_endpoint: str = Field(default="https://router.huggingface.co/v1/chat/completions", alias="LLM_API_ENDPOINT")

    # Logging
    log_level: int = Field(default=logging.INFO, alias="LOG_LEVEL")

    class Config:
        case_sensitive = False

    # helpers
    def validate_llm_config(self) -> bool:
        return bool((self.llm_api_key or "").strip()
                    and (self.llm_model or "").strip()
                    and (self.llm_api_endpoint or "").strip())

    def log_summary(self) -> None:
        logger.info("App: %s | Env: %s | Debug=%s", self.app_name, self.environment, self.debug)
        logger.info("Database URL: %s", self.database_url)
        logger.info("LLM Provider: %s", self.llm_provider)
        logger.info("LLM Model: %s", self.llm_model)
        logger.info("LLM Endpoint: %s", self.llm_api_endpoint)
        logger.info("LLM API key set: %s", "yes" if self.llm_api_key else "no")
        ok = self.validate_llm_config()
        logger.info("LLM config valid: %s", ok)
        if not ok:
            missing = []
            if not (self.llm_api_key or "").strip(): missing.append("LLM_API_KEY")
            if not (self.llm_model or "").strip(): missing.append("LLM_MODEL")
            if not (self.llm_api_endpoint or "").strip(): missing.append("LLM_API_ENDPOINT")
            logger.warning("Missing LLM keys: %s", ", ".join(missing) or "unknown")


    def ensure_directories(self):
        """Create required directories if they don't exist."""
        directories = [
            self.upload_dir,
            self.output_dir,
            self.log_dir,
        ]
        
        logger.info(f"Creating directories from project root: {PROJECT_ROOT}")
        for directory in directories:
            if directory:  # skip None
                Path(directory).mkdir(parents=True, exist_ok=True)
                logger.debug(f" - {directory}")

    @property
    def database_path(self) -> Path:
        """Return a filesystem Path for sqlite URLs; best-effort fallback otherwise."""
        prefix = "sqlite:///"
        if self.database_url.startswith(prefix):
            return Path(self.database_url[len(prefix):])
        parsed = urlparse(self.database_url)
        if parsed.scheme == "sqlite" and parsed.path:
            return Path(parsed.path)
        # Fallback (non-sqlite): point at default data file so health can still pass path checks
        return Path(PROJECT_ROOT) / "data" / "slide_review.db"


settings = Settings()
logger.setLevel(getattr(settings, "log_level", logging.INFO))

# CLI self-test
def _parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="Settings self-test")
    ap.add_argument("--verbose", action="store_true", help="DEBUG logs")
    return ap.parse_args()

def main():
    args = _parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
    logger.debug("PROJECT_ROOT=%s | .env.example exists=%s", PROJECT_ROOT, ENV_EXAMPLE.exists())
    settings.log_summary()

if __name__ == "__main__":
    main()
