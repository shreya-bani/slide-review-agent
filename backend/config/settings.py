"""
Application configuration management using Pydantic settings.
"""
import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env.example file
load_dotenv(".env.example")

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    app_name: str = Field(default="slide-review-agent", alias="APP_NAME")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    debug: bool = Field(default=True, alias="DEBUG")
    
    # Database
    database_url: str = Field(default="sqlite:///./data/slide_review.db", alias="DATABASE_URL")
    
    # LLM Provider
    llm_provider: str = Field(default="groq", alias="LLM_PROVIDER")
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.1-8b-instant", alias="GROQ_MODEL")
    
    # Fallback providers
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4", alias="OPENAI_MODEL")
    
    # File handling
    max_file_size_mb: int = Field(default=50, alias="MAX_FILE_SIZE_MB")
    upload_dir: str = Field(default="./data/uploads", alias="UPLOAD_DIR")
    output_dir: str = Field(default="./data/outputs", alias="OUTPUT_DIR")
    
    # Demo mode
    demo_mode: bool = Field(default=True, alias="DEMO_MODE")
    
    # Paths
    style_guide_path: str = Field(default="./docs/style_template/", alias="STYLE_GUIDE_PATH")
    past_docs_dir: str = Field(default="./docs/past_docs/", alias="PAST_DOCS_DIR")
    
    class Config:
        case_sensitive = False
        
    def ensure_directories(self):
        """Create required directories if they don't exist."""
        directories = [
            self.upload_dir,
            self.output_dir,
            "data/logs",
            "data/vector_store",
            self.style_guide_path,
            self.past_docs_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    @property
    def database_path(self) -> str:
        """Extract database file path from SQLite URL."""
        if self.database_url.startswith("sqlite:///"):
            return self.database_url.replace("sqlite:///", "")
        return self.database_url
    
    def validate_llm_config(self) -> bool:
        """Validate LLM provider configuration."""
        if self.llm_provider == "groq" and not self.groq_api_key:
            return False
        elif self.llm_provider == "openai" and not self.openai_api_key:
            return False
        return True


# Global settings instance
settings = Settings()