from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # Environment
    environment: str = "development"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    
    # Google Cloud
    google_application_credentials: Optional[str] = None
    
    # OCR Settings
    max_image_size: int = 1500
    ocr_timeout: int = 30
    default_ocr_confidence_threshold: float = 0.6
    
    # NLP Settings
    fuzzy_match_threshold: float = 0.6
    min_brand_confidence: float = 0.5
    
    # Redis (optional)
    redis_url: Optional[str] = None
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()