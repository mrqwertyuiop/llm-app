"""
Configuration management for LLM API.

Simplified configuration using Pydantic Settings with essential parameters only.
"""

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="LLM API", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    environment: str = Field(default="development", description="Environment")

    # Server
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")

    # Ollama LLM Provider
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL",
    )
    ollama_model: str = Field(
        default="gemma3:270m",
        description="Default Ollama model",
    )
    ollama_timeout: int = Field(
        default=60,
        description="Ollama API timeout in seconds",
    )

    # LLM Parameters
    default_temperature: float = Field(
        default=0.7,
        description="Default temperature for generation",
        ge=0.0,
        le=2.0,
    )
    default_max_tokens: int = Field(
        default=500,
        description="Default max tokens",
        gt=0,
    )

    # CORS
    cors_enabled: bool = Field(default=True, description="Enable CORS")
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:8080",
        description="Allowed CORS origins (comma-separated)",
    )
    cors_methods: str = Field(
        default="GET,POST",
        description="Allowed CORS methods (comma-separated)",
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="json",
        description="Log format: json or console",
    )

    # Cache Configuration
    cache_enabled: bool = Field(default=True, description="Enable caching")
    cache_backend: str = Field(
        default="memory",
        description="Cache backend: memory or redis",
    )
    cache_max_size: int = Field(
        default=1000,
        description="Max cache entries (for in-memory)",
    )
    cache_ttl: int = Field(
        default=3600,
        description="Cache TTL in seconds",
    )
    cache_min_temperature: float = Field(
        default=0.5,
        description="Only cache responses with temperature <= this value",
    )

    # Database Configuration
    database_url: str = Field(
        default="sqlite+aiosqlite:///./llm_api.db",
        description="Database URL (SQLite or PostgreSQL)",
    )
    database_echo: bool = Field(
        default=False,
        description="Echo SQL queries (for debugging)",
    )

    # Retry Configuration
    retry_enabled: bool = Field(default=True, description="Enable retry logic")
    retry_max_attempts: int = Field(
        default=3,
        description="Max retry attempts",
    )
    retry_min_wait: int = Field(
        default=1,
        description="Min wait between retries (seconds)",
    )
    retry_max_wait: int = Field(
        default=10,
        description="Max wait between retries (seconds)",
    )

    @field_validator("default_temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is between 0 and 2."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins into a list."""
        if not self.cors_origins:
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",")]

    @property
    def cors_methods_list(self) -> List[str]:
        """Parse CORS methods into a list."""
        if not self.cors_methods:
            return ["*"]
        return [method.strip() for method in self.cors_methods.split(",")]

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment.lower() in ("development", "dev", "local")

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment.lower() in ("production", "prod")


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Returns:
        Settings: Application settings
    """
    return Settings()
