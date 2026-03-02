"""
Pydantic schemas for request/response validation.

Simplified DTOs for generate and summarize endpoints only.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# Enums
class HealthStatus(str, Enum):
    """Health status enumeration."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


class SummaryLength(str, Enum):
    """Summary length options."""

    SHORT = "short"  # 2-3 sentences
    MEDIUM = "medium"  # 1 paragraph
    LONG = "long"  # 2-3 paragraphs


# Request Models
class GenerateRequest(BaseModel):
    """Request model for text generation."""

    prompt: str = Field(
        ...,
        description="Text prompt for generation",
        min_length=1,
        max_length=4000,
        examples=["Explain quantum computing in simple terms"],
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum tokens to generate (uses default if not specified)",
        gt=0,
        le=2000,
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Temperature for randomness (0.0-2.0, uses default if not specified)",
        ge=0.0,
        le=2.0,
    )
    model: Optional[str] = Field(
        default=None,
        description="Specific model to use (uses default if not specified)",
        examples=["gemma3:270m", "llama3.2:latest"],
    )

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        """Validate prompt is not empty or whitespace."""
        if not v.strip():
            raise ValueError("Prompt cannot be empty or whitespace")
        return v.strip()


class SummarizeRequest(BaseModel):
    """Request model for document summarization."""

    text: str = Field(
        ...,
        description="Text to summarize",
        min_length=50,
        max_length=10000,
        examples=["Long article or document text here..."],
    )
    summary_length: SummaryLength = Field(
        default=SummaryLength.MEDIUM,
        description="Desired summary length",
    )
    focus_points: Optional[List[str]] = Field(
        default=None,
        description="Specific points to focus on in summary",
        max_length=5,
        examples=[["key findings", "recommendations"]],
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Validate text is not empty or whitespace."""
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace")
        return v.strip()


# Response Models
class GenerateResponse(BaseModel):
    """Response model for text generation."""

    generated_text: str = Field(
        ...,
        description="Generated text response",
    )
    prompt: str = Field(
        ...,
        description="Original prompt",
    )
    model: str = Field(
        ...,
        description="Model used for generation",
    )
    tokens_used: Optional[int] = Field(
        default=None,
        description="Estimated tokens used",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp",
    )


class SummarizeResponse(BaseModel):
    """Response model for document summarization."""

    summary: str = Field(
        ...,
        description="Generated summary",
    )
    summary_length: SummaryLength = Field(
        ...,
        description="Length category of summary",
    )
    original_length: int = Field(
        ...,
        description="Character count of original text",
    )
    summary_length_chars: int = Field(
        ...,
        description="Character count of summary",
    )
    compression_ratio: float = Field(
        ...,
        description="Compression ratio (original/summary)",
    )
    key_points: List[str] = Field(
        default_factory=list,
        description="Extracted key points",
    )
    model: str = Field(
        ...,
        description="Model used for summarization",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp",
    )


# Health Check Models
class HealthResponse(BaseModel):
    """Response model for health check."""

    status: HealthStatus = Field(
        ...,
        description="Overall health status",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Health check timestamp",
    )
    services: dict = Field(
        default_factory=dict,
        description="Status of individual services",
        examples=[{"ollama": "healthy", "api": "healthy"}],
    )


# Welcome Response
class WelcomeResponse(BaseModel):
    """Response model for root endpoint."""

    message: str = Field(
        ...,
        description="Welcome message",
    )
    version: str = Field(
        ...,
        description="API version",
    )
    endpoints: List[str] = Field(
        ...,
        description="Available endpoints",
    )


# Error Response
class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(
        ...,
        description="Error type",
    )
    message: str = Field(
        ...,
        description="Error message",
    )
    detail: Optional[str] = Field(
        default=None,
        description="Additional error details",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error timestamp",
    )
