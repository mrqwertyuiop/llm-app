"""
Base LLM provider interface.

Defines the abstract base class for LLM providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    """Standard response format from LLM providers."""

    text: str
    model: str
    tokens_used: Optional[int] = None


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""

    pass


class ModelNotFoundError(LLMProviderError):
    """Raised when the specified model is not found."""

    pass


class TimeoutError(LLMProviderError):
    """Raised when the LLM request times out."""

    pass


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate text based on a prompt.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for randomness (0.0-2.0)
            model: Specific model to use

        Returns:
            LLMResponse: Generated text and metadata

        Raises:
            LLMProviderError: If generation fails
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the LLM provider is healthy and accessible.

        Returns:
            bool: True if healthy, False otherwise
        """
        pass
