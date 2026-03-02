"""
Ollama LLM provider implementation.

Provides integration with Ollama local LLM service with retry logic.
"""

import json
from typing import Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core.config import get_settings
from core.logging_config import get_logger
from services.llm.base import (
    BaseLLMProvider,
    LLMProviderError,
    LLMResponse,
    ModelNotFoundError,
    TimeoutError,
)

logger = get_logger(__name__)


class OllamaProvider(BaseLLMProvider):
    """Ollama LLM provider implementation."""

    def __init__(self):
        """Initialize Ollama provider with configuration."""
        self.settings = get_settings()
        self.base_url = self.settings.ollama_base_url
        self.default_model = self.settings.ollama_model
        self.timeout = self.settings.ollama_timeout
        self.client = httpx.AsyncClient(timeout=self.timeout)

    async def _generate_with_retry(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
    ) -> LLMResponse:
        """
        Internal method to generate text with retry logic.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for randomness (0.0-2.0)
            model: Specific model to use

        Returns:
            LLMResponse: Generated text and metadata

        Raises:
            LLMProviderError: If generation fails after retries
        """
        model_name = model or self.default_model

        # Build options
        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        # Prepare request
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
        }
        if options:
            payload["options"] = options

        try:
            logger.info("generating_text", model=model_name, temperature=temperature)
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json=payload,
            )
            response.raise_for_status()

            result = response.json()
            generated_text = result.get("response", "")

            # Estimate tokens (rough approximation: ~4 chars per token)
            tokens_used = len(generated_text) // 4 if generated_text else None

            logger.info("generation_successful", tokens_used=tokens_used)
            return LLMResponse(
                text=generated_text,
                model=model_name,
                tokens_used=tokens_used,
            )

        except httpx.TimeoutException as e:
            logger.warning("ollama_timeout", error=str(e))
            raise TimeoutError(f"Request to Ollama timed out after {self.timeout}s")

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.error("model_not_found", model=model_name)
                raise ModelNotFoundError(f"Model '{model_name}' not found")
            logger.warning("http_error", status_code=e.response.status_code)
            raise LLMProviderError(f"HTTP error: {e.response.status_code}")

        except Exception as e:
            logger.error("generation_failed", error=str(e))
            raise LLMProviderError(f"Failed to generate text: {str(e)}")

    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate text using Ollama.

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
        if self.settings.retry_enabled:
            # Create a retry-decorated version of the internal method
            retry_decorator = retry(
                retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
                stop=stop_after_attempt(self.settings.retry_max_attempts),
                wait=wait_exponential(
                    multiplier=1,
                    min=self.settings.retry_min_wait,
                    max=self.settings.retry_max_wait,
                ),
                reraise=True,
            )
            generate_fn = retry_decorator(self._generate_with_retry)
            logger.info("retry_enabled", max_attempts=self.settings.retry_max_attempts)
        else:
            generate_fn = self._generate_with_retry
            logger.info("retry_disabled")

        return await generate_fn(prompt, max_tokens, temperature, model)

    async def health_check(self) -> bool:
        """
        Check if Ollama service is healthy.

        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
