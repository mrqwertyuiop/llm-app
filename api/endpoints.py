"""
API endpoints for LLM operations.

Simplified endpoints for generate and summarize operations with caching and logging.
"""

import time
from typing import Dict

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from core.logging_config import get_logger
from models.schemas import (
    GenerateRequest,
    GenerateResponse,
    SummarizeRequest,
    SummarizeResponse,
    SummaryLength,
)
from services.llm.base import LLMProviderError, ModelNotFoundError, TimeoutError

logger = get_logger(__name__)

# Create API router
router = APIRouter(prefix="/api/v1", tags=["LLM Operations"])


@router.post(
    "/generate",
    response_model=GenerateResponse,
    summary="Generate text from a prompt",
    description="Generate text using the configured LLM based on a prompt",
)
async def generate_text(
    request_data: GenerateRequest,
    request: Request,
) -> GenerateResponse:
    """
    Generate text based on a prompt.

    Args:
        request_data: Generation request parameters
        request: FastAPI request object

    Returns:
        GenerateResponse: Generated text and metadata

    Raises:
        HTTPException: If generation fails
    """
    llm_provider = request.app.state.llm_provider
    cache_service = getattr(request.app.state, "cache_service", None)
    db_service = getattr(request.app.state, "db_service", None)

    start_time = time.time()
    cache_hit = False
    generated_text = None
    tokens_used = None
    model_used = request_data.model or "default"
    status_code = 200
    error_message = None

    try:
        logger.info(
            "generate_request",
            prompt_length=len(request_data.prompt),
            temperature=request_data.temperature,
            max_tokens=request_data.max_tokens,
        )

        # Try to get from cache if enabled
        if cache_service:
            cached_response = await cache_service.get_cached_response(
                endpoint="/generate",
                prompt=request_data.prompt,
                temperature=request_data.temperature,
                max_tokens=request_data.max_tokens,
                model=request_data.model,
            )

            if cached_response:
                cache_hit = True
                generated_text = cached_response["generated_text"]
                model_used = cached_response["model"]
                tokens_used = cached_response.get("tokens_used")
                
                logger.info("cache_hit", endpoint="/generate")

                latency_ms = (time.time() - start_time) * 1000

                # Log to database
                if db_service:
                    await db_service.log_request(
                        endpoint="/generate",
                        method="POST",
                        prompt=request_data.prompt,
                        temperature=request_data.temperature,
                        max_tokens=request_data.max_tokens,
                        model=model_used,
                        generated_text=generated_text,
                        tokens_used=tokens_used,
                        latency_ms=latency_ms,
                        cache_hit=True,
                        status_code=status_code,
                    )

                return GenerateResponse(
                    generated_text=generated_text,
                    prompt=request_data.prompt,
                    model=model_used,
                    tokens_used=tokens_used,
                )

        # Call LLM provider
        response = await llm_provider.generate(
            prompt=request_data.prompt,
            max_tokens=request_data.max_tokens,
            temperature=request_data.temperature,
            model=request_data.model,
        )

        generated_text = response.text
        model_used = response.model
        tokens_used = response.tokens_used

        # Cache the response if enabled
        if cache_service:
            await cache_service.cache_response(
                endpoint="/generate",
                prompt=request_data.prompt,
                temperature=request_data.temperature,
                max_tokens=request_data.max_tokens,
                model=request_data.model,
                response={
                    "generated_text": generated_text,
                    "model": model_used,
                    "tokens_used": tokens_used,
                },
            )

        latency_ms = (time.time() - start_time) * 1000

        logger.info(
            "generate_success",
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            cache_hit=cache_hit,
        )

        # Log to database
        if db_service:
            await db_service.log_request(
                endpoint="/generate",
                method="POST",
                prompt=request_data.prompt,
                temperature=request_data.temperature,
                max_tokens=request_data.max_tokens,
                model=model_used,
                generated_text=generated_text,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                cache_hit=cache_hit,
                status_code=status_code,
            )

        return GenerateResponse(
            generated_text=generated_text,
            prompt=request_data.prompt,
            model=model_used,
            tokens_used=tokens_used,
        )

    except ModelNotFoundError as e:
        status_code = 404
        error_message = str(e)
        logger.error("model_not_found", error=error_message)
        
        latency_ms = (time.time() - start_time) * 1000
        if db_service:
            await db_service.log_request(
                endpoint="/generate",
                method="POST",
                prompt=request_data.prompt,
                temperature=request_data.temperature,
                max_tokens=request_data.max_tokens,
                model=model_used,
                generated_text=None,
                tokens_used=None,
                latency_ms=latency_ms,
                cache_hit=False,
                status_code=status_code,
                error_message=error_message,
            )
        
        raise HTTPException(status_code=status_code, detail=error_message)

    except TimeoutError as e:
        status_code = 504
        error_message = str(e)
        logger.error("request_timeout", error=error_message)
        
        latency_ms = (time.time() - start_time) * 1000
        if db_service:
            await db_service.log_request(
                endpoint="/generate",
                method="POST",
                prompt=request_data.prompt,
                temperature=request_data.temperature,
                max_tokens=request_data.max_tokens,
                model=model_used,
                generated_text=None,
                tokens_used=None,
                latency_ms=latency_ms,
                cache_hit=False,
                status_code=status_code,
                error_message=error_message,
            )
        
        raise HTTPException(status_code=status_code, detail=error_message)

    except LLMProviderError as e:
        status_code = 500
        error_message = f"Generation failed: {str(e)}"
        logger.error("llm_provider_error", error=str(e))
        
        latency_ms = (time.time() - start_time) * 1000
        if db_service:
            await db_service.log_request(
                endpoint="/generate",
                method="POST",
                prompt=request_data.prompt,
                temperature=request_data.temperature,
                max_tokens=request_data.max_tokens,
                model=model_used,
                generated_text=None,
                tokens_used=None,
                latency_ms=latency_ms,
                cache_hit=False,
                status_code=status_code,
                error_message=error_message,
            )
        
        raise HTTPException(status_code=status_code, detail=error_message)

    except Exception as e:
        status_code = 500
        error_message = "Internal server error"
        logger.error("unexpected_error", error=str(e))
        
        latency_ms = (time.time() - start_time) * 1000
        if db_service:
            await db_service.log_request(
                endpoint="/generate",
                method="POST",
                prompt=request_data.prompt,
                temperature=request_data.temperature,
                max_tokens=request_data.max_tokens,
                model=model_used,
                generated_text=None,
                tokens_used=None,
                latency_ms=latency_ms,
                cache_hit=False,
                status_code=status_code,
                error_message=str(e),
            )
        
        raise HTTPException(status_code=status_code, detail=error_message)


@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    summary="Summarize a document",
    description="Generate a summary of provided text with configurable length",
)
async def summarize_document(
    request_data: SummarizeRequest,
    request: Request,
) -> SummarizeResponse:
    """
    Summarize a document.

    Args:
        request_data: Summarization request parameters
        request: FastAPI request object

    Returns:
        SummarizeResponse: Summary and metadata

    Raises:
        HTTPException: If summarization fails
    """
    llm_provider = request.app.state.llm_provider
    cache_service = getattr(request.app.state, "cache_service", None)
    db_service = getattr(request.app.state, "db_service", None)

    start_time = time.time()
    cache_hit = False
    summary_text = None
    model_used = "default"
    status_code = 200
    error_message = None

    try:
        logger.info(
            "summarize_request",
            text_length=len(request_data.text),
            summary_length=request_data.summary_length.value,
        )

        # Build prompt based on summary length
        length_instructions = {
            SummaryLength.SHORT: "in 2-3 concise sentences",
            SummaryLength.MEDIUM: "in one well-structured paragraph",
            SummaryLength.LONG: "in 2-3 detailed paragraphs",
        }

        focus_instruction = ""
        if request_data.focus_points:
            focus_points_str = ", ".join(request_data.focus_points)
            focus_instruction = f"\nFocus on these key aspects: {focus_points_str}."

        prompt = f"""Summarize the following text {length_instructions[request_data.summary_length]}.{focus_instruction}
                    Text to summarize:
                    {request_data.text}

                    Summary:"""

        # Try to get from cache if enabled
        if cache_service:
            cached_response = await cache_service.get_cached_response(
                endpoint="/summarize",
                prompt=prompt,
                temperature=0.3,
                max_tokens=800,
                model=None,
            )

            if cached_response:
                cache_hit = True
                summary_text = cached_response["summary"]
                model_used = cached_response["model"]
                
                logger.info("cache_hit", endpoint="/summarize")

                # Calculate metrics from cached response
                original_length = len(request_data.text)
                summary_length_chars = len(summary_text)
                compression_ratio = cached_response["compression_ratio"]
                key_points = cached_response["key_points"]

                latency_ms = (time.time() - start_time) * 1000

                # Log to database
                if db_service:
                    await db_service.log_request(
                        endpoint="/summarize",
                        method="POST",
                        prompt=prompt,
                        temperature=0.3,
                        max_tokens=800,
                        model=model_used,
                        generated_text=summary_text,
                        tokens_used=cached_response.get("tokens_used"),
                        latency_ms=latency_ms,
                        cache_hit=True,
                        status_code=status_code,
                    )

                return SummarizeResponse(
                    summary=summary_text,
                    summary_length=request_data.summary_length,
                    original_length=original_length,
                    summary_length_chars=summary_length_chars,
                    compression_ratio=compression_ratio,
                    key_points=key_points,
                    model=model_used,
                )

        # Call LLM provider
        response = await llm_provider.generate(
            prompt=prompt,
            max_tokens=800,
            temperature=0.3,  # Lower temperature for more focused summaries
        )

        # Extract key points (simplified - extract first sentence of each paragraph)
        summary_text = response.text.strip()
        model_used = response.model
        key_points = []
        paragraphs = summary_text.split("\n\n")
        for para in paragraphs[:3]:  # Max 3 key points
            if para.strip():
                first_sentence = para.split(".")[0].strip()
                if first_sentence:
                    key_points.append(first_sentence + ".")

        # Calculate compression ratio
        original_length = len(request_data.text)
        summary_length_chars = len(summary_text)
        compression_ratio = (
            round(original_length / summary_length_chars, 2)
            if summary_length_chars > 0
            else 0.0
        )

        # Cache the response if enabled
        if cache_service:
            await cache_service.cache_response(
                endpoint="/summarize",
                prompt=prompt,
                temperature=0.3,
                max_tokens=800,
                model=None,
                response={
                    "summary": summary_text,
                    "model": model_used,
                    "compression_ratio": compression_ratio,
                    "key_points": key_points,
                    "tokens_used": response.tokens_used,
                },
            )

        latency_ms = (time.time() - start_time) * 1000

        logger.info(
            "summarize_success",
            compression_ratio=compression_ratio,
            latency_ms=latency_ms,
            cache_hit=cache_hit,
        )

        # Log to database
        if db_service:
            await db_service.log_request(
                endpoint="/summarize",
                method="POST",
                prompt=prompt,
                temperature=0.3,
                max_tokens=800,
                model=model_used,
                generated_text=summary_text,
                tokens_used=response.tokens_used,
                latency_ms=latency_ms,
                cache_hit=cache_hit,
                status_code=status_code,
            )

        return SummarizeResponse(
            summary=summary_text,
            summary_length=request_data.summary_length,
            original_length=original_length,
            summary_length_chars=summary_length_chars,
            compression_ratio=compression_ratio,
            key_points=key_points,
            model=model_used,
        )

    except ModelNotFoundError as e:
        status_code = 404
        error_message = str(e)
        logger.error("model_not_found", error=error_message)
        
        latency_ms = (time.time() - start_time) * 1000
        if db_service:
            await db_service.log_request(
                endpoint="/summarize",
                method="POST",
                prompt=request_data.text[:500],
                temperature=0.3,
                max_tokens=800,
                model=model_used,
                generated_text=None,
                tokens_used=None,
                latency_ms=latency_ms,
                cache_hit=False,
                status_code=status_code,
                error_message=error_message,
            )
        
        raise HTTPException(status_code=status_code, detail=error_message)

    except TimeoutError as e:
        status_code = 504
        error_message = str(e)
        logger.error("request_timeout", error=error_message)
        
        latency_ms = (time.time() - start_time) * 1000
        if db_service:
            await db_service.log_request(
                endpoint="/summarize",
                method="POST",
                prompt=request_data.text[:500],
                temperature=0.3,
                max_tokens=800,
                model=model_used,
                generated_text=None,
                tokens_used=None,
                latency_ms=latency_ms,
                cache_hit=False,
                status_code=status_code,
                error_message=error_message,
            )
        
        raise HTTPException(status_code=status_code, detail=error_message)

    except LLMProviderError as e:
        status_code = 500
        error_message = f"Summarization failed: {str(e)}"
        logger.error("llm_provider_error", error=str(e))
        
        latency_ms = (time.time() - start_time) * 1000
        if db_service:
            await db_service.log_request(
                endpoint="/summarize",
                method="POST",
                prompt=request_data.text[:500],
                temperature=0.3,
                max_tokens=800,
                model=model_used,
                generated_text=None,
                tokens_used=None,
                latency_ms=latency_ms,
                cache_hit=False,
                status_code=status_code,
                error_message=error_message,
            )
        
        raise HTTPException(status_code=status_code, detail=error_message)

    except Exception as e:
        status_code = 500
        error_message = "Internal server error"
        logger.error("unexpected_error", error=str(e))
        
        latency_ms = (time.time() - start_time) * 1000
        if db_service:
            await db_service.log_request(
                endpoint="/summarize",
                method="POST",
                prompt=request_data.text[:500],
                temperature=0.3,
                max_tokens=800,
                model=model_used,
                generated_text=None,
                tokens_used=None,
                latency_ms=latency_ms,
                cache_hit=False,
                status_code=status_code,
                error_message=str(e),
            )
        
        raise HTTPException(status_code=status_code, detail=error_message)
