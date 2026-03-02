"""
FastAPI application entry point for LLM API, with caching, database logging, and structured logging.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.endpoints import router
from core.config import get_settings
from core.logging_config import configure_logging, get_logger
from database import close_database, get_database_service, initialize_database
from models.schemas import ErrorResponse, HealthResponse, HealthStatus, WelcomeResponse
from services.cache_service import create_cache_service
from services.llm.ollama_provider import OllamaProvider

settings = get_settings()

# Configure structured logging
configure_logging(log_level="INFO", log_format=settings.log_format)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.

    Handles initialization and cleanup of resources.
    """
    # Startup
    logger.info("application_startup", version=settings.app_version)

    # Initialize cache service
    if settings.cache_enabled:
        cache_service = create_cache_service()
        app.state.cache_service = cache_service
        logger.info(
            "cache_service_initialized",
            backend=settings.cache_backend,
            max_size=settings.cache_max_size,
            ttl=settings.cache_ttl,
        )
    else:
        logger.info("cache_disabled")

    # Initialize database
    try:
        await initialize_database()
        app.state.db_service = get_database_service()
        logger.info("database_initialized", url=settings.database_url)
    except Exception as e:
        logger.error("database_initialization_failed", error=str(e))
        # Continue without database - non-critical for basic functionality

    # Initialize LLM provider
    llm_provider = OllamaProvider()
    app.state.llm_provider = llm_provider
    logger.info("llm_provider_initialized", provider="ollama")

    # Check Ollama health
    is_healthy = await llm_provider.health_check()
    if is_healthy:
        logger.info("ollama_health_check", status="healthy")
    else:
        logger.warning("ollama_health_check", status="unhealthy")

    logger.info("application_started", host=settings.host, port=settings.port)

    # Log registered routes
    logger.info("registered_routes")
    for route in app.routes:
        if hasattr(route, "methods"):
            logger.debug("route", method=list(route.methods)[0], path=route.path)

    yield

    # Shutdown
    logger.info("application_shutdown")
    
    # Cleanup LLM provider
    await llm_provider.close()
    logger.info("llm_provider_closed")
    
    # Cleanup database
    await close_database()
    logger.info("database_closed")
    
    logger.info("shutdown_complete")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="FastAPI application for LLM integration with Ollama",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
if settings.cors_enabled:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=settings.cors_methods_list,
        allow_headers=["*"],
    )
    logger.info("cors_enabled", origins=settings.cors_origins_list)


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    logger.error("unhandled_exception", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_server_error",
            message="An unexpected error occurred",
            detail=str(exc) if settings.debug else None,
        ).model_dump(),
    )


# Root endpoint
@app.get(
    "/",
    response_model=WelcomeResponse,
    summary="Welcome message",
    description="Get welcome message and API information",
)
async def root() -> WelcomeResponse:
    """
    Root endpoint with welcome message.

    Returns:
        WelcomeResponse: Welcome message and available endpoints
    """
    return WelcomeResponse(
        message="Welcome to LLM API",
        version=settings.app_version,
        endpoints=[
            "/docs - Interactive API documentation",
            "/health - Health check with detailed status",
            "/health/ready - Kubernetes readiness probe",
            "/health/live - Kubernetes liveness probe",
            "/api/v1/generate - Text generation with caching",
            "/api/v1/summarize - Document summarization with caching",
        ],
    )


# Health check endpoint
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health status of the API and its dependencies",
)
async def health_check(request: Request) -> HealthResponse:
    """
    Health check endpoint.

    Args:
        request: FastAPI request object

    Returns:
        HealthResponse: Health status of the API and services
    """
    services = {"api": "healthy"}

    # Check Ollama
    llm_provider = request.app.state.llm_provider
    ollama_healthy = await llm_provider.health_check()
    services["ollama"] = "healthy" if ollama_healthy else "unhealthy"

    # Check database
    db_service = getattr(request.app.state, "db_service", None)
    if db_service:
        db_healthy = await db_service.health_check()
        services["database"] = "healthy" if db_healthy else "unhealthy"

    # Check cache
    cache_service = getattr(request.app.state, "cache_service", None)
    if cache_service:
        services["cache"] = "healthy"
        cache_stats = await cache_service.get_stats()
        services["cache_stats"] = cache_stats

    # Determine overall status
    if all(
        status == "healthy"
        for key, status in services.items()
        if key not in ["cache_stats"]
    ):
        overall_status = HealthStatus.HEALTHY
    elif services.get("ollama") == "unhealthy":
        overall_status = HealthStatus.DEGRADED
    else:
        overall_status = HealthStatus.UNHEALTHY

    return HealthResponse(
        status=overall_status,
        services=services,
    )


# Kubernetes readiness probe
@app.get(
    "/health/ready",
    summary="Readiness check",
    description="Kubernetes readiness probe - checks if app can serve traffic",
)
async def readiness_check(request: Request):
    """
    Readiness check for Kubernetes.

    Returns 200 if ready to serve traffic, 503 otherwise.
    """
    llm_provider = request.app.state.llm_provider
    ollama_healthy = await llm_provider.health_check()
    
    db_service = getattr(request.app.state, "db_service", None)
    db_healthy = True
    if db_service:
        db_healthy = await db_service.health_check()

    if ollama_healthy and db_healthy:
        return JSONResponse(
            status_code=200,
            content={"status": "ready", "ollama": "healthy", "database": "healthy"},
        )
    else:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "ollama": "healthy" if ollama_healthy else "unhealthy",
                "database": "healthy" if db_healthy else "unhealthy",
            },
        )


# Kubernetes liveness probe
@app.get(
    "/health/live",
    summary="Liveness check",
    description="Kubernetes liveness probe - checks if app is alive",
)
async def liveness_check():
    """
    Liveness check for Kubernetes.

    Returns 200 if application is alive.
    """
    return JSONResponse(
        status_code=200,
        content={"status": "alive"},
    )


# Include API router
app.include_router(router)
