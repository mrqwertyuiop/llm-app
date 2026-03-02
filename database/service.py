"""
Database service for managing connections and request logging.

Provides connection pooling, session management, and request logging functionality.
"""

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import Integer, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.config import get_settings
from core.logging_config import get_logger
from database.models import Base, RequestLog

logger = get_logger(__name__)


class DatabaseService:
    """Service for managing database connections and operations."""

    def __init__(self):
        """Initialize database service with configuration."""
        self.settings = get_settings()
        self._engine = None
        self._session_factory = None
        self._initialized = False

    async def initialize(self):
        """Initialize database engine and create tables."""
        if self._initialized:
            logger.warning("database_already_initialized")
            return

        try:
            logger.info("initializing_database", url=self.settings.database_url)
            
            # Create async engine
            self._engine = create_async_engine(
                self.settings.database_url,
                echo=self.settings.database_echo,
                future=True,
            )

            # Create session factory
            self._session_factory = async_sessionmaker(
                self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )

            # Create tables
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            self._initialized = True
            logger.info("database_initialized")

        except Exception as e:
            logger.error("database_initialization_failed", error=str(e))
            raise

    async def close(self):
        """Close database connections."""
        if self._engine:
            logger.info("closing_database_connections")
            await self._engine.dispose()
            self._initialized = False
            logger.info("database_connections_closed")

    @asynccontextmanager
    async def get_session(self):
        """
        Get a database session.

        Yields:
            AsyncSession: Database session

        Example:
            async with db_service.get_session() as session:
                # Use session
                pass
        """
        if not self._initialized:
            raise RuntimeError("Database service not initialized")

        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def log_request(
        self,
        endpoint: str,
        method: str,
        prompt: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
        model: str,
        generated_text: Optional[str],
        tokens_used: Optional[int],
        latency_ms: float,
        cache_hit: bool,
        status_code: int,
        error_message: Optional[str] = None,
    ) -> RequestLog:
        """
        Log an API request to the database.

        Args:
            endpoint: API endpoint path
            method: HTTP method
            prompt: Input prompt text
            temperature: Temperature parameter
            max_tokens: Max tokens parameter
            model: Model used
            generated_text: Generated response text
            tokens_used: Tokens consumed
            latency_ms: Request latency in milliseconds
            cache_hit: Whether response was from cache
            status_code: HTTP status code
            error_message: Error message if failed

        Returns:
            RequestLog: The created log entry
        """
        try:
            log_entry = RequestLog(
                endpoint=endpoint,
                method=method,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                model=model,
                generated_text=generated_text,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                cache_hit=cache_hit,
                status_code=status_code,
                error_message=error_message,
            )

            async with self.get_session() as session:
                session.add(log_entry)
                await session.flush()
                await session.refresh(log_entry)

            logger.debug(
                "request_logged",
                endpoint=endpoint,
                status=status_code,
                cache_hit=cache_hit,
                latency_ms=latency_ms,
            )

            return log_entry

        except Exception as e:
            logger.error("request_logging_failed", error=str(e))
            raise

    async def get_request_stats(
        self, limit: int = 100, offset: int = 0
    ) -> Dict[str, any]:
        """
        Get request statistics from logs.

        Args:
            limit: Maximum number of logs to retrieve
            offset: Offset for pagination

        Returns:
            Dict containing statistics and recent logs
        """
        try:
            async with self.get_session() as session:
                # Get total count
                count_stmt = select(func.count(RequestLog.id))
                total_count = await session.scalar(count_stmt)

                # Get recent logs
                logs_stmt = (
                    select(RequestLog)
                    .order_by(RequestLog.timestamp.desc())
                    .limit(limit)
                    .offset(offset)
                )
                result = await session.execute(logs_stmt)
                logs = result.scalars().all()

                # Calculate statistics
                stats_stmt = select(
                    func.count(RequestLog.id).label("total_requests"),
                    func.avg(RequestLog.latency_ms).label("avg_latency_ms"),
                    func.sum(
                        func.cast(RequestLog.cache_hit, Integer)
                    ).label("cache_hits"),
                    func.sum(RequestLog.tokens_used).label("total_tokens"),
                )
                stats_result = await session.execute(stats_stmt)
                stats = stats_result.one()

                cache_hit_rate = (
                    (stats.cache_hits / stats.total_requests * 100)
                    if stats.total_requests > 0
                    else 0
                )

                return {
                    "total_requests": total_count,
                    "avg_latency_ms": float(stats.avg_latency_ms or 0),
                    "cache_hit_rate": float(cache_hit_rate),
                    "total_tokens_used": int(stats.total_tokens or 0),
                    "recent_logs": [log.to_dict() for log in logs],
                    "limit": limit,
                    "offset": offset,
                }

        except Exception as e:
            logger.error("get_stats_failed", error=str(e))
            raise

    async def health_check(self) -> bool:
        """
        Check database connectivity.

        Returns:
            bool: True if database is accessible
        """
        try:
            if not self._initialized:
                return False

            async with self.get_session() as session:
                await session.execute(select(1))
                return True

        except Exception as e:
            logger.error("database_health_check_failed", error=str(e))
            return False


# Global database service instance
_db_service: Optional[DatabaseService] = None


def get_database_service() -> DatabaseService:
    """
    Get the global database service instance.

    Returns:
        DatabaseService: The database service instance

    Raises:
        RuntimeError: If database service not initialized
    """
    global _db_service
    if _db_service is None:
        _db_service = DatabaseService()
    return _db_service


async def initialize_database():
    """Initialize the global database service."""
    db_service = get_database_service()
    await db_service.initialize()


async def close_database():
    """Close the global database service."""
    global _db_service
    if _db_service:
        await _db_service.close()
        _db_service = None


async def get_all_logs_safe(limit: Optional[int] = None, offset: int = 0):
    """
    Safely get all logs with auto-initialization.
    
    Use this in standalone scripts or when database might not be initialized.
    
    Args:
        limit: Maximum number of logs to retrieve (None = all)
        offset: Offset for pagination
        
    Returns:
        List of log dictionaries
    """
    db_service = get_database_service()
    
    # Auto-initialize if needed
    if not db_service._initialized:
        await db_service.initialize()
    
    async with db_service.get_session() as session:
        stmt = select(RequestLog).order_by(RequestLog.timestamp.desc())
        
        if limit is not None:
            stmt = stmt.limit(limit)
        
        if offset > 0:
            stmt = stmt.offset(offset)
        
        result = await session.execute(stmt)
        logs = result.scalars().all()
        
        return [log.to_dict() for log in logs]
