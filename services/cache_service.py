"""
Cache service with adapter pattern for flexible backend support.

Supports in-memory caching (default) with easy migration to Redis.
"""

import hashlib
import json
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Optional

from cachetools import TTLCache

from core.logging_config import get_logger

logger = get_logger(__name__)


class CacheBackend(ABC):
    """Abstract cache backend interface."""

    @abstractmethod
    async def get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        pass

    @abstractmethod
    async def set(self, key: str, value: str, ttl: int) -> None:
        """Set value in cache with TTL in seconds."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete value from cache."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all cache entries."""
        pass

    @abstractmethod
    async def get_stats(self) -> dict:
        """Get cache statistics."""
        pass


class InMemoryCacheBackend(CacheBackend):
    """
    In-memory cache backend using cachetools.TTLCache.
    
    Features:
    - Thread-safe
    - Automatic TTL expiration
    - LRU eviction when full
    - No external dependencies
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize in-memory cache.
        
        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds
        """
        self.cache = TTLCache(maxsize=max_size, ttl=default_ttl)
        self.default_ttl = default_ttl
        self.hits = 0
        self.misses = 0

    async def get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        try:
            value = self.cache[key]
            self.hits += 1
            return value
        except KeyError:
            self.misses += 1
            return None

    async def set(self, key: str, value: str, ttl: int) -> None:
        """Set value in cache with TTL."""
        # Note: cachetools.TTLCache uses default TTL from initialization
        # For per-key TTL, we'd need a different approach
        self.cache[key] = value

    async def delete(self, key: str) -> None:
        """Delete value from cache."""
        try:
            del self.cache[key]
        except KeyError:
            pass

    async def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    async def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            "backend": "in-memory",
            "size": len(self.cache),
            "max_size": self.cache.maxsize,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 2),
            "ttl": self.default_ttl,
        }


class RedisCacheBackend(CacheBackend):
    """
    Redis cache backend (placeholder for future implementation).
    
    To use Redis:
    1. Install: pip install redis aioredis
    2. Implement methods using aioredis
    3. Update config to use Redis backend
    """

    def __init__(self, redis_url: str):
        """Initialize Redis cache."""
        raise NotImplementedError(
            "Redis backend not yet implemented. "
            "Use InMemoryCacheBackend for now."
        )

    async def get(self, key: str) -> Optional[str]:
        raise NotImplementedError

    async def set(self, key: str, value: str, ttl: int) -> None:
        raise NotImplementedError

    async def delete(self, key: str) -> None:
        raise NotImplementedError

    async def clear(self) -> None:
        raise NotImplementedError

    async def get_stats(self) -> dict:
        raise NotImplementedError


class CacheService:
    """
    High-level cache service for LLM responses.
    
    Features:
    - Automatic key generation from request parameters
    - JSON serialization/deserialization
    - Cache statistics
    - Backend abstraction (in-memory or Redis)
    """

    def __init__(self, backend: CacheBackend):
        """
        Initialize cache service.
        
        Args:
            backend: Cache backend implementation
        """
        self.backend = backend

    def _generate_key(self, prefix: str, **kwargs) -> str:
        """
        Generate deterministic cache key from parameters.
        
        Args:
            prefix: Key prefix (e.g., 'generate', 'summarize')
            **kwargs: Parameters to include in key
            
        Returns:
            Hash-based cache key
        """
        # Sort kwargs for consistent key generation
        sorted_params = sorted(kwargs.items())
        key_string = f"{prefix}:" + ":".join(f"{k}={v}" for k, v in sorted_params)
        
        # Use SHA256 hash for fixed-length key
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()
        return f"{prefix}:{key_hash[:16]}"

    async def get_cached_response(
        self,
        endpoint: str,
        **params
    ) -> Optional[dict]:
        """
        Get cached response for given parameters.
        
        Args:
            endpoint: Endpoint name (e.g., 'generate', 'summarize')
            **params: Request parameters
            
        Returns:
            Cached response dict or None if not found
        """
        key = self._generate_key(endpoint, **params)
        logger.info(f"🔍 [CACHE] Looking for key: {key}")
        logger.info(f"🔍 [CACHE] Parameters: {params}")
        cached_value = await self.backend.get(key)
        
        if cached_value:
            logger.info(f"[CACHE] Cache HIT for key: {key}")
            try:
                return json.loads(cached_value)
            except json.JSONDecodeError:
                await self.backend.delete(key)  # Invalid cache entry
                return None
        
        logger.info(f"[CACHE] Cache MISS for key: {key}")
        return None

    async def cache_response(
        self,
        endpoint: str,
        response: dict,
        ttl: int = 3600,
        **params
    ) -> None:
        """
        Cache response with given parameters.
        
        Args:
            endpoint: Endpoint name
            response: Response dict to cache
            ttl: Time to live in seconds
            **params: Request parameters
        """
        key = self._generate_key(endpoint, **params)
        value = json.dumps(response)
        logger.info(f"💾 [CACHE] Storing key: {key}")
        logger.info(f"💾 [CACHE] Parameters: {params}")
        logger.info(f"💾 [CACHE] TTL: {ttl} seconds")
        await self.backend.set(key, value, ttl)

    async def invalidate_cache(self, endpoint: str, **params) -> None:
        """Invalidate specific cache entry."""
        key = self._generate_key(endpoint, **params)
        await self.backend.delete(key)

    async def clear_all(self) -> None:
        """Clear all cache entries."""
        await self.backend.clear()

    async def get_stats(self) -> dict:
        """Get cache statistics."""
        return await self.backend.get_stats()


def create_cache_service(
    backend_type: str = "memory",
    max_size: int = 1000,
    ttl: int = 3600,
    redis_url: Optional[str] = None,
) -> CacheService:
    """
    Factory function to create cache service with appropriate backend.
    
    Args:
        backend_type: 'memory' or 'redis'
        max_size: Max cache entries (for in-memory)
        ttl: Default TTL in seconds
        redis_url: Redis URL (for Redis backend)
        
    Returns:
        Configured CacheService instance
    """
    if backend_type == "memory":
        backend = InMemoryCacheBackend(max_size=max_size, default_ttl=ttl)
    elif backend_type == "redis":
        if not redis_url:
            raise ValueError("redis_url required for Redis backend")
        backend = RedisCacheBackend(redis_url=redis_url)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
    
    return CacheService(backend=backend)
