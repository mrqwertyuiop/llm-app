"""
Database models for request logging.

Uses SQLAlchemy ORM for database abstraction (SQLite → PostgreSQL migration ready).
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class RequestLog(Base):
    """
    Request log model for tracking API usage and analytics.
    
    Schema Design:
    - Captures every API request/response
    - Enables usage analytics and debugging
    - Supports migration to PostgreSQL without code changes
    """

    __tablename__ = "request_logs"

    # Primary key
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    # Timestamp
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Request info
    endpoint = Column(String(50), nullable=False, index=True)  # /generate, /summarize
    method = Column(String(10), nullable=False)  # POST, GET
    
    # Request parameters
    prompt = Column(Text, nullable=True)  # Original prompt (can be large)
    temperature = Column(Float, nullable=True)
    max_tokens = Column(Integer, nullable=True)
    model = Column(String(50), nullable=True)

    # Response info
    generated_text = Column(Text, nullable=True)  # Generated response
    tokens_used = Column(Integer, nullable=True)
    
    # Performance metrics
    latency_ms = Column(Float, nullable=False)  # Request duration in ms
    cache_hit = Column(Integer, nullable=False, default=0)  # 1 if cached, 0 if generated
    
    # Status
    status_code = Column(Integer, nullable=False, index=True)  # HTTP status code
    error_message = Column(Text, nullable=True)  # Error details if failed

    def __repr__(self):
        return (
            f"<RequestLog(id={self.id}, endpoint={self.endpoint}, "
            f"status={self.status_code}, latency={self.latency_ms}ms)>"
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "endpoint": self.endpoint,
            "method": self.method,
            "prompt": self.prompt[:100] + "..." if self.prompt and len(self.prompt) > 100 else self.prompt,
            "model": self.model,
            "tokens_used": self.tokens_used,
            "latency_ms": self.latency_ms,
            "cache_hit": bool(self.cache_hit),
            "status_code": self.status_code,
        }
