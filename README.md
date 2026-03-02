# LLM API - FastAPI Application

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green.svg)](https://fastapi.tiangolo.com/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-orange.svg)](https://ollama.ai/)

A FastAPI application providing text generation and document summarization via local LLMs through Ollama, featuring caching, request logging, and retry mechanisms.

---

## 📑 Table of Contents

- [Quick Start](#quick-start)
- [Design Rationale](#design-rationale)
- [Main Dependencies](#main-dependencies)
- [API Examples](#api-examples)
- [Configuration](#configuration)

---

## Quick Start

### Prerequisites

- **Python 3.12+**
- **Ollama** - [Install Guide](https://ollama.ai/)

### Installation

```bash
# 1. Install and start Ollama
brew install ollama              # macOS
ollama serve                     # Start service
ollama pull gemma3:270m          # Pull model (270M params, fast)

# 2. Setup application
cd /path/to/llm-app
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp env.example.txt .env

# 3. Run application
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 4. Verify
curl http://localhost:8000/health

# Visit: http://localhost:8000/docs
```

---

## Design Rationale

### Architecture Flow

```
Client → FastAPI → Cache Check → [Hit: Return | Miss: Ollama] → DB Log → Response
```

### Key Design Principles

#### 1. **Layered Architecture**
```
main.py              → Application lifecycle
api/endpoints.py     → HTTP request/response
services/            → Business logic (cache, LLM)
database/            → Data persistence
models/schemas.py    → Validation
core/                → Configuration, logging
```

#### 2. **Hash-Based Caching**

Cache keys generated from all request parameters using SHA256:

```python
# Generate deterministic key
key = "generate:" + SHA256("prompt=Hello:temperature=0.7:max_tokens=500:model=gemma3:270m")[:16]
```

**Example**:
```
Request 1: prompt="Explain AI", temp=0.7 → Key: "generate:a1b2c3d4..." (OLLAMA)
Request 2: prompt="Explain AI", temp=0.7 → Key: "generate:a1b2c3d4..." (CACHE HIT!)
Request 3: prompt="Explain AI", temp=0.8 → Key: "generate:f9e8d7c6..." (OLLAMA - different key)
```

#### 2.1. **Cache Strategy Comparison: `/generate` vs `/summarize`**

**`/generate` - User Controls Everything:**
```python
# Users specify all parameters
get_cached_response(endpoint="/generate", prompt=user_input, 
                   temperature=user_temp, max_tokens=user_tokens, model=user_model)
```
- **Why**: Creative use case needs flexibility (try different temperatures/settings)
- **Result**: Lower hit rate (~5-20%) - many parameter variations

**`/summarize` - Service Controls Parameters:**
```python
# Service constructs prompt + fixes parameters
prompt = f"Summarize...{summary_length}...{text}"  # Includes length instructions
get_cached_response(endpoint="/summarize", prompt=prompt,
                   temperature=0.3, max_tokens=800, model=None)  # FIXED
```
- **Why**: Consistency matters - same text should produce same summary
- **Result**: Higher hit rate (~60-80%) - fewer variations, but different `summary_length` = different keys

**Quick Comparison:**
| | `/generate` | `/summarize` |
|---|---|---|
| **User Controls** | All params | Only text + length |
| **Temperature** | Variable | Fixed 0.3 |
| **Cache Key** | Raw prompt + params | Constructed prompt + fixed params |
| **Hit Rate** | Lower (flexibility) | Higher (consistency) |

#### 3. **Adapter Pattern for Cache**

```python
CacheBackend (Abstract)
├── InMemoryCacheBackend (Current: TTL + LRU)
└── RedisCacheBackend (Future migration path)
```

**Benefits**: Start simple, migrate to Redis without changing endpoint code.

#### 4. **Retry Logic with Exponential Backoff**

```python
# In ollama_provider.py - Dynamic retry decorator
if settings.retry_enabled:
    retry_decorator = retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
        stop=stop_after_attempt(settings.retry_max_attempts),  # Default: 3
        wait=wait_exponential(
            multiplier=1,
            min=settings.retry_min_wait,  # Default: 1 second
            max=settings.retry_max_wait,  # Default: 10 seconds
        ),
        reraise=True,
    )
    generate_fn = retry_decorator(self._generate_with_retry)
```

**Retry Sequence (with defaults):**
- Attempt 1: Call Ollama → Fails (timeout/error)
- Wait: 1 second (min wait)
- Attempt 2: Retry → Fails again
- Wait: 2 seconds (exponential: 1 × 2¹)
- Attempt 3: Retry → Fails again
- Wait: 4 seconds (exponential: 1 × 2²)
- Final: Raise exception (max attempts reached)

**Configuration:**
- `RETRY_ENABLED=true` - Enable/disable retry logic
- `RETRY_MAX_ATTEMPTS=3` - Maximum retry attempts
- `RETRY_MIN_WAIT=1` - Minimum wait time (seconds)
- `RETRY_MAX_WAIT=10` - Maximum wait time cap (seconds)

#### 5. **Structured Logging**

```python
logger.info("generate_request", prompt_length=150, temperature=0.7, max_tokens=500)
```

**JSON Output**:
```json
{"event":"generate_request","prompt_length":150,"temperature":0.7,"timestamp":"2025-12-12T10:30:00.123Z","level":"info"}
```

**Benefits**: Parseable logs for ELK/Datadog, rich debugging context.

#### 6. **Database Logging**

Every request logged to SQLite for analytics:
- Cache hit rate: `SUM(cache_hit) / COUNT(*)`
- Latency trends: `AVG(latency_ms) GROUP BY hour`
- Token usage tracking
- Audit trail

---

## Main Dependencies

| Package | Version | Purpose | Why Chosen |
|---------|---------|---------|------------|
| **FastAPI** | 0.109.0 | Web framework | Auto docs, Pydantic validation, async support |
| **Uvicorn** | 0.27.0 | ASGI server | Lightning fast, production-ready |
| **httpx** | 0.26.0 | HTTP client | Async support, connection pooling |
| **Ollama** | Local | LLM service | Free, local, no API keys, privacy |
| **Pydantic** | 2.5.3 | Validation | Auto-validation, type hints, JSON schema |
| **cachetools** | 5.3.2 | Caching | Pure Python, TTL+LRU, thread-safe |
| **SQLAlchemy** | 2.0.25 | ORM | Async support, type-safe queries |
| **aiosqlite** | 0.19.0 | SQLite driver | No separate DB server, file-based |
| **tenacity** | 8.2.3 | Retry logic | Declarative policies, exponential backoff |
| **structlog** | 24.1.0 | Logging | Key-value logs, JSON output |

**Architecture Choice**: Adapter pattern for cache (start in-memory, migrate to Redis), retry mechanism for resilience, structured logging for observability.

---

## API Examples

### 1. Generate Text

**Endpoint**: `POST /api/v1/generate`

**Basic Request**:
```bash
curl -X POST "http://localhost:8000/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum computing in simple terms"}'
```

**Response**:
```json
{
  "generated_text": "Quantum computing uses quantum mechanics principles...",
  "prompt": "Explain quantum computing in simple terms",
  "model": "gemma3:270m",
  "tokens_used": 67,
  "timestamp": "2025-12-12T10:30:00.123456Z"
}
```

**With Parameters**:
```bash
curl -X POST "http://localhost:8000/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a haiku about coding",
    "temperature": 0.9,
    "max_tokens": 100,
    "model": "gemma3:270m"
  }'
```

**Python Example**:
```python
import httpx

response = httpx.post(
    "http://localhost:8000/api/v1/generate",
    json={"prompt": "List 3 benefits of async programming", "temperature": 0.7},
    timeout=30.0
)
print(response.json()["generated_text"])
```

**JavaScript Example**:
```javascript
const response = await fetch('http://localhost:8000/api/v1/generate', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({prompt: 'Explain REST APIs', temperature: 0.7})
});
const data = await response.json();
console.log(data.generated_text);
```

---

### 2. Summarize Document

**Endpoint**: `POST /api/v1/summarize`

**Short Summary**:
```bash
curl -X POST "http://localhost:8000/api/v1/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Artificial intelligence (AI) has transformed many industries...",
    "summary_length": "short"
  }'
```

**Response**:
```json
{
  "summary": "AI has revolutionized industries through machine learning...",
  "summary_length": "short",
  "original_length": 653,
  "summary_length_chars": 215,
  "compression_ratio": 3.04,
  "key_points": ["AI has revolutionized industries...", "Ethical concerns must be addressed..."],
  "model": "gemma3:270m",
  "timestamp": "2025-12-12T10:32:00.789012Z"
}
```

**With Focus Points**:
```bash
curl -X POST "http://localhost:8000/api/v1/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The global climate crisis requires immediate action...",
    "summary_length": "medium",
    "focus_points": ["renewable energy", "policy actions"]
  }'
```

**Python Example**:
```python
import httpx

response = httpx.post(
    "http://localhost:8000/api/v1/summarize",
    json={
        "text": "Your long article text here...",
        "summary_length": "medium",
        "focus_points": ["key findings", "recommendations"]
    },
    timeout=30.0
)
result = response.json()
print(f"Summary: {result['summary']}")
print(f"Compression: {result['compression_ratio']}x")
```

---

### 3. Health Checks

**Comprehensive Health**:
```bash
curl http://localhost:8000/health
```

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-12-12T10:35:00.123456Z",
  "services": {
    "api": "healthy",
    "ollama": "healthy",
    "database": "healthy",
    "cache": "healthy"
  },
  "cache_stats": {
    "backend": "in-memory",
    "size": 42,
    "max_size": 1000,
    "hits": 1250,
    "misses": 500,
    "hit_rate": 71.43,
    "ttl": 3600
  }
}
```

**Kubernetes Probes**:
```bash
# Readiness (returns 503 if Ollama/DB unavailable)
curl http://localhost:8000/health/ready

# Liveness (basic ping)
curl http://localhost:8000/health/live
```

---

### 4. Error Responses

**Missing Required Field (422)**:
```json
{
  "detail": [{
    "type": "missing",
    "loc": ["body", "prompt"],
    "msg": "Field required"
  }]
}
```

**Invalid Parameter (422)**:
```json
{
  "detail": [{
    "type": "less_than_equal",
    "loc": ["body", "temperature"],
    "msg": "Input should be less than or equal to 2.0",
    "input": 3.0
  }]
}
```

**Model Not Found (404)**:
```json
{"detail": "Model 'nonexistent-model' not found"}
```

**Timeout (504)**:
```json
{"detail": "Request to Ollama timed out after 60s"}
```

---

## Configuration

### Environment Variables (.env)

```bash
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:270m
OLLAMA_TIMEOUT=60

# Cache Configuration
CACHE_ENABLED=true
CACHE_BACKEND=memory
CACHE_MAX_SIZE=1000
CACHE_TTL=3600
CACHE_MIN_TEMPERATURE=0.5        # Only cache if temp ≤ this

# Database
DATABASE_URL=sqlite+aiosqlite:///./llm_api.db

# Retry Configuration
RETRY_ENABLED=true
RETRY_MAX_ATTEMPTS=3
RETRY_MIN_WAIT=1
RETRY_MAX_WAIT=10

# Logging
LOG_FORMAT=json                   # json or console
LOG_LEVEL=INFO

# CORS
CORS_ENABLED=true
CORS_ORIGINS=http://localhost:3000,http://localhost:8080
```

### Configuration Profiles

**Development**:
```bash
DEBUG=true
LOG_FORMAT=console
DATABASE_ECHO=true
RETRY_ENABLED=false
```

**Production**:
```bash
DEBUG=false
LOG_FORMAT=json
RETRY_ENABLED=true
CACHE_MAX_SIZE=5000
CACHE_TTL=7200
```

---

## Monitoring & Observability

### View Logs

**Console (Development)**:
```bash
export LOG_FORMAT=console
uvicorn main:app --reload
```

**JSON (Production)**:
```bash
export LOG_FORMAT=json
uvicorn main:app
```

### Database Analytics

```bash
sqlite3 llm_api.db

# Recent requests
SELECT datetime(timestamp) as time, endpoint, 
       substr(prompt, 1, 50) as prompt_preview,
       tokens_used, ROUND(latency_ms, 2) as latency_ms,
       cache_hit, status_code
FROM request_log ORDER BY timestamp DESC LIMIT 10;

# Statistics
SELECT COUNT(*) as total_requests,
       ROUND(AVG(latency_ms), 2) as avg_latency_ms,
       ROUND(100.0 * SUM(cache_hit) / COUNT(*), 2) as cache_hit_rate,
       SUM(tokens_used) as total_tokens
FROM request_log;

# By endpoint
SELECT endpoint, COUNT(*) as count,
       ROUND(AVG(latency_ms), 2) as avg_latency,
       ROUND(100.0 * SUM(cache_hit) / COUNT(*), 2) as cache_rate
FROM request_log GROUP BY endpoint;
```

### Cache Statistics

```bash
curl http://localhost:8000/health | jq '.cache_stats'
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **Ollama unavailable** | `ollama serve` → `ollama pull gemma3:270m` |
| **Cache not working** | Check `CACHE_ENABLED=true` and `CACHE_MIN_TEMPERATURE` threshold |
| **Database errors** | Check permissions: `ls -la llm_api.db` or reset: `rm llm_api.db` |
| **High latency** | Use smaller model: `OLLAMA_MODEL=gemma3:270m` |

---

## Additional Resources

- **Interactive API Docs**: http://localhost:8000/docs
- **Ollama Documentation**: https://ollama.ai/
- **FastAPI Documentation**: https://fastapi.tiangolo.com/

