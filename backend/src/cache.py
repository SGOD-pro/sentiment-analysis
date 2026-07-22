"""
Redis cache — graceful, never breaks.

Purpose: Optional cache layer for expensive queries. If Redis is unavailable
         or errors, logs a warning and returns None (cache miss). The app
         continues to work normally without cache.
Input: Cache key (str), value (dict).
Output: Cached dict or None on miss/error.
Dependencies: redis, config, logger
"""

import json

from config import get_settings
from logger import get_logger

log = get_logger(__name__)

_client = None
_unavailable = False  # avoid log spam after first failure


def get_redis():
    """Return a Redis client, or None if unavailable."""
    global _client, _unavailable
    if _unavailable:
        return None
    if _client is not None:
        return _client
    try:
        import redis
        settings = get_settings()
        _client = redis.from_url(settings.redis_url, decode_responses=True, socket_connect_timeout=2)
        _client.ping()
        log.info("Redis connected", extra={"url": settings.redis_url})
        return _client
    except Exception:
        _unavailable = True
        log.warning("redis connection not available — cache disabled, app continues normally")
        return None


def cache_get(key: str) -> dict | None:
    """Try to read from cache. Returns None on miss or any error."""
    r = get_redis()
    if r is None:
        return None
    try:
        raw = r.get(key)
        if raw is None:
            return None
        return json.loads(raw)
    except Exception:
        log.warning("redis cache_get error — treating as miss", extra={"key": key[:80]})
        return None


def cache_set(key: str, value: dict) -> None:
    """Try to write to cache. Silently skips on any error."""
    r = get_redis()
    if r is None:
        return
    try:
        r.set(key, json.dumps(value, default=str))
    except Exception:
        log.warning("redis cache_set error — skipping", extra={"key": key[:80]})


def cache_delete_prefix(prefix: str) -> None:
    """Delete all keys matching prefix*. Silently skips on any error."""
    r = get_redis()
    if r is None:
        return
    try:
        keys = r.keys(f"{prefix}*")
        if keys:
            r.delete(*keys)
    except Exception:
        log.warning("redis cache_delete_prefix error — skipping", extra={"prefix": prefix[:80]})
