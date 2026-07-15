"""
Structured JSON logger.

Purpose: Provide structured logging throughout the backend. No print() anywhere.
Input: Module name for logger creation.
Output: stdlib Logger instance with JSON-formatted output.
Dependencies: stdlib logging, json
Example:
    from logger import get_logger
    log = get_logger(__name__)
    log.info("batch started", extra={"batch_id": "abc-123", "total": 500})
"""

import json
import logging
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        # Merge any extra keys passed via extra={}
        for key in record.__dict__:
            if key not in logging.LogRecord(
                "", 0, "", 0, "", (), None
            ).__dict__ and key not in ("message", "msg"):
                entry[key] = record.__dict__[key]
        if record.exc_info and record.exc_info[0]:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


_formatter = JSONFormatter()
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(_formatter)


def get_logger(name: str) -> logging.Logger:
    """Return a logger with structured JSON output."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(_handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
    return logger
