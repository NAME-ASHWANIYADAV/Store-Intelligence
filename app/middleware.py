"""
Store Intelligence System - Structured Logging Middleware
Logs every request with: trace_id, store_id, endpoint, latency_ms, event_count, status_code.
"""

import time
import uuid
import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = structlog.get_logger("store_intelligence")


def setup_logging():
    """Configure structlog for JSON structured logging."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(20),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that logs every API request with structured fields:
    - trace_id: unique request identifier for distributed tracing
    - store_id: extracted from path if available
    - endpoint: the request path
    - method: HTTP method
    - latency_ms: request processing time
    - status_code: HTTP response status
    - event_count: number of events (for ingest endpoint)
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        trace_id = str(uuid.uuid4())[:8]
        start_time = time.perf_counter()

        # Extract store_id from path if present
        store_id = None
        path_parts = request.url.path.strip("/").split("/")
        if "stores" in path_parts:
            idx = path_parts.index("stores")
            if idx + 1 < len(path_parts):
                store_id = path_parts[idx + 1]

        # Add trace_id to request state for downstream use
        request.state.trace_id = trace_id

        try:
            response = await call_next(request)
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

            # Log the request
            log_data = {
                "trace_id": trace_id,
                "method": request.method,
                "endpoint": request.url.path,
                "status_code": response.status_code,
                "latency_ms": latency_ms,
            }
            if store_id:
                log_data["store_id"] = store_id

            # Check for event_count header (set by ingest endpoint)
            event_count = response.headers.get("X-Event-Count")
            if event_count:
                log_data["event_count"] = int(event_count)

            if response.status_code >= 500:
                logger.error("request_failed", **log_data)
            elif response.status_code >= 400:
                logger.warning("request_client_error", **log_data)
            else:
                logger.info("request_completed", **log_data)

            return response

        except Exception as exc:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                "request_exception",
                trace_id=trace_id,
                method=request.method,
                endpoint=request.url.path,
                latency_ms=latency_ms,
                error=str(exc),
            )
            raise
