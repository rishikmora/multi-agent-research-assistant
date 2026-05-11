"""
FastAPI application entrypoint.
Wires together: logging, CORS, rate limiting, metrics, tracing,
health checks, and all API routers.
"""
from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from app.core.memory.session import get_redis
from app.api.v1.endpoints.research import router as research_router
from app.schemas.research import HealthResponse

log = get_logger(__name__)


# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_logging()
    log.info("app.starting",
             name=settings.app_name,
             version=settings.app_version,
             env=settings.environment)

    # Verify Redis connectivity
    try:
        redis = await get_redis()
        await redis.ping()
        log.info("redis.connected")
    except Exception as exc:
        log.error("redis.connection_failed", error=str(exc))
        # Don't crash — degrade gracefully

    yield

    log.info("app.shutting_down")
    # Cancel any orphaned pipeline tasks
    for task in asyncio.all_tasks():
        if task.get_name().startswith("pipeline-"):
            task.cancel()


# ── App factory ───────────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Production-grade Multi-Agent Research Assistant API",
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        openapi_url="/openapi.json" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # ── Middleware (order matters — outermost first) ───────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(SlowAPIMiddleware)

    # Request ID + structured logging middleware
    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        import uuid
        import time
        import structlog.contextvars

        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        start = time.monotonic()
        try:
            response: Response = await call_next(request)
        except Exception as exc:
            log.error("request.unhandled_error", error=str(exc))
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"},
                headers={"X-Request-ID": request_id},
            )
        finally:
            latency = time.monotonic() - start
            log.info("request.complete",
                     status_code=response.status_code,
                     latency_ms=round(latency * 1000))
            structlog.contextvars.clear_contextvars()

        response.headers["X-Request-ID"] = request_id
        return response

    # ── Rate limit error handler ───────────────────────────────────────────
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # ── Prometheus metrics endpoint ────────────────────────────────────────
    if settings.prometheus_enabled:
        metrics_app = make_asgi_app()
        app.mount("/metrics", metrics_app)

    # ── Routers ───────────────────────────────────────────────────────────
    app.include_router(research_router, prefix=settings.api_prefix)

    # ── Health endpoints ───────────────────────────────────────────────────
    @app.get("/health", response_model=HealthResponse, tags=["health"])
    async def health() -> HealthResponse:
        checks: dict[str, bool] = {}
        try:
            redis = await get_redis()
            await redis.ping()
            checks["redis"] = True
        except Exception:
            checks["redis"] = False

        checks["anthropic_key"] = bool(settings.anthropic_api_key)

        return HealthResponse(
            status="ok" if all(checks.values()) else "degraded",
            version=settings.app_version,
            environment=settings.environment,
            checks=checks,
        )

    @app.get("/ready", tags=["health"])
    async def readiness() -> dict[str, str]:
        """Kubernetes readiness probe."""
        return {"status": "ready"}

    @app.get("/live", tags=["health"])
    async def liveness() -> dict[str, str]:
        """Kubernetes liveness probe."""
        return {"status": "alive"}

    return app


app = create_app()
