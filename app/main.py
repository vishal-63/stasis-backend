import asyncio
import logging
import os
from contextlib import asynccontextmanager

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.auth import CurrentUser
from app.config import get_settings
from app import db
from app.queue import Job, job_queue
from app.security import (
    check_api_secret,
    limiter,
    validate_note_id,
    validate_reel_url,
)

# ─── Logging setup ────────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── App lifespan ─────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    worker_task = asyncio.create_task(job_queue.start_worker())
    watchdog_task = asyncio.create_task(_watchdog())
    poller_task = asyncio.create_task(_job_poller())

    logger.info("Stasis backend started")
    yield

    await job_queue.stop()
    worker_task.cancel()
    watchdog_task.cancel()
    poller_task.cancel()                        
    logger.info("Stasis backend shut down")


async def _watchdog():
    """Periodically requeue stalled jobs."""
    while True:
        await asyncio.sleep(5 * 60)  # every 5 minutes
        try:
            db.requeue_stalled_jobs(stall_threshold_minutes=15)
        except Exception as e:
            logger.warning(f"Watchdog error: {e}")

async def _job_poller():
    """
    Fallback only — picks up jobs that were missed due to server
    restart or queue overflow. Skips anything already active.
    """
    logger.info("Job poller started (fallback mode)")
    while True:
        try:
            await asyncio.sleep(30)

            job_row = db.dequeue_job()
            if not job_row:
                continue
            logger.info(f"Job {job_row}")
            note_id = job_row["note_id"]
            job_id  = job_row["id"]
            user_id = job_row["user_id"]

            logger.info(f"Poller (fallback) picked up missed job {job_id}")

            note_result = db.get_supabase().table("notes") \
                .select("source_url") \
                .eq("id", note_id) \
                .single() \
                .execute()

            if not note_result.data:
                logger.warning(f"Poller: note {note_id} not found, skipping")
                continue

            source_url = note_result.data.get("source_url")
            if not source_url:
                logger.warning(f"Poller: note {note_id} has no source_url, skipping")
                continue

            try:
                await job_queue.enqueue(Job(
                    note_id=note_id,
                    job_id=job_id,
                    user_id=user_id,
                    source_url=source_url,
                ))
            except RuntimeError as e:
                logger.warning(f"Poller: queue full, resetting job {job_id}: {e}")
                db.update_job_progress(job_id, "Queued…", 0, "queued")

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.exception(f"Job poller error: {e}")

# ─── App setup ────────────────────────────────────────────────────────

settings = get_settings()

app = FastAPI(
    title="Stasis API",
    # Hide docs in production
    docs_url="/docs" if settings.is_development else None,
    redoc_url="/redoc" if settings.is_development else None,
    openapi_url="/openapi.json" if settings.is_development else None,
    lifespan=lifespan,
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# CORS — only allow configured origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type", "X-API-Secret"],
)

# Trusted host — Render provides the hostname
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=(
        ["*"] if settings.is_development
        else ["*.onrender.com", "*.stasis.app"]
    ),
)


# ─── Security middleware ───────────────────────────────────────────────

@app.middleware("http")
async def security_headers(request: Request, call_next):
    """Add security headers to every response."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    # Remove server header to avoid fingerprinting
    if "Server" in response.headers:
        del response.headers["Server"]
    return response


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Structured request logging. Never logs Authorization headers."""
    logger.info(
        "request",
        extra={
            "method": request.method,
            "path": request.url.path,
            # Log client IP for abuse tracking
            "ip": request.client.host if request.client else "unknown",
        },
    )
    response = await call_next(request)
    logger.info("response", extra={"status": response.status_code})
    return response


# ─── Request/Response models ───────────────────────────────────────────

class ProcessReelRequest(BaseModel):
    url: str
    note_id: str

    @field_validator("url")
    @classmethod
    def url_must_be_string(cls, v):
        if not isinstance(v, str):
            raise ValueError("url must be a string")
        return v.strip()

    @field_validator("note_id")
    @classmethod
    def note_id_must_be_string(cls, v):
        if not isinstance(v, str):
            raise ValueError("note_id must be a string")
        return v.strip()


class ProcessReelResponse(BaseModel):
    job_id: str
    note_id: str
    status: str = "queued"


class JobStatusResponse(BaseModel):
    job_id: str
    note_id: str
    status: str
    stage: str | None
    progress: int
    error: str | None


class HealthResponse(BaseModel):
    status: str
    queue_size: int


# ─── Routes ───────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Public health check — no auth required."""
    return HealthResponse(status="ok", queue_size=job_queue.size)


@app.post(
    "/process",
    response_model=ProcessReelResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
@limiter.limit(f"{settings.max_requests_per_minute}/minute")
async def process_reel(
    request: Request,
    body: ProcessReelRequest,
    current_user: CurrentUser,
):
    try:
        check_api_secret(request)
        clean_url = validate_reel_url(body.url)
        note_id = validate_note_id(body.note_id)

        if not db.verify_note_ownership(note_id, current_user.user_id):
            raise HTTPException(status_code=403, detail="Note not found or access denied")

        note_count = db.get_user_note_count(current_user.user_id)
        if note_count > 1000:
            raise HTTPException(status_code=429, detail="Note limit reached")

        # Check if a non-failed job already exists
        existing = db.get_supabase().table("processing_jobs") \
            .select("id, status") \
            .eq("note_id", note_id) \
            .not_.in_("status", ["failed"]) \
            .order("created_at", desc=True) \
            .limit(1) \
            .execute()

        if existing.data:
            existing_job = existing.data[0]
            logger.info(f"Job already exists for note {note_id}: {existing_job['id']}")
            return ProcessReelResponse(
                job_id=existing_job["id"],
                note_id=note_id,
                status=existing_job["status"],
            )

        # Create job row
        job_result = db.get_supabase().table("processing_jobs").insert({
            "note_id": note_id,
            "user_id": current_user.user_id,
            "status": "queued",
        }).execute()

        if not job_result.data:
            raise HTTPException(status_code=500, detail="Failed to create processing job")

        job_id = job_result.data[0]["id"]

        try:
            await job_queue.enqueue(Job(
                note_id=note_id,
                job_id=job_id,
                user_id=current_user.user_id,
                source_url=clean_url,
            ))
            logger.info(f"Job {job_id} enqueued immediately")
        except RuntimeError as e:
            logger.warning(f"Queue full, job {job_id} will be picked up by poller: {e}")

        return ProcessReelResponse(
            job_id=job_id,
            note_id=note_id,
            status="queued",
        )
    except Exception as e:
        logger.error("An error occurrd while processing reel:", e)

@app.get(
    "/jobs/{note_id}",
    response_model=JobStatusResponse,
)
@limiter.limit("60/minute")
async def get_job_status(
    request: Request,
    note_id: str,
    current_user: CurrentUser,
):
    """
    Poll job status for a note.
    Verifies note ownership before returning job data.
    """
    check_api_secret(request)

    # Validate note_id
    note_id = validate_note_id(note_id)

    # Verify ownership before exposing job data
    if not db.verify_note_ownership(note_id, current_user.user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Note not found or access denied",
        )

    job = db.get_supabase().table("processing_jobs") \
        .select("*") \
        .eq("note_id", note_id) \
        .order("created_at", desc=True) \
        .limit(1) \
        .execute()

    if not job.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )

    j = job.data[0]
    return JobStatusResponse(
        job_id=j["id"],
        note_id=note_id,
        status=j["status"],
        stage=j.get("stage"),
        progress=j.get("progress", 0),
        error=j.get("error"),
    )


@app.delete(
    "/notes/{note_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
@limiter.limit("30/minute")
async def delete_note_assets(
    request: Request,
    note_id: str,
    current_user: CurrentUser,
):
    """
    Delete storage assets for a note.
    Called by the client after deleting the note row from Supabase.
    """
    check_api_secret(request)
    note_id = validate_note_id(note_id)

    # Verify ownership before deleting assets
    if not db.verify_note_ownership(note_id, current_user.user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Note not found or access denied",
        )

    db.delete_note_assets(current_user.user_id, note_id)


# ─── Global error handler ─────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch-all error handler.
    Never expose internal error details in production.
    """
    logger.exception(f"Unhandled exception: {exc}")

    if settings.is_development:
        detail = str(exc)
    else:
        detail = "An internal error occurred"

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": detail},
    )