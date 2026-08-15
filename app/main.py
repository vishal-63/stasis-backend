import asyncio
import logging
import os
from contextlib import asynccontextmanager

import structlog
from datetime import datetime, timezone, timedelta
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
    # Verify cookies on startup
    _check_cookies_on_startup()

    app.state.worker_task = asyncio.create_task(job_queue.start_worker())
    app.state.poller_task = asyncio.create_task(_poller())
    
    logger.info("Stasis backend started")
    yield

    await job_queue.stop()
    app.state.worker_task.cancel()
    app.state.poller_task.cancel()
    logger.info("Stasis backend shut down")

def _check_cookies_on_startup():
    import os
    from app.config import get_settings
    settings = get_settings()
    path = settings.instagram_cookies_path
    if not path:
        logger.warning("No Instagram cookies path configured")
        return
    if not os.path.exists(path):
        logger.warning(f"Cookies file not found at {path}")
        return
    with open(path, 'r') as f:
        content = f.read()
    if 'sessionid' not in content:
        logger.warning("Cookies file missing sessionid — Instagram downloads will fail")
        return
    logger.info("Cookies file found and contains sessionid")


async def _poller():
    """
    Unified poller — runs every 10 seconds and handles:

    1. MISSED JOBS     — queued jobs older than 60s not picked up by /process
    2. RETRYABLE JOBS  — failed jobs whose next_retry_at has passed
    3. STALLED JOBS    — jobs stuck in active status for > 15 minutes
    """
    logger.info("Unified poller started")

    while True:
        try:
            await asyncio.sleep(10)
            await _poll_once()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.exception(f"Poller error: {e}")


async def _poll_once():
    """Single poll cycle — called every 10 seconds."""
    now = datetime.now(timezone.utc)

    # ── 1. Requeue stalled jobs ────────────────────────────────────
    # Jobs stuck in an active status for more than 15 minutes
    # are assumed crashed — reset them to queued for pickup
    stall_cutoff = (now - timedelta(minutes=15)).isoformat()
    stalled = db.get_supabase().table("processing_jobs") \
        .select("*") \
        .in_("status", ["downloading", "transcribing", "extracting"]) \
        .lt("updated_at", stall_cutoff) \
        .execute()

    for job in (stalled.data or []):
        logger.info(
            f"Stalled job found: {job['id']}, updated_at={job['updated_at']}"
        )
        db.get_supabase().table("processing_jobs").update({
            "status":   "queued",
            "stage":    "Requeued after stall…",
            "progress": 0,
        }).eq("id", job["id"]).execute()
        db.set_note_status(job["note_id"], "queued")

        logger.info(
            f"Job {job['id']} requeued after stall — note: {job['note_id']}"
        )
        # Enqueue directly instead of waiting for missed jobs check
        await _enqueue_job(
            job=job,
            resume_from=None,  # restart from beginning — stall means state is unknown
            reason="stalled",
        )

    # ── 2. Pick up retryable failed jobs ──────────────────────────
    # Failed jobs with next_retry_at in the past and retries remaining
    retryable = db.get_supabase().table("processing_jobs") \
        .select("*") \
        .eq("status", "failed") \
        .lte("next_retry_at", now.isoformat()) \
        .execute()

    retryable_jobs = [
        j for j in (retryable.data or [])
        if j.get("retry_count", 0) < j.get("max_retries", 3)
        and j.get("next_retry_at") is not None
    ]

    for job in retryable_jobs:
        logger.info(
            f"Retryable job found: {job['id']}, retry_count={job.get('retry_count', 0)}, next_retry_at={job.get('next_retry_at')}",
        )
        await _enqueue_job(
            job=job,
            resume_from=job.get("failed_stage"),
            reason="retry",
        )

    # ── 3. Pick up missed queued jobs ─────────────────────────────
    # Jobs sitting in queued for > 60s — missed by /process endpoint
    missed_cutoff = (now - timedelta(seconds=60)).isoformat()
    missed = db.get_supabase().table("processing_jobs") \
        .select("*") \
        .eq("status", "queued") \
        .lt("created_at", missed_cutoff) \
        .execute()

    for job in (missed.data or []):
        # Skip if this note already has an active job
        active = db.get_supabase().table("processing_jobs") \
            .select("id") \
            .eq("note_id", job["note_id"]) \
            .in_("status", ["downloading", "transcribing", "extracting"])   \
            .execute()

        if active.data:
            continue

        logger.info(f"Missed job found: {job['id']}, created_at={job['created_at']}")
        await _enqueue_job(
            job=job,
            resume_from=None,
            reason="missed",
        )
    # ── 4. Notes with no processing job ──────────────────────────
    # Notes stuck in queued/failed with no processing_job row
    # Happens when /process endpoint failed after note creation
    orphan_cutoff = (now - timedelta(seconds=15)).isoformat()
    orphaned_notes = db.get_supabase().table("notes") \
        .select("id, user_id, source_url") \
        .in_("status", ["queued", "failed"]) \
        .lt("created_at", orphan_cutoff) \
        .execute()

    for note in (orphaned_notes.data or []):
        note_id = note["id"]

        # Check if a processing job exists for this note
        existing_job = db.get_supabase().table("processing_jobs") \
            .select("id") \
            .eq("note_id", note_id) \
            .execute()

        if existing_job.data:
            continue  # job exists — handled by other checks

        if not note.get("source_url"):
            logger.warning(f"Orphaned note found without source URL: {note_id}")
            continue

        logger.warning(f"Orphaned note found: {note_id}")

        
        # Create the missing job row
        job_result = db.get_supabase().table("processing_jobs").insert({
            "note_id": note_id,
            "user_id": note["user_id"],
            "status":  "queued",
        }).execute()

        if not job_result.data:
            logger.error(f"Failed to create processing job for orphaned note {note_id}")
            continue

        job = job_result.data[0]

        await _enqueue_job(
            job=job,
            resume_from=None,
            reason="orphaned",
        )


async def _enqueue_job(job: dict, resume_from: str | None, reason: str) -> None:
    """
    Fetch the note's source_url and enqueue the job.
    Shared by all three poller cases.
    """
    job_id  = job["id"]
    note_id = job["note_id"]
    user_id = job["user_id"]

    note_result = db.get_supabase().table("notes") \
        .select("source_url") \
        .eq("id", note_id) \
        .single() \
        .execute()

    if not note_result.data or not note_result.data.get("source_url"):
        logger.warning(f"Job {job_id} has no source URL", job_id=job_id, note_id=note_id)
        return

    source_url = note_result.data["source_url"]

    # Reset status so client sees activity
    db.reset_job_for_retry(job_id, resume_from or "download")

    try:
        await job_queue.enqueue(Job(
            note_id=note_id,
            job_id=job_id,
            user_id=user_id,
            source_url=source_url,
            resume_from=resume_from,
        ))
        logger.info(f"Job {job_id} enqueued from poller (reason: {reason})")
    except RuntimeError as e:
        logger.warning(f"Queue full for job {job_id}, will retry later (reason: {reason}): {e}",)
        db.get_supabase().table("processing_jobs").update({
            "status":    "failed" if reason == "retry" else "queued",
            "stage":     "Waiting for queue space…",
        }).eq("id", job_id).execute()

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
    allow_headers=["Authorization", "Content-Type"],
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

@app.get("/health")
async def health(request: Request):
    worker_task = request.app.state.worker_task
    poller_task = request.app.state.poller_task
    return {
        "status": "ok",
        "queue_size": job_queue._queue.qsize(),
        "worker_running": job_queue._running,
        "worker_task_done": worker_task.done(),
        "worker_task_cancelled": worker_task.cancelled(),
        "worker_task_exception": str(worker_task.exception()) if worker_task.done() and not worker_task.cancelled() else None,
        "poller_task_done": poller_task.done(),
    }


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
        logger.info("Received /process request")
        clean_url = validate_reel_url(body.url)
        note_id = validate_note_id(body.note_id)
        logger.info(f"Processing request — note: {note_id}, user: {current_user.user_id}, url: {clean_url}")
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
                resume_from=None,
            ))
            logger.info(f"Job {job_id} enqueued immediately")
        except RuntimeError as e:
            logger.warning(f"Queue full, job {job_id} will be picked up by poller: {e}")

        return ProcessReelResponse(
            job_id=job_id,
            note_id=note_id,
            status="queued",
        )
    except Exception:
        logger.exception("Unhandled exception in /process")
        raise

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