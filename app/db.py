import logging
from typing import Any

from supabase import Client, create_client
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings

logger = logging.getLogger(__name__)


# Created once at module load, reused for all calls
_client: Client | None = None

def get_supabase() -> Client:
    global _client
    if _client is None:
        settings = get_settings()
        _client = create_client(
            settings.supabase_url,
            settings.supabase_secret_key,
        )
        logger.info("Supabase client initialised")
    return _client


# ─── Notes ────────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def set_note_status(note_id: str, status: str) -> None:
    db = get_supabase()
    db.table("notes").update({"status": status}).eq("id", note_id).execute()
    logger.info(f"Note {note_id} status → {status}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def set_note_error(note_id: str, message: str) -> None:
    # Truncate error message — never write unbounded strings to DB
    safe_message = str(message)[:500]
    db = get_supabase()
    db.table("notes").update({
        "status": "failed",
        "error_message": safe_message,
    }).eq("id", note_id).execute()
    logger.error(f"Note {note_id} failed: {safe_message}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def update_note_content(note_id: str, content: dict[str, Any]) -> None:
    """
    Write AI-generated content to a note.
    Only whitelisted fields are written — never pass raw dicts
    from external services directly to this function.
    """
    allowed_fields = {
        "title", "summary", "transcript",
        "key_points", "action_items",
        "thumbnail_url", "status",
    }
    safe_content = {
        k: v for k, v in content.items() if k in allowed_fields
    }
    if not safe_content:
        logger.warning(f"update_note_content called with no valid fields for {note_id}")
        return

    db = get_supabase()
    db.table("notes").update(safe_content).eq("id", note_id).execute()
    logger.info(f"Note {note_id} content updated: {list(safe_content.keys())}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
# def auto_tag_note(note_id: str, user_id: str, tag_names: list[str]) -> None:
#     """
#     Create tags (if they don't exist) and link them to the note.
#     Tags are scoped to the user — no cross-user tag leakage.
#     """
#     if not tag_names:
#         return

#     # Sanitise tag names
#     safe_tags = [str(t).strip().lower()[:50] for t in tag_names[:10]]
#     safe_tags = [t for t in safe_tags if t]

#     db = get_supabase()

#     for tag_name in safe_tags:
#         # Upsert tag for this user
#         result = db.table("tags").upsert(
#             {"user_id": user_id, "name": tag_name},
#             on_conflict="user_id,name",
#         ).execute()

#         if result.data:
#             tag_id = result.data[0]["id"]
#             # Link tag to note (ignore if already exists)
#             db.table("note_tags").upsert(
#                 {"note_id": note_id, "tag_id": tag_id},
#                 on_conflict="note_id,tag_id",
#             ).execute()

#     logger.info(f"Tagged note {note_id} with: {safe_tags}")


def verify_note_ownership(note_id: str, user_id: str) -> bool:
    """
    Confirm the note belongs to the user before the backend processes it.
    Even though the client creates the note, we verify on the backend
    to prevent a race condition where a malicious client submits
    someone else's note_id.
    """
    db = get_supabase()
    result = db.table("notes") \
        .select("id") \
        .eq("id", note_id) \
        .eq("user_id", user_id) \
        .execute()
    return len(result.data) > 0


def get_user_note_count(user_id: str) -> int:
    """Used for quota enforcement."""
    db = get_supabase()
    result = db.table("notes") \
        .select("id", count="exact") \
        .eq("user_id", user_id) \
        .execute()
    return result.count or 0


# ─── Processing jobs ──────────────────────────────────────────────────

def dequeue_job() -> dict | None:
    """
    Fallback poller — only picks up jobs that have been queued
    for more than 60 seconds (i.e. missed by the /process endpoint).
    """
    from datetime import datetime, timezone, timedelta
    db = get_supabase()

    # Only pick up jobs queued for more than 60 seconds
    # Jobs enqueued immediately by /process will be in active status by then
    cutoff = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()

    result = db.table("notes") \
        .select("*") \
        .eq("status", "queued") \
        .lt("created_at", cutoff) \
        .order("created_at") \
        .limit(1) \
        .execute()

    if not result.data:
        return None

    note = result.data[0]

    result = db.table("processing_jobs") \
        .select("*") \
        .eq("note_id", note["id"]) \
        .execute()
    
    if not result.data:
        logger.info(f"No job create for note {note['id']}, creating new job.")
        new_job_row = {
            "status": "queued",
            "note_id": note["id"],
            "user_id": note["user_id"]
        }
        try:
            response = db.table("processing_jobs").insert(new_job_row).execute()
            logger.info(f"Created new Job {response}")
            return response.data[0]
        except Exception as e:
            logger.error("An error occurred while creating new job:", e)
            return
    else:
        # Check no active job exists for this note
        
        job = result.data[0]
        active = db.table("processing_jobs") \
            .select("id") \
            .eq("note_id", note["id"]) \
            .in_("status", ["downloading", "transcribing", "summarising"]) \
            .execute()

        if active.data:
            logger.warning(f"Poller: note {note['id']} already active, skipping")
            return None

        # Atomically claim
        claim = db.table("processing_jobs") \
            .update({"status": "downloading", "progress": 1}) \
            .eq("id", job["id"]) \
            .eq("status", "queued") \
            .execute()

        if not claim.data:
            logger.warning(f"Job {job['id']} already claimed, skipping")
            return None

        return job

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def update_job_progress(
    job_id: str,
    stage: str,
    progress: int,
    status: str = "downloading",
) -> None:
    progress = max(0, min(100, progress))  # clamp to 0-100
    safe_stage = str(stage)[:100]
    db = get_supabase()
    db.table("processing_jobs").update({
        "stage": safe_stage,
        "progress": progress,
        "status": status,
    }).eq("id", job_id).execute()


def mark_job_complete(job_id: str) -> None:
    from datetime import datetime, timezone
    db = get_supabase()
    db.table("processing_jobs").update({
        "status": "done",
        "progress": 100,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", job_id).execute()


def mark_job_failed(job_id: str, error: str) -> None:
    from datetime import datetime, timezone
    safe_error = str(error)[:500]
    db = get_supabase()
    db.table("processing_jobs").update({
        "status": "failed",
        "error": safe_error,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", job_id).execute()
    logger.error(f"Job {job_id} failed: {safe_error}")


def requeue_stalled_jobs(stall_threshold_minutes: int = 15) -> int:
    """
    Reset jobs stuck in a non-terminal state for too long.
    Called by the watchdog task every few minutes.
    """
    from datetime import datetime, timezone, timedelta
    db = get_supabase()
    cutoff = (
        datetime.now(timezone.utc) - timedelta(minutes=stall_threshold_minutes)
    ).isoformat()

    result = db.table("processing_jobs").update({
        "status": "queued",
        "stage": None,
        "progress": 0,
    }).in_("status", ["downloading", "transcribing", "summarising"]) \
      .lt("updated_at", cutoff) \
      .execute()

    count = len(result.data) if result.data else 0
    logger.info(f"Requeued {count} stalled jobs (threshold: {stall_threshold_minutes} minutes)")
    if count:
        logger.warning(f"Requeued {count} stalled jobs")
    return count


# ─── Storage ──────────────────────────────────────────────────────────

def upload_thumbnail(user_id: str, note_id: str, image_bytes: bytes) -> str:
    """
    Upload thumbnail to Supabase Storage.
    Path is scoped to user_id so users cannot overwrite each other's files.
    Returns the public URL.
    """
    if len(image_bytes) > 5 * 1024 * 1024:  # 5MB max
        raise ValueError("Thumbnail exceeds maximum size")

    db = get_supabase()
    path = f"{user_id}/{note_id}/thumbnail.jpg"

    db.storage.from_("thumbnails").upload(
        path,
        image_bytes,
        {"content-type": "image/jpeg", "upsert": "true"},
    )

    result = db.storage.from_("thumbnails").get_public_url(path)
    return result


def delete_note_assets(user_id: str, note_id: str) -> None:
    """Remove all storage assets for a note."""
    db = get_supabase()
    paths = [f"{user_id}/{note_id}/thumbnail.jpg"]
    try:
        db.storage.from_("thumbnails").remove(paths)
    except Exception as e:
        # Log but don't raise — asset cleanup is best-effort
        logger.warning(f"Asset cleanup failed for note {note_id}: {e}")