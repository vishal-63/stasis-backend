import asyncio
import logging
import os

from app import db
from app.pipeline.downloader import DownloadResult, _cleanup_dir, download_reel
from app.pipeline.summariser import summarise_transcript
from app.pipeline.transcriber import transcribe_audio

logger = logging.getLogger(__name__)

STAGE_DOWNLOAD   = "download"
STAGE_TRANSCRIBE = "transcribe"
STAGE_SUMMARISE  = "summarise"
STAGE_SAVE       = "save"

# In-memory idempotency lock — prevents the same job from running
# concurrently if the poller and /process both enqueue it
_active_jobs: set[str] = set()
_active_jobs_lock = asyncio.Lock()


async def run_pipeline(
    note_id: str,
    job_id: str,
    user_id: str,
    source_url: str,
    resume_from: str | None = None,  # stage to resume from on retry
) -> None:
    """
    Full processing pipeline with stage-level failure tracking.
    On retry, resumes from the failed stage using cached intermediate results.

    resume_from: one of 'download', 'transcribe', 'summarise', 'save'
                 None means start from the beginning.
    """

    # ── Idempotency check ──────────────────────────────────────────
    async with _active_jobs_lock:
        if job_id in _active_jobs:
            logger.warning(f"Job {job_id} is already running — skipping duplicate")
            return
        _active_jobs.add(job_id)

    logger.info(f"Pipeline started — note: {note_id}, job: {job_id}, user: {user_id}, url: {source_url}")


    download_result: DownloadResult | None = None
    transcript: str | None = None
    temp_dir: str | None = None

    try:

        # ── Stage 1: Download ──────────────────────────────────────
        if resume_from in (None, STAGE_DOWNLOAD):
            logger.info(f"Stage: download starting — note: {note_id}")
            db.update_job_progress(job_id, "Downloading…", 5, "downloading")
            db.set_note_status(note_id, "downloading")

            try:
                download_result = await download_reel(source_url)
                logger.info(f"Stage: download completed — note: {note_id}, job: {job_id}, duration: {download_result.duration}")
            except Exception as e:
                logger.error(
                    f"Stage: download failed — note: {note_id}, job: {job_id}, error: {e}",
                    exc_info=True
                )
                db.mark_stage_failed(job_id, note_id, STAGE_DOWNLOAD, str(e))
                return

            db.update_job_progress(job_id, "Downloaded", 25, "transcribing")

            # Upload thumbnail — non-fatal if fails
            if download_result.thumbnail_path:
                try:
                    with open(download_result.thumbnail_path, "rb") as f:
                        image_bytes = f.read()
                    thumbnail_url = db.upload_thumbnail(user_id, note_id, image_bytes)
                    db.get_supabase().table("notes").update({
                        "thumbnail_url": thumbnail_url
                    }).eq("id", note_id).execute()
                    logger.info(f"Pipeline thumbnail uploaded — note: {note_id}, job: {job_id}")
                except Exception as e:
                    logger.warning(
                        f"Pipeline thumbnail failed — note: {note_id}, job: {job_id}, error: {e}",
                        retryable=False,
                    )

        # ── Stage 2: Transcribe ────────────────────────────────────
        if resume_from in (None, STAGE_DOWNLOAD, STAGE_TRANSCRIBE):
            if download_result is None:
                # Resuming from transcribe but no download result
                # This shouldn't happen — fall back to full retry
                logger.error(
                    f"Stage: transcribe failed — note: {note_id}, job: {job_id}, error: Download result unavailable — restarting from download",
                    exc_info=True
                )
                db.mark_stage_failed(
                    job_id, note_id, STAGE_DOWNLOAD,
                    "Download result unavailable for retry — restarting"
                )
                return

            logger.info(f"Stage: transcribe starting — note: {note_id}, job: {job_id}")
            db.set_note_status(note_id, "transcribing")
            db.update_job_progress(job_id, "Transcribing…", 40, "transcribing")

            try:
                transcript = await transcribe_audio(
                    download_result.audio_path,
                    description_text=download_result.description if download_result else None)
                logger.info(f"Stage: transcribe completed — note: {note_id}, job: {job_id}, transcript_length: {len(transcript)}")
            except Exception as e:
                logger.error(
                    f"Stage: transcribe failed — note: {note_id}, job: {job_id}, error: {e}",
                    exc_info=True
                )
            except Exception as e:
                logger.error(
                    f"Stage: transcribe failed — note: {note_id}, job: {job_id}, error: {e}",
                    exc_info=True
                )
                db.mark_stage_failed(job_id, note_id, STAGE_TRANSCRIBE, str(e))
                _cleanup_temp(download_result)
                return

            db.update_job_progress(job_id, "Transcribed", 65, "summarising")

        # ── Stage 3: Summarise ─────────────────────────────────────
        if resume_from in (None, STAGE_DOWNLOAD, STAGE_TRANSCRIBE, STAGE_SUMMARISE):
            if transcript is None:
                logger.error(
                    f"Stage: summarise failed — note: {note_id}, job: {job_id}, error: Transcript unavailable — restarting from download",
                    exc_info=True
                )
                db.mark_stage_failed(
                    job_id, note_id, STAGE_DOWNLOAD,
                    "Transcript unavailable for retry — restarting from download"
                )
                return
            
            logger.info(f"Stage: summarise starting — note: {note_id}, job: {job_id}")
            db.set_note_status(note_id, "summarising")
            db.update_job_progress(job_id, "Summarising…", 75, "summarising")

            try:
                summary = await summarise_transcript(
                    transcript=transcript,
                    title_hint=download_result.title if download_result else None,
                    description_hint=download_result.description if download_result else None,
                )
                logger.info(f"Stage: summarise completed — note: {note_id}, job: {job_id}, summary_length: {len(summary.summary)}")
            except Exception as e:
                logger.error(
                    f"Stage: summarise failed — note: {note_id}, job: {job_id}, error: {e}",
                    exc_info=True
                )
                db.mark_stage_failed(job_id, note_id, STAGE_SUMMARISE, str(e))
                _cleanup_temp(download_result)
                return

            db.update_job_progress(job_id, "Saving…", 90, "summarising")

        # ── Stage 4: Save ──────────────────────────────────────────
        logger.info(f"Stage: save starting — note: {note_id}, job: {job_id}")
        try:
            db.update_note_content(note_id, {
                "title":        summary.title,
                "summary":      summary.summary,
                "transcript":   transcript,
                # "key_points":   summary.key_points,
                # "action_items": summary.action_items,
                "status":       "done",
            })
            db.mark_job_complete(job_id)
            logger.info(f"Stage: save completed — note: {note_id}, job: {job_id}")

        except Exception as e:
            logger.error(
                f"Stage: save failed — note: {note_id}, job: {job_id}, error: {e}",
                exc_info=True
            )
            db.mark_stage_failed(job_id, note_id, STAGE_SAVE, str(e))

    except Exception as e:
        # Catch-all for unexpected errors — treat as download stage failure
        logger.exception(
            f"Stage: download failed — note: {note_id}, job: {job_id}, error: {e}"
        )
        db.mark_stage_failed(job_id, note_id, STAGE_DOWNLOAD, str(e))

    finally:
        _cleanup_temp(download_result)


def _cleanup_temp(download_result: DownloadResult | None) -> None:
    """Clean up temp files — always runs even on failure."""
    if download_result and download_result.audio_path:
        temp_dir = os.path.dirname(download_result.audio_path)
        _cleanup_dir(temp_dir)