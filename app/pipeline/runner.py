import asyncio
import logging
import shutil

from app import db
from app.pipeline.downloader import DownloadResult, _cleanup_dir, download_reel
from app.pipeline.summariser import summarise_transcript
from app.pipeline.transcriber import transcribe_audio

logger = logging.getLogger(__name__)


async def run_pipeline(
    note_id: str,
    job_id: str,
    user_id: str,
    source_url: str,
) -> None:
    """
    Full processing pipeline: download → transcribe → summarise → save.
    Updates job progress at each stage via Supabase Realtime.
    Guarantees temp file cleanup even on failure.
    """
    logger.info(f"Pipeline started — note: {note_id}, job: {job_id}, user: {user_id}, url: {source_url}")
    download_result: DownloadResult | None = None

    try:
        logger.debug(f"Stage: download starting — note: {note_id}")
        # ── Stage 1: Download ──────────────────────────────────────
        db.update_job_progress(job_id, "Downloading reel…", 5, "downloading")
        db.set_note_status(note_id, "downloading")

        download_result = await download_reel(source_url)
        db.update_job_progress(job_id, "Download complete", 30, "transcribing")

        logger.debug(f"Stage: thumbnail upload starting — note: {note_id}")
        # ── Stage 2: Upload thumbnail ──────────────────────────────
        thumbnail_url = None
        if download_result.thumbnail_path:
            try:
                with open(download_result.thumbnail_path, "rb") as f:
                    image_bytes = f.read()
                thumbnail_url = db.upload_thumbnail(user_id, note_id, image_bytes)
                logger.debug(f"Thumbnail uploaded for note {note_id}")
            except Exception as e:
                # Thumbnail failure is non-fatal
                logger.warning(f"Thumbnail upload failed: {e}")

        logger.debug(f"Stage: transcribe starting — note: {note_id}, audio: {download_result.audio_path}")
        # ── Stage 3: Transcribe ────────────────────────────────────
        db.set_note_status(note_id, "transcribing")
        db.update_job_progress(job_id, "Transcribing audio…", 40, "transcribing")

        transcript = await transcribe_audio(download_result.audio_path, description_text=download_result.description)
        db.update_job_progress(job_id, "Transcription complete", 70, "summarising")

        logger.debug(f"Stage: summarise starting — note: {note_id}, transcript length: {len(transcript)}")
        # ── Stage 4: Summarise ─────────────────────────────────────
        db.set_note_status(note_id, "summarising")
        db.update_job_progress(job_id, "Generating notes…", 75, "summarising")

        summary = await summarise_transcript(
            transcript=transcript,
            title_hint=download_result.title,
            description_hint=download_result.description,
        )
        db.update_job_progress(job_id, "Saving note…", 90, "summarising")

        logger.debug(f"Stage: save starting — note: {note_id}")
        # ── Stage 5: Save to DB ────────────────────────────────────
        db.update_note_content(note_id, {
            "title": summary.title,
            "summary": summary.summary,
            "transcript": transcript,
            # "key_points": summary.key_points,
            # "action_items": summary.action_items,
            "thumbnail_url": thumbnail_url,
            "status": "done",
        })

        # Tag the note with AI-generated tags
        db.auto_tag_note(note_id, user_id, summary.tags)

        # Mark job complete
        db.mark_job_complete(job_id)
        logger.info(f"Pipeline complete for note {note_id}")

    except Exception as e:
        logger.exception(f"Pipeline error — note: {note_id}, job: {job_id}: {e}")
        db.set_note_error(note_id, str(e))
        db.mark_job_failed(job_id, str(e))

    finally:
        # Always clean up temp files — even on failure
        if download_result and download_result.audio_path:
            import os
            temp_dir = os.path.dirname(download_result.audio_path)
            _cleanup_dir(temp_dir)