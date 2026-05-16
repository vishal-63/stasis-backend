import asyncio
import logging
import os
import re
import tempfile
import uuid
from pathlib import Path

import yt_dlp

from app.config import get_settings

logger = logging.getLogger(__name__)

YT_DLP_ALLOWED_DOMAINS = ["instagram.com", "www.instagram.com"]


class DownloadResult:
    def __init__(
        self,
        audio_path: str,
        thumbnail_path: str | None,
        duration: float,
        title: str | None,
        description: str | None,
    ):
        self.audio_path = audio_path
        self.thumbnail_path = thumbnail_path
        self.duration = duration
        self.title = title
        self.description = description


async def download_reel(url: str) -> DownloadResult:
    """
    Download audio and thumbnail from an Instagram reel URL.

    Security considerations:
    - URL is validated before this function is called (see security.py)
    - yt-dlp is sandboxed: no arbitrary code execution, no shell=True
    - Output directory is a temp dir cleaned up by the caller
    - Duration is checked against the configured maximum
    - No user-controlled strings are passed to subprocess
    """
    logger.debug(f"Download starting — url: {url}")
    settings = get_settings()

    # Create isolated temp directory for this download
    # Use a random name to prevent path traversal / prediction
    temp_dir = tempfile.mkdtemp(prefix=f"reel_{uuid.uuid4().hex}_")
    output_template = os.path.join(temp_dir, "audio.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "writethumbnail": True,
        "nocheckcertificate": False,
        "quiet": True,
        "no_warnings": False,

        # "cookiesfrombrowser": ("chrome", None, None, None),
        # Or use a cookies file instead:
        "cookiefile": settings.instagram_cookies_path,

        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "96",
        }],

        "filesize_max": 200 * 1024 * 1024,
        "allowed_extractors": ["instagram"],
        "external_downloader": None,
    }
    
    # Run yt-dlp in a thread (it's synchronous)
    loop = asyncio.get_event_loop()
    info = await loop.run_in_executor(
        None,
        lambda: _run_ytdlp(url, ydl_opts),
    )

    # Enforce duration limit
    duration = info.get("duration") or 0
    if duration > settings.max_video_duration_seconds:
        # Clean up before raising
        _cleanup_dir(temp_dir)
        raise ValueError(
            f"Reel duration {duration}s exceeds maximum "
            f"{settings.max_video_duration_seconds}s"
        )

    # Find downloaded audio file
    audio_path = _find_file(temp_dir, ".mp3")
    if not audio_path:
        _cleanup_dir(temp_dir)
        raise RuntimeError("Audio download failed: no output file found")

    # Find thumbnail if downloaded
    thumbnail_path = (
        _find_file(temp_dir, ".jpg")
        or _find_file(temp_dir, ".webp")
        or _find_file(temp_dir, ".png")
    )

    # Sanitise metadata strings — never trust external data
    raw_title = info.get("title") or ""
    raw_description = info.get("description") or ""

    logger.debug(f"Download complete — duration: {duration}s, audio: {audio_path}, thumbnail: {thumbnail_path}")
    return DownloadResult(
        audio_path=audio_path,
        thumbnail_path=thumbnail_path,
        duration=duration,
        title=_sanitise_text(raw_title, max_length=500),
        description=_sanitise_text(raw_description, max_length=5000),
    )


def _run_ytdlp(url: str, opts: dict) -> dict:
    """Run yt-dlp synchronously and return the info dict."""
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return info or {}


def _find_file(directory: str, extension: str) -> str | None:
    """Find the first file with the given extension in a directory."""
    for entry in Path(directory).iterdir():
        if entry.suffix.lower() == extension:
            return str(entry)
    return None


def _sanitise_text(text: str, max_length: int) -> str:
    """Remove control characters and truncate."""
    # Remove null bytes and other dangerous control chars
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return cleaned[:max_length].strip()


def _cleanup_dir(directory: str) -> None:
    """Best-effort cleanup of a temp directory."""
    import shutil
    try:
        shutil.rmtree(directory, ignore_errors=True)
    except Exception as e:
        logger.warning(f"Temp dir cleanup failed: {e}")