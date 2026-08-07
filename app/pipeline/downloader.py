import asyncio
import logging
import os
import re
import tempfile
import subprocess
import uuid
import shutil
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

def _extract_audio(video_path: str, output_dir: str) -> str:
    """
    Extract audio from video using ffmpeg directly.
    Bypasses yt-dlp postprocessor to avoid ffprobe codec detection issues.
    Returns path to the extracted mp3 file.
    """
    audio_path = os.path.join(output_dir, "audio.mp3")

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",                   # strip video stream
        "-acodec", "libmp3lame", # explicit codec — no ffprobe guessing
        "-ar", "16000",          # 16kHz — optimal for Whisper
        "-ac", "1",              # mono — halves file size
        "-b:a", "32k",           # 32kbps sufficient for speech
        "-y",                    # overwrite without asking
        "-loglevel", "error",    # suppress non-error output
        audio_path,
    ]

    logger.info(f"Extracting audio with ffmpeg — video: {video_path}, output: {audio_path}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        logger.error(
            "audio_extraction_failed",
            returncode=result.returncode,
            stderr=result.stderr,
        )
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr}")

    if not os.path.exists(audio_path):
        raise RuntimeError("ffmpeg ran successfully but audio file not found")

    logger.info(f"Audio extraction completed — audio: {audio_path}")
    return audio_path


def _find_thumbnail(temp_dir: str, base_name: str) -> str | None:
    """Find thumbnail file downloaded by yt-dlp."""
    for ext in ["jpg", "jpeg", "webp", "png"]:
        path = os.path.join(temp_dir, f"{base_name}.{ext}")
        if os.path.exists(path):
            return path
    return None


def _find_video(temp_dir: str) -> str | None:
    """Find the downloaded video file regardless of extension."""
    for ext in ["mp4", "webm", "mkv", "m4a", "mp3"]:
        for f in os.listdir(temp_dir):
            if f.endswith(f".{ext}") and not f.endswith(".part"):
                return os.path.join(temp_dir, f)
    return None

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
    logger.info(f"Download starting — url: {url}")
    settings = get_settings()

    temp_dir = tempfile.mkdtemp(prefix=f"stasis_{uuid.uuid4().hex}_")
    base_name = "media"
    output_template = os.path.join(temp_dir, f"{base_name}.%(ext)s")


    ig_cookies_path = None
    if settings.instagram_cookies_path:
        source = Path(settings.instagram_cookies_path)
        if source.exists():
            ig_cookies_path = os.path.join(temp_dir, "ig_cookies.txt")
            shutil.copy2(str(source), ig_cookies_path)
            os.chmod(ig_cookies_path, 0o600)
            logger.info(f"Copied cookies to writable path: {ig_cookies_path}")
        else:
            logger.warning(f"Cookies file not found: {settings.instagram_cookies_path}")

    ydl_opts = {
        # Download best available format as a single file
        # Do NOT use bestaudio — it often produces DASH streams
        # that ffprobe struggles with on certain ffmpeg builds
        "format": "best[ext=mp4]/best",
        "outtmpl": output_template,
        "writethumbnail": True,
        "quiet": True,
        "no_warnings": False,
        "cookiefile": ig_cookies_path,
        "allowed_extractors": ["instagram"],
        "socket_timeout": 30,
        "filesize_max": 50 * 1024 * 1024,
        "retries": 3,
        "fragment_retries": 3,
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                "Version/17.0 Mobile/15E148 Safari/604.1"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
    }

    logger.info(f"Download starting — url: {url}")


    try:
        # Run yt-dlp in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        info = await loop.run_in_executor(None, lambda: _run_ytdlp(url, ydl_opts))
    except Exception as e:
        _cleanup_dir(temp_dir)
        raise

    # Enforce duration limit
    duration = info.get("duration") or 0
    if duration > settings.max_video_duration_seconds:
        # Clean up before raising
        _cleanup_dir(temp_dir)
        raise ValueError(
            f"Reel duration {duration}s exceeds maximum "
            f"{settings.max_video_duration_seconds}s"
        )

     # Find the downloaded video file
    video_path = _find_video(temp_dir)
    if not video_path:
        _cleanup_dir(temp_dir)
        raise RuntimeError("yt-dlp completed but no video file found in temp dir")

    logger.info(f"Download completed — video: {video_path}, duration: {duration}s, url: {url}")

    # Extract audio manually with ffmpeg — bypasses postprocessor entirely
    try:
        audio_path = await loop.run_in_executor(
            None, _extract_audio, video_path, temp_dir
        )
    except Exception as e:
        _cleanup_dir(temp_dir)
        raise RuntimeError(f"Audio extraction failed: {e}") from e

    # Find thumbnail
    thumbnail_path = _find_thumbnail(temp_dir, base_name)

    # Sanitise metadata strings — never trust external data
    raw_title = info.get("title") or ""
    raw_description = info.get("description") or ""

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