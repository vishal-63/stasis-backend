import asyncio
import logging
import os
import re
import tempfile
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

def get_js_runtime() -> dict:
    """Find available JS runtime for yt-dlp."""
    node_path = shutil.which("node")
    if node_path:
        logger.info(f"Found Node.js at: {node_path}")
        return {"node": {"path": node_path}}

    deno_path = shutil.which("deno")
    if deno_path:
        logger.info(f"Found Deno at: {deno_path}")
        return {"deno": {"path": deno_path}}

    logger.warning("No JS runtime found — YouTube extraction may fail")
    return {}

# ydl_opts = {
#         "format": "bestaudio[ext=m4a]/bestaudio[ext=mp4]/bestaudio/best",
#         "outtmpl": output_template,
#         "writethumbnail": True,
#         "nocheckcertificate": False,
#         "quiet": True,
#         "no_warnings": False,

#         "cookiefile": cookies_path,

#         "postprocessors": [{
#             "key": "FFmpegExtractAudio",
#             "preferredcodec": "mp3",
#             "preferredquality": "64",  # lower quality = smaller file = faster
#             "nopostoverwrites": False,
#         }],

#         "postprocessor_args": {
#             "ffmpeg": ["-hide_banner", "-loglevel", "error"],
#         },

#         "filesize_max": 50 * 1024 * 1024,
#         "allowed_extractors": ["instagram", "youtube", "YoutubeIE", "YoutubeShorts"],
#         "external_downloader": None,

#         # Limit download speed check — abort if too slow
#         "socket_timeout": 30,
#         "js_runtimes": get_js_runtime(),

#         "merge_output_format": "mp4",
#         "keepvideo": False,
#     }

def get_ydl_opts(url: str, output_template: str, ig_cookies_path: str, yt_cookies_path: str) -> dict:
    # 1. Base options shared across all platforms
    base_opts = {
        "outtmpl": output_template,
        "writethumbnail": True,
        "nocheckcertificate": False,
        "quiet": True,
        "no_warnings": False,

        # Forces extraction to mp3 at 64kbps regardless of what format is downloaded
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "64",
            "nopostoverwrites": False,
        }],

        "filesize_max": 50 * 1024 * 1024,
        "socket_timeout": 30,
        "keepvideo": False,
    }

    # 2. YouTube-specific configuration
    if "youtube.com" in url or "youtu.be" in url:
        return {
            **base_opts,
            # Permissive format string is required here because mobile client spoofing 
            # often breaks strict format requests like "bestaudio[ext=m4a]"

            "format": "ba/b",
            "cookiefile": yt_cookies_path,
            "allowed_extractors": ["youtube", "YoutubeIE", "YoutubeShorts"],
            "js_runtimes": get_js_runtime(),
        
            "cachedir": "/tmp/yt-dlp-cache",
            
            "extractor_args": {
                "youtube": {
                    "player_client": ["tv", "web"] # TV client is less strict on bot detection
                }
            }
        }

    # 3. Instagram-specific configuration
    elif "instagram.com" in url:
        return {
            **base_opts,
            "format": "bestaudio[ext=m4a]/bestaudio[ext=mp4]/bestaudio/best",
            "cookiefile": ig_cookies_path,
            "allowed_extractors": ["instagram"],
        }

    # Fallback for unexpected URLs
    return base_opts

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

    # Create isolated temp directory for this download
    # Use a random name to prevent path traversal / prediction
    temp_dir = tempfile.mkdtemp(prefix=f"reel_{uuid.uuid4().hex}_")
    output_template = os.path.join(temp_dir, "audio.%(ext)s")

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

    yt_cookies_path = None
    if settings.yt_cookies_path:
        source = Path(settings.yt_cookies_path)
        if source.exists():
            yt_cookies_path = os.path.join(temp_dir, "yt_cookies.txt")
            shutil.copy2(str(source), yt_cookies_path)
            os.chmod(yt_cookies_path, 0o600)
            logger.info(f"Copied cookies to writable path: {yt_cookies_path}")
        else:
            logger.warning(f"Cookies file not found: {settings.yt_cookies_path}")

    ydl_opts = get_ydl_opts(url, output_template, ig_cookies_path, yt_cookies_path)

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

    logger.info(f"Download complete — duration: {duration}s, audio: {audio_path}, thumbnail: {thumbnail_path}")
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