import logging
import re
from urllib.parse import urlparse

from fastapi import HTTPException, Request, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import get_settings

logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)

ALLOWED_DOWNLOAD_DOMAINS = frozenset([
    "www.instagram.com",
    "instagram.com",
    "www.youtube.com",
    "youtube.com",
    "youtu.be",
    "m.youtube.com", 
])

INSTAGRAM_REEL_PATTERN = re.compile(
    r"^https://(?:www\.)?instagram\.com/reel/([A-Za-z0-9_-]{1,30})/?$"
)

YOUTUBE_SHORT_PATTERN = re.compile(
    r"^https://(?:(?:www|m)\.)?(?:youtube\.com/shorts/|youtu\.be/)([A-Za-z0-9_-]{1,20})(?:[?#].*)?$"
)

MAX_URL_LENGTH = 512


def validate_reel_url(url: str) -> str:
    """
    Validate and normalise an Instagram reel URL.
    Returns the cleaned URL or raises HTTPException.

    Security considerations:
    - Length check prevents memory exhaustion
    - Strict regex prevents SSRF via crafted URLs
    - Domain allowlist prevents downloading from arbitrary hosts
    - Fragment and query params stripped to avoid cache-busting tricks
    """
    if not url or not isinstance(url, str):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="URL is required",
        )

    url = url.strip()

    if len(url) > MAX_URL_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="URL exceeds maximum length",
        )

    match = None
    clean_url = None

    if INSTAGRAM_REEL_PATTERN.match(url):
        match = INSTAGRAM_REEL_PATTERN.match(url)
        clean_url = f"https://www.instagram.com/reel/{match.group(1)}/"
    elif YOUTUBE_SHORT_PATTERN.match(url):
        match = YOUTUBE_SHORT_PATTERN.match(url)
        clean_url = f"https://www.youtube.com/shorts/{match.group(1)}"

    if not match:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="URL must be a valid Instagram Reel URL",
        )

    # Parse and validate domain (defence in depth against regex bypass)
    try:
        parsed = urlparse(url)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Malformed URL",
        )

    if parsed.scheme != "https":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="URL must use HTTPS",
        )

    if parsed.netloc not in ALLOWED_DOWNLOAD_DOMAINS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="URL domain not permitted",
        )

    logger.info(f"URL validated: {clean_url}")
    return clean_url


def validate_note_id(note_id: str) -> str:
    """Validate note_id is a UUID to prevent injection."""
    uuid_pattern = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
        re.IGNORECASE,
    )
    if not uuid_pattern.match(note_id):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid note ID format",
        )
    return note_id


def check_api_secret(request: Request) -> None:
    """
    Verify the shared API secret header sent by the mobile app.
    This is a second layer on top of JWT auth — both must pass.
    Prevents random internet traffic from hitting the API.
    """
    settings = get_settings()
    secret = request.headers.get("X-API-Secret", "")
    if not secret or secret != settings.api_secret_header:
        logger.warning(
            f"Invalid API secret from {request.client.host if request.client else 'unknown'}"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )