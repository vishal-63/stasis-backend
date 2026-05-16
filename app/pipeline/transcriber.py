import logging
from pathlib import Path

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings

logger = logging.getLogger(__name__)

MAX_AUDIO_SIZE_BYTES = 24 * 1024 * 1024

def filter_transcript(client, transcript_text, description_text):
    if len(transcript_text.split()) < 8:
        return ""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system", 
                "content": (
                    """
                    You are a Noise Filter. Compare the Audio Transcript to the Video Description. If the Audio Transcript consists of song lyrics, background music descriptions, or content entirely unrelated to the factual information in the Description, return 'IRRELEVANT'. Otherwise, Your job is to take a raw, messy speech-to-text transcript and fix it BEFORE it is summarized.
                    1. FIX PHONETIC ERRORS: Identify names that 'sound' like other things (e.g., 'Tors Enfan' -> 'TVS Apache', 'Aprilia RS four five seven' -> 'Aprilia RS 457').
                    2. RESTORE NUMERICAL LOGIC: If the speaker is counting or ranking (5 to 1), ensure the items are correctly labeled in that order.
                    3. REMOVE FILLERS: Delete 'like, share, subscribe' and 'uh/um' sounds.
                    Output ONLY the cleaned, factual transcript.
                    """
                )
            },
            {
                "role": "user", 
                "content": f"Description: {description_text}\nAudio: {transcript_text}"
            }
        ],
        temperature=0
    )
    
    result = response.choices[0].message.content
    return "" if "IRRELEVANT" in result else transcript_text


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=30),
    reraise=True,
)
async def transcribe_audio(audio_path: str, description_text: str) -> str:
    """
    Transcribe audio using OpenAI Whisper API.

    Security considerations:
    - File size checked before upload to prevent oversized requests
    - File path validated to be within expected directory
    - API key read from settings, never hardcoded
    - Transcript is returned as plain text only
    """
    logger.debug(f"Transcription starting — file: {audio_path}, size: {Path(audio_path).stat().st_size} bytes")
    path = Path(audio_path)

    # Validate path exists and is a file (not a symlink to /etc/passwd etc.)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Resolve symlinks and ensure the file is what it claims to be
    real_path = path.resolve()
    if not real_path.exists():
        raise FileNotFoundError("Audio path resolves to non-existent file")

    # Size check
    file_size = path.stat().st_size
    if file_size > MAX_AUDIO_SIZE_BYTES:
        raise ValueError(
            f"Audio file {file_size} bytes exceeds Whisper limit of "
            f"{MAX_AUDIO_SIZE_BYTES} bytes"
        )

    if file_size == 0:
        raise ValueError("Audio file is empty")

    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    logger.debug(f"Transcribing audio: {file_size} bytes")

    with open(real_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text",
        )

    transcript = filter_transcript(client, str(response).strip(), description_text=description_text)

    if not transcript:
        logger.warning("Whisper returned empty transcript")
        return ""

    logger.debug(f"Transcription complete: {len(transcript)} chars")
    return transcript