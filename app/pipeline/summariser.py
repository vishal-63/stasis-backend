import json
import logging
import re

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings

logger = logging.getLogger(__name__)

OPEN_AI_SYSTEM_PROMPT = """
### ROLE
You are a 'High-Fidelity' Information Architect. Your task is to transform raw social media transcripts and captions into dense, professional reference notes. You prioritize data retention over brevity.

### OUTPUT FORMAT
You must respond with VALID JSON ONLY. Do not include markdown code fences (```json), preambles, or post-scripts.
{
  "title": "string, specific and descriptive",
  "summary": "string, uses Markdown for hierarchy. Preserve all technical data.",
  "tags": ["array of 3-6 lowercase specific tags"]
}

### DATA RETENTION & PRECISION RULES
1. ZERO-LOSS METRICS: You are STRICTLY PROHIBITED from omitting numerical data. You must retain every temperature (e.g., 375°F), duration (e.g., 25 mins), ratio (e.g., 1:1), and price mentioned.
2. ENTITY FIDELITY: Never generalize. If the text says 'Sony ZV-E10', do not write 'camera'. If a brand is phonetically misspelled in the transcript but clear in context (e.g., 'Tors Enfan' -> 'TVS Apache'), use the correct technical name.
3. SOURCE FUSION: Use the Instagram Caption as the 'Source of Truth' for spellings and names. Use the Transcript for procedural flow.

### STRUCTURAL LOGIC (MANDATORY)
1. RANKINGS/LISTS: 
   - If a ranking is detected (5th to 1st), you MUST REVERSE it to Ascending Order (1 to 5).
   - The 'Winner' (1st Place) must ALWAYS be at the top of the list.
   - Use the format: '1. **Name**'. 
   - NO REDUNDANT DESCRIPTIONS: If an item is numbered, you are STRICTLY PROHIBITED from writing "Ranked X" or "Number X" in the description. Only append details if substantive facts are mentioned (e.g., '1. **Name** - Won 3 Oscars'). If no facts exist, output just the numbered name.
2. PROCEDURES (RECIPES/TUTORIALS):
   - Divide into logical phases using '###' headers (e.g., ### Preparation, ### Assembly).
   - Include 'The Last Mile': Resting times, garnishes, and finishing touches.
3. SENTIMENT: 
   - Include specific qualitative quotes in a 'Personal Verdict' section at the end ONLY if they add value. 

### NOISE REDUCTION & FILTERING
- BAN ON BANTER: Completely ignore conversational filler, guessing games, wrong guesses, hints, and redundant statements (e.g., ignore a host saying "Is it Titanic? No."). Extract ONLY the final, factual data points.
- STRICT BAN ON ENGAGEMENT BAIT: Completely omit any marketing calls to action. You MUST NOT include instructions like "comment below," "link in bio," "save this reel," "follow for more," or any prompts asking the viewer to subscribe or get a resource.
- NO META-COMMENTARY: Do not create "Additional Notes" to describe the flow of the conversation or the fact that people were guessing.
- STRICT BAN ON HASHTAGS: Do not include any hashtags (#) in your response. 
- FORMATTING ONLY: Use proper English and Markdown for emphasis. Never use social media jargon or symbols (e.g., emojis, @mentions, or trending tags).

### JSON SAFETY PROTOCOL
- Use ONLY single quotes (' ') for any internal dialogue or quotes within the text.
- Never use unescaped double quotes (" ") inside the JSON string values.
- Ensure all newlines are escaped as '\\n'.

### EXAMPLE OF EXPECTED DENSITY
Input: "Bake at 375 for 25 mins covered then 5 mins uncovered."
Output: "### Baking Instructions\\n* **Initial Bake:** 375°F for 25 minutes (Covered).\\n* **Final Crisp:** 5 minutes (Uncovered)."
"""

class SummaryResult:
    def __init__(
        self,
        title: str,
        summary: str,
        # key_points: list[str],
        # action_items: list[str],
        tags: list[str],
    ):
        self.title = title
        self.summary = summary
        # self.key_points = key_points
        # self.action_items = action_items
        self.tags = tags


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=30),
    reraise=True,
)
async def summarise_transcript(
    transcript: str,
    title_hint: str | None = None,
    description_hint: str | None = None,
) -> SummaryResult:
    """
    Use OpenAI gpt-4o-mini to extract structured note content from a transcript.

    Security considerations:
    - Transcript is passed as user content, not interpolated into the system prompt
    - Response is strictly parsed as JSON — no eval, no exec
    - All output fields are validated and clamped to safe lengths
    - API key read from settings only
    """
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    # Clamp inputs to prevent token abuse
    safe_transcript = (transcript or "")[:15_000]
    safe_title_hint = (title_hint or "")[:200]
    safe_description = (description_hint or "")

    # Build user message — external content is clearly delimited
    # and placed in user turn, never interpolated into system prompt
    user_message = (
        f"Transcript:\n<transcript>\n{safe_transcript}\n</transcript>\n\n"
        f"Reel title (if available): {safe_title_hint}\n"
        f"Reel description (if available): {safe_description}"
    )

    response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": OPEN_AI_SYSTEM_PROMPT
                },
                {
                    "role": "user", 
                    "content": f"INPUT DATA:\nDescription/Caption: {safe_description}\nAudio Transcript: {safe_transcript if safe_transcript else 'N/A'}"
                }
            ],
            temperature=0,                   # Zero randomness
            seed=42,
            response_format={"type": "json_object"}
        )

    raw = json.loads(response.choices[0].message.content)

    # result = _parse_and_validate_response(raw)
    result = _build_result(raw)
    logger.info(f"Summarisation complete — title: {result.title}, tags: {result.tags}")
    return result

def _parse_and_validate_response(raw: str) -> SummaryResult:
    # Strip markdown fences
    cleaned = re.sub(r"^```(?:json)?\n?", "", raw).rstrip("`").strip()

    # Attempt 1: parse as-is
    try:
        data = json.loads(cleaned)
        return _build_result(data)
    except json.JSONDecodeError:
        pass

    # Attempt 2: fix unescaped quotes inside string values
    # Replace smart/curly quotes with escaped straight quotes
    fixed = cleaned
    fixed = fixed.replace("\u201c", '\\"').replace("\u201d", '\\"')  # curly quotes
    fixed = fixed.replace("\u2018", "'").replace("\u2019", "'")       # curly apostrophes

    try:
        data = json.loads(fixed)
        return _build_result(data)
    except json.JSONDecodeError:
        pass

    # Attempt 3: use regex to extract fields individually as last resort
    logger.warning("JSON parsing failed twice, attempting field extraction")
    try:
        title = re.search(r'"title"\s*:\s*"([^"]*)"', cleaned)
        summary = re.search(r'"summary"\s*:\s*"(.*?)"(?=\s*,\s*"(?:key_points|action_items|tags)")', cleaned, re.DOTALL)
        tags_match = re.search(r'"tags"\s*:\s*\[([^\]]*)\]', cleaned)

        extracted_tags = []
        if tags_match:
            extracted_tags = [
                t.strip().strip('"')
                for t in tags_match.group(1).split(",")
                if t.strip().strip('"')
            ]

        return SummaryResult(
            title=title.group(1) if title else "Untitled Reel",
            summary=summary.group(1).replace('\\"', '"') if summary else "",
            # key_points=[],
            # action_items=[],
            tags=extracted_tags[:5],
        )
    except Exception as e:
        logger.error(f"Field extraction also failed: {e}")
        return _fallback_result()


def _build_result(data: dict) -> SummaryResult:
    """Build and validate a SummaryResult from a parsed dict."""
    if not isinstance(data, dict):
        logger.error("Claude returned non-dict JSON")
        return _fallback_result()

    title = _safe_str(data.get("title"), max_len=100, fallback="Untitled Reel")
    summary = _safe_str(data.get("summary"), max_len=2000, fallback="")
    # key_points = _safe_str_list(data.get("key_points"), max_items=8, max_item_len=150)
    # action_items = _safe_str_list(data.get("action_items"), max_items=5, max_item_len=150)
    tags = _safe_str_list(data.get("tags"), max_items=5, max_item_len=50)

    tags = [re.sub(r"[^a-z0-9-]", "", t.lower()) for t in tags]
    tags = [t for t in tags if t]

    return SummaryResult(
        title=title,
        summary=summary,
        # key_points=key_points,
        # action_items=action_items,
        tags=tags,
    )

def _safe_str(value: any, max_len: int, fallback: str = "") -> str:
    if not isinstance(value, str):
        return fallback
    return value.strip()[:max_len]


def _safe_str_list(
    value: any,
    max_items: int,
    max_item_len: int,
) -> list[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value[:max_items]:
        if isinstance(item, str) and item.strip():
            result.append(item.strip()[:max_item_len])
    return result


def _fallback_result() -> SummaryResult:
    return SummaryResult(
        title="Untitled Reel",
        summary="",
        # key_points=[],
        # action_items=[],
        tags=[],
    )