import json
import logging
import re

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings

HASHTAG_PATTERN = re.compile(r'#\w+')

logger = logging.getLogger(__name__)

OPEN_AI_SYSTEM_PROMPT = """
### ROLE

You are a Knowledge Extraction Engine.
Your job is NOT to summarize.
Your job is to convert transcripts and captions into structured reference notes that maximize information retention while improving readability.

Success = preserving information.
Failure = losing information through summarization.

---
### OUTPUT
Return VALID JSON ONLY.
{
  "title": "specific descriptive title",
  "content": "markdown formatted reference note"
}
---

### CORE RULES

1. DO NOT SUMMARIZE
- Preserve information.
- Reorganize information.
- Improve readability.
- Never compress multiple meaningful points into one vague statement.

2. PRESERVE ALL NAMED ENTITIES
- Preserve all tool names, products, apps, websites, brands, frameworks, APIs, companies, books, people, and methods.
- Never replace a specific name with a generic description.

Bad:
- AI resume builder

Good:
- Teal AI Resume Builder

3. PRESERVE ALL NUMBERS
Never omit:
- percentages
- prices
- dates
- quantities
- durations
- temperatures
- rankings
- measurements
- version numbers

4. PRESERVE PROCEDURES
If the content contains a workflow, tutorial, recipe, strategy, or guide:
Retain every meaningful step.

Bad:
- Optimize resume using AI

Good:
1. Open Teal AI Resume Builder
2. Import resume from LinkedIn or upload a file
3. Review Job Matcher results
4. Add missing keywords using Write with AI
5. Re-check match score

5. CAPTURE SOFTWARE WORKFLOWS
When software is demonstrated, structure as:

### Tool
### Workflow
### Results
### Pricing

Include button names, menu names, features, inputs, outputs, and outcomes.

6. USE CAPTION AS SOURCE OF TRUTH
Use the caption for names, spellings, and terminology.
Use the transcript for workflow and details.

---

### CONTENT FILTERING

Remove:
- greetings
- filler conversation
- repeated statements
- engagement bait
- follow/comment/share prompts
- sponsor disclaimers
- hashtags
- emojis

Keep:
- facts
- instructions
- opinions with informational value
- comparisons
- results
- examples

---

### STRUCTURING RULES

Use Markdown.
Use headings when useful.
For rankings:
- Convert descending rankings into ascending order.
- Place the winner first.

For tutorials:
- Organize into logical sections.
- Preserve every step.

Prefer extraction over summarization.
If unsure whether information is important, KEEP IT.

---
### JSON SAFETY
Return valid JSON only.
Escape newlines as \\n.
Do not include markdown code fences.
"""

class ExtractionResult:
    def __init__(
        self,
        title: str,
        content: str,
    ):
        self.title = title
        self.content = content


def remove_hashtags(text: str) -> str:
    return re.sub(r'#\S+', '', text)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=30),
    reraise=True,
)
async def extract_knowledge(
    transcript: str,
    title_hint: str | None = None,
    description_hint: str | None = None,
) -> ExtractionResult:
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
    safe_description = (remove_hashtags(description_hint) or "")

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
    logger.info(f"Extraction complete — title: {result.title}")
    return result

def _parse_and_validate_response(raw: str) -> ExtractionResult:
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
        content = re.search(r'"content"\s*:\s*"(.*?)"(?=\s*,\s*"(?:key_points|action_items|tags)")', cleaned, re.DOTALL)

        return ExtractionResult(
            title=title.group(1) if title else "Untitled Reel",
            content=content.group(1).replace('\\"', '"') if content else ""
        )
    except Exception as e:
        logger.error(f"Field extraction also failed: {e}")
        return _fallback_result()


def _build_result(data: dict) -> ExtractionResult:
    """Build and validate a ExtractionResult from a parsed dict."""
    if not isinstance(data, dict):
        logger.error("Claude returned non-dict JSON")
        return _fallback_result()

    title = _safe_str(data.get("title"), max_len=100, fallback="Untitled Reel")
    content = _safe_str(data.get("content"), max_len=2000, fallback="")
    # key_points = _safe_str_list(data.get("key_points"), max_items=8, max_item_len=150)
    # action_items = _safe_str_list(data.get("action_items"), max_items=5, max_item_len=150)

    return ExtractionResult(
        title=title,
        content=content,
        # key_points=key_points,
        # action_items=action_items,
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


def _fallback_result() -> ExtractionResult:
    return ExtractionResult(
        title="Untitled Reel",
        content="",
    )