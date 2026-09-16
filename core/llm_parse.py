"""Shared utilities for parsing LLM text output.

All structured-output parsing (JSON, code blocks, numbered lists) should
use these helpers so that repair heuristics, error messages, and fallback
behaviour are consistent across agents.
"""

import json
import re
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# JSON extraction + repair
# ---------------------------------------------------------------------------

# Common Python-to-JSON fixups applied BEFORE json.loads
_JSON_FIXUPS = [
    # Python booleans / None / NaN / Infinity → JSON equivalents
    (re.compile(r"\bTrue\b"), "true"),
    (re.compile(r"\bFalse\b"), "false"),
    (re.compile(r"\bNone\b"), "null"),
    (re.compile(r"\bNaN\b"), "null"),
    (re.compile(r"\bInfinity\b"), "1e308"),
    (re.compile(r"\b-Infinity\b"), "-1e308"),
]

# Trailing commas before } or ]  (e.g.  {"a": 1,} )
_TRAILING_COMMA = re.compile(r",\s*([}\]])")

# Single-line // comments (not inside strings — best-effort)
_LINE_COMMENT = re.compile(r"//[^\n]*")


def _repair_json_string(raw: str) -> str:
    """Best-effort repair of common LLM JSON quirks."""
    s = raw

    # 1. Strip single-line comments
    s = _LINE_COMMENT.sub("", s)

    # 2. Python literals → JSON
    for pat, repl in _JSON_FIXUPS:
        s = pat.sub(repl, s)

    # 3. Trailing commas
    s = _TRAILING_COMMA.sub(r"\1", s)

    # 4. Single quotes → double quotes (simple heuristic: only outside strings)
    #    This handles the common case {'key': 'value'} but won't handle
    #    nested quotes like {"key": "it's fine"}.  We try json.loads first
    #    with the original, and only apply this if it fails.
    return s


def _extract_braces(text: str) -> Optional[str]:
    """Find the outermost { … } substring, handling nesting."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    # Unbalanced — return from first { to end (let json.loads report the error)
    return text[start:]


def extract_json(text: str) -> Dict[str, Any]:
    """Extract a JSON object from LLM text, with repair heuristics.

    Strategy:
      1. Strip markdown fences (```json … ```)
      2. Find outermost { … }
      3. Try json.loads as-is
      4. On failure, apply repair heuristics and retry
      5. On failure, try replacing single quotes with double quotes and retry

    Raises ``ValueError`` with a descriptive message on failure.
    """
    if not text or not text.strip():
        raise ValueError("Empty LLM response — expected JSON object")

    # Strip markdown code fences
    cleaned = text.strip()
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL)
    if m:
        cleaned = m.group(1).strip()

    # Find outermost braces
    candidate = _extract_braces(cleaned)
    if candidate is None:
        raise ValueError(
            f"No JSON object found in LLM response (no '{{' character). "
            f"Response starts with: {text[:120]!r}"
        )

    # Attempt 1: parse as-is
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Attempt 2: repair common quirks
    repaired = _repair_json_string(candidate)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Attempt 3: single quotes → double quotes (aggressive)
    sq_fixed = repaired.replace("'", '"')
    try:
        return json.loads(sq_fixed)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse JSON from LLM response after repair attempts. "
            f"json.JSONDecodeError: {exc}. "
            f"Candidate text: {candidate[:200]!r}"
        ) from exc


def extract_json_or(text: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    """Like :func:`extract_json` but returns *fallback* on failure instead of raising."""
    try:
        return extract_json(text)
    except (ValueError, TypeError):
        return dict(fallback)


# ---------------------------------------------------------------------------
# Code-block extraction
# ---------------------------------------------------------------------------

def extract_code_block(text: str, *, min_length: int = 50) -> str:
    """Extract a code block from LLM text.

    Strategy:
      1. Look for ``` python … ``` fenced block
      2. Fall back to ``` … ``` (any language tag)
      3. Fall back to raw text (stripped)

    Raises ``ValueError`` if the result is shorter than *min_length*.
    """
    if not text or not text.strip():
        raise ValueError("Empty LLM response — expected code block")

    # Try python-specific fence first
    m = re.search(r"```python\s*\n(.*?)\n\s*```", text, re.DOTALL)
    if m:
        code = m.group(1).strip()
        if len(code) >= min_length:
            return code

    # Try generic fence
    m = re.search(r"```\w*\s*\n(.*?)\n\s*```", text, re.DOTALL)
    if m:
        code = m.group(1).strip()
        if len(code) >= min_length:
            return code

    # Fall back to raw text
    code = text.strip()
    if len(code) < min_length:
        raise ValueError(
            f"Extracted code is too short ({len(code)} chars, min {min_length}). "
            f"Response starts with: {text[:120]!r}"
        )
    return code


# ---------------------------------------------------------------------------
# Numbered / bulleted list extraction
# ---------------------------------------------------------------------------

_NUMBERED = re.compile(r"^\d+[\.\)]\s*(.+)$", re.MULTILINE)
_BULLETED = re.compile(r"^[-*•]\s*(.+)$", re.MULTILINE)
_SKIP_PREFIXES = (
    "note:", "format:", "here", "the following", "below",
    "sure", "certainly", "of course",
)


def extract_list(text: str, max_items: int = 0) -> List[str]:
    """Extract a list of items from LLM text.

    Tries numbered list first, then bullets, then plain lines.
    Filters out instruction / meta lines.
    """
    if not text or not text.strip():
        return []

    # Stage 1: numbered list
    items = _NUMBERED.findall(text)
    if items:
        items = [s.strip() for s in items if s.strip()]
        if max_items > 0:
            items = items[:max_items]
        return items

    # Stage 2: bullet list
    items = _BULLETED.findall(text)
    if items:
        items = [s.strip() for s in items if s.strip()]
        if max_items > 0:
            items = items[:max_items]
        return items

    # Stage 3: plain lines (filter meta-text)
    items = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if any(line.lower().startswith(p) for p in _SKIP_PREFIXES):
            continue
        if len(line) < 10:
            continue
        items.append(line)

    if max_items > 0:
        items = items[:max_items]
    return items
