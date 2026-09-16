"""
Variable catalog + lightweight retrieval for wide survey datasets (e.g., WVS).

Goal:
- Build a reproducible, deterministic "variable card" per column from a codebook
- Retrieve only a small subset of relevant variables per round to keep LLM prompts small

This intentionally uses TF-IDF (sklearn) as an always-available baseline.
You can later swap/augment with embeddings + FAISS if needed.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import csv
import json
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


_VAR_CODE_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,30}$")
_OPTION_RE = re.compile(r"^\s*(-?\d+)\s*(?:[-.]*\s*){1,4}(.+?)\s*$")
_OPTION_LIKE_START_RE = re.compile(r"^\s*-?\d+\s*(?:[\-–\.\):]\s*){1,4}\S+")
# Q-range in concept menu / theme string, e.g. "(Q1–Q45)" or "(Q1-Q56)" (en-dash or hyphen)
_Q_RANGE_RE = re.compile(r"\(Q(\d+)\s*[–\-]\s*Q?(\d+)\)", re.IGNORECASE)
# WVS question variable: Q<number>
_Q_VAR_RE = re.compile(r"^Q(\d+)$", re.IGNORECASE)
_KNOWN_SECTION_HEADERS = {
    "contents",
    "technical variables",
    "core variables",
    "wvs indexes",
    "contextual variables",
    "annex",
}
_AMBIGUOUS_THREE_LETTER_RE = re.compile(r"^[A-Z]{3}$")
_KNOWN_THREE_LETTER_VAR_CODES: set[str] = set()
_MISSING_TEXT_HINTS = ("missing", "not asked", "no answer", "don't know", "don´t know", "not available")
_COUNTRY_SPECIFIC_HINTS = (
    "country-specific",
    "country specific",
    "codes are available in annex",
    "codes are available in the annex",
    "list of codes in annex",
    "list of codes in the annex",
    "party preference",
)
_ANNEX_HINTS = ("annex", "list of codes", "country-specific list of codes")


def _strip_q_range_suffix(text: str) -> str:
    return re.sub(r"\s*\(Q\d+\s*[–\-]\s*Q?\d+\)\s*$", "", text or "", flags=re.IGNORECASE).strip()


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return slug or "unknown"


def _normalize_text_for_match(text: str) -> str:
    low = (text or "").lower().replace("’", "'").replace("‘", "'").replace("´", "'")
    low = re.sub(r"[^a-z0-9]+", " ", low)
    return re.sub(r"\s+", " ", low).strip()


def _clean_phrase(text: str, *, lowercase: bool = True) -> str:
    cleaned = _clean_line(text or "")
    cleaned = re.sub(r"^[\-–:;,\.\s]+", "", cleaned)
    cleaned = re.sub(r"[\-–:;,\.\s]+$", "", cleaned)
    return cleaned.lower() if lowercase else cleaned


def _truncate_text(text: str, max_len: int = 220) -> str:
    cleaned = _clean_line(text or "")
    return cleaned if len(cleaned) <= max_len else cleaned[: max_len - 3].rstrip() + "..."


def _truncate_glossary_text(text: str, max_len: int = 160) -> str:
    cleaned = " ".join(str(text or "").split())
    return cleaned if len(cleaned) <= max_len else cleaned[: max_len - 3].rstrip() + "..."


def _non_missing_options(options: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for opt in options or []:
        text = _clean_line(str(opt.get("text", "") or ""))
        code = opt.get("code")
        if isinstance(code, (int, float)) and int(code) < 0:
            continue
        if any(h in text.lower() for h in _MISSING_TEXT_HINTS):
            continue
        if text:
            out.append({"code": code, "text": text})
    return out


def _option_texts(options: Sequence[Dict[str, Any]]) -> List[str]:
    return [_normalize_text_for_match(str(o.get("text", "") or "")) for o in _non_missing_options(options)]


def _looks_country_specific_text(*parts: str) -> bool:
    text = " ".join([_normalize_text_for_match(p) for p in parts if p])
    return any(h in text for h in _COUNTRY_SPECIFIC_HINTS)


def normalize_module_to_section_theme(module: str) -> str:
    low = _normalize_text_for_match(_strip_q_range_suffix(module))
    mapping = [
        ("social capital trust and organizational membership", "social_capital_trust"),
        ("happiness and wellbeing", "wellbeing"),
        ("social values norms stereotypes", "social_values_norms"),
        ("economic values", "economic_values"),
        ("perceptions of corruption", "corruption"),
        ("perceptions of migration", "migration"),
        ("perceptions of security", "security"),
        ("postmaterialism value change", "value_change"),
        ("science and technology perceptions", "science_technology"),
        ("religious values", "religious_values"),
        ("ethical values", "ethical_values"),
        ("political interest and political participation", "political_participation"),
        ("political culture and political regimes", "political_culture_regimes"),
        ("demographics and ses", "demographics_ses"),
        ("technical variables", "technical"),
        ("wvs indexes", "index"),
        ("contextual variables", "contextual"),
        ("annex", "annex"),
    ]
    for needle, value in mapping:
        if needle in low:
            return value
    return _slugify(_strip_q_range_suffix(module))


def infer_scale_id(options: List[Dict[str, Any]], label: str, question: str) -> str:
    option_texts = _option_texts(options)
    context = " ".join([_normalize_text_for_match(label), _normalize_text_for_match(question)])

    if _looks_country_specific_text(label, question):
        return "country_specific_codes"
    if not option_texts:
        if "numeric variable" in context or "years old" in context or "point scale" in context:
            return "numeric_continuous"
        return "open_text_or_unlisted_codes"
    if len(option_texts) >= 8 and all(t.isdigit() for t in option_texts[:10]):
        return "numeric_continuous"
    if option_texts[:4] == [
        "trust completely",
        "trust somewhat",
        "do not trust very much",
        "do not trust at all",
    ]:
        return "trust_4pt"
    if option_texts[:4] == ["a great deal", "quite a lot", "not very much", "none at all"]:
        return "confidence_4pt"
    if len(option_texts) >= 4 and option_texts[0] in {"very often", "often", "fairly often"}:
        last = option_texts[-1]
        mid = " ".join(option_texts[1:3])
        if "never" in last or "not at all" in last:
            if "sometimes" in mid or "rarely" in mid or "not often" in mid:
                return "freq_often_never_4pt"
    if option_texts[:3] == ["have done", "might do", "would never do"]:
        return "done_might_never_3pt"
    if option_texts[:4] == ["strongly agree", "agree", "disagree", "strongly disagree"]:
        return "agree_4pt"
    if option_texts[:2] in (["yes", "no"], ["mentioned", "not mentioned"]):
        return "binary_yes_no"
    if option_texts[:3] == ["not a member", "inactive member", "active member"]:
        return "membership_3pt"
    if option_texts[:4] == ["very important", "rather important", "not very important", "not at all important"]:
        return "importance_4pt"
    if option_texts[:3] == ["lower", "middle", "higher"]:
        return "education_3pt"
    if len(option_texts) >= 2:
        return "ordinal_categorical"
    return "categorical"


def scale_id_to_summary(scale_id: str, options: List[Dict[str, Any]]) -> str:
    mapping = {
        "trust_4pt": "trust completely -> not at all",
        "confidence_4pt": "a great deal -> none at all",
        "freq_often_never_4pt": "often -> never",
        "done_might_never_3pt": "have done / might do / would never do",
        "agree_4pt": "strongly agree -> strongly disagree",
        "binary_yes_no": "yes / no",
        "membership_3pt": "not a member / inactive member / active member",
        "importance_4pt": "very important -> not at all important",
        "education_3pt": "lower / middle / higher",
        "country_specific_codes": "country-specific annex codes",
        "numeric_continuous": "numeric or continuous scale",
        "external_index": "external or derived index",
        "open_text_or_unlisted_codes": "open text or annex-coded values",
    }
    if scale_id in mapping:
        return mapping[scale_id]

    option_texts = [str(o.get("text", "") or "") for o in _non_missing_options(options)]
    if len(option_texts) >= 2:
        return f"{_clean_phrase(option_texts[0])} -> {_clean_phrase(option_texts[-1])}"
    if option_texts:
        return _clean_phrase(option_texts[0])
    return ""


def _extract_item_from_suffix(label: str, question: str, prefixes: Sequence[str]) -> str:
    label_clean = _clean_line(label or "")
    for prefix in prefixes:
        low_prefix = prefix.lower()
        if label_clean.lower().startswith(low_prefix):
            return _clean_phrase(label_clean[len(prefix):])
    if ":" in label_clean:
        return _clean_phrase(label_clean.split(":", 1)[1])
    q_clean = _clean_line(question or "")
    if "?" in q_clean:
        tail = q_clean.split("?")[-1]
        return _clean_phrase(tail)
    return _clean_phrase(label_clean)


def infer_battery_structure(label: str, question: str, module: str) -> Tuple[str, str, str]:
    label_low = _normalize_text_for_match(label)
    question_low = _normalize_text_for_match(question)
    module_low = _normalize_text_for_match(module)

    if (
        "how much you trust people from various groups" in question_low
        or label_low.startswith("how much you trust")
        or label_low.startswith("trust ")
        or label_low.startswith("trust:")
    ):
        return (
            "trust_people_groups",
            "trust toward [group]",
            _extract_item_from_suffix(label, question, ["How much you trust: ", "Trust: "]),
        )

    if (
        "name a number of organizations" in question_low
        and "confidence" in question_low
    ) or label_low.startswith("confidence:"):
        return (
            "confidence_in_institutions",
            "confidence in [institution]",
            _extract_item_from_suffix(label, question, ["Confidence: "]),
        )

    if "gone without" in question_low and "last twelve months" in question_low:
        return (
            "deprivation_last12m",
            "frequency of [deprivation] in last 12 months",
            _extract_item_from_suffix(label, question, []),
        )

    if "forms of political action and social activism" in question_low:
        item = _extract_item_from_suffix(label, question, ["Social activism: ", "Political action: "])
        if any(k in f"{label_low} {question_low}" for k in ["electronic", "online", "internet", "social media"]):
            return ("political_actions_online", "online political action: [action]", item)
        return ("political_actions_general", "political action: [action]", item)

    if "voluntary organizations" in question_low and "member" in question_low:
        return (
            "organizational_membership",
            "membership in [organization]",
            _extract_item_from_suffix(label, question, ["Active/Inactive membership: ", "Membership: "]),
        )

    if "for each of the following aspects" in question_low and "important it is in your life" in question_low:
        return (
            "importance_in_life",
            "importance in life: [aspect]",
            _extract_item_from_suffix(label, question, ["Important in life: "]),
        )

    if "would not like to have as neighbors" in question_low:
        return (
            "undesired_neighbors",
            "would not like as neighbors: [group]",
            _extract_item_from_suffix(label, question, ["Neighbors: "]),
        )

    if "how often do the following things occur in this country s elections" in question_low or (
        "how often" in question_low and "elections" in question_low and "occur" in question_low
    ):
        return (
            "election_irregularities",
            "frequency of election issue: [issue]",
            _extract_item_from_suffix(label, question, []),
        )

    if "technical variables" in module_low or "contextual variables" in module_low or "wvs indexes" in module_low:
        return ("", "", "")

    return ("", "", _clean_phrase(label or question))


def _classify_variable_kind_from_fields(module: str, label: str, question: str) -> str:
    module_low = _normalize_text_for_match(module)
    label_low = _normalize_text_for_match(label)
    question_low = _normalize_text_for_match(question)
    joined = " ".join([module_low, label_low, question_low])

    if "technical variables" in module_low:
        return "technical"
    if "wvs indexes" in module_low or "recoded variable" in question_low or "index" in label_low:
        return "index"
    if "contextual variables" in module_low:
        return "contextual"
    if "annex" in module_low or any(h in joined for h in _ANNEX_HINTS):
        return "annex"
    if _looks_country_specific_text(label, question):
        return "country_specific"
    return "standard_survey_item"


def classify_variable_kind(card: "VariableCard") -> str:
    return _classify_variable_kind_from_fields(card.module, card.label, card.question)


def _build_semantic_summary_from_fields(
    *,
    var_name: str,
    label: str,
    question: str,
    variable_kind: str,
    battery_id: str,
    item_text: str,
) -> str:
    if variable_kind == "technical":
        return f"technical variable: {_clean_phrase(label or question)}"
    if variable_kind == "index":
        return f"index or recoded variable: {_clean_phrase(label or question)}"
    if variable_kind == "contextual":
        return f"contextual variable: {_clean_phrase(label or question)}"
    if variable_kind == "annex":
        return f"annex-coded variable: {_clean_phrase(label or question)}"
    if variable_kind == "country_specific":
        return f"country-specific coded variable: {_clean_phrase(label or question)}"

    item = _clean_phrase(item_text)
    if battery_id == "trust_people_groups":
        return f"trust toward {item}" if item else "trust toward social group"
    if battery_id == "confidence_in_institutions":
        return f"confidence in {item}" if item else "confidence in institution"
    if battery_id == "deprivation_last12m":
        return f"household deprivation: {item}" if item else "household deprivation in last 12 months"
    if battery_id == "political_actions_online":
        return f"online political action: {item}" if item else "online political action"
    if battery_id == "political_actions_general":
        return f"political action: {item}" if item else "political action"
    if battery_id == "organizational_membership":
        return f"organizational membership: {item}" if item else "organizational membership"
    if battery_id == "importance_in_life":
        return f"importance in life: {item}" if item else "importance in life"
    if battery_id == "undesired_neighbors":
        return f"neighbor exclusion: {item}" if item else "neighbor exclusion preference"
    if battery_id == "election_irregularities":
        return f"election irregularity perception: {item}" if item else "election irregularity perception"

    fallback = _clean_phrase(label or question)
    return fallback or _clean_phrase(var_name)


def build_semantic_summary(card: "VariableCard") -> str:
    return _build_semantic_summary_from_fields(
        var_name=card.var_name,
        label=card.label,
        question=card.question,
        variable_kind=card.variable_kind,
        battery_id=card.battery_id,
        item_text=card.item_text,
    )


def _infer_variable_abstraction(
    *,
    var_name: str,
    module: str,
    label: str,
    question: str,
    options: List[Dict[str, Any]],
    missing_codes: Dict[str, str],
) -> Dict[str, str]:
    del missing_codes  # reserved for future heuristics

    section_theme = normalize_module_to_section_theme(module)
    scale_id = infer_scale_id(options, label, question)
    battery_id, stem_template, item_text = infer_battery_structure(label, question, module)
    variable_kind = _classify_variable_kind_from_fields(module, label, question)
    scale_summary = scale_id_to_summary(scale_id, options)
    if variable_kind in {"technical", "contextual", "index"} and not scale_summary:
        scale_id = "external_index" if variable_kind in {"contextual", "index"} else "numeric_continuous"
        scale_summary = scale_id_to_summary(scale_id, options)
    return {
        "section_theme": section_theme,
        "variable_kind": variable_kind,
        "battery_id": battery_id,
        "stem_template": stem_template,
        "item_text": item_text,
        "scale_id": scale_id,
        "scale_summary": scale_summary,
        "semantic_summary": _build_semantic_summary_from_fields(
            var_name=var_name,
            label=label,
            question=question,
            variable_kind=variable_kind,
            battery_id=battery_id,
            item_text=item_text,
        ),
    }


def build_variable_glossary_detail(
    record: "VariableCard | Dict[str, Any]",
    *,
    use_generalized: bool = True,
    include_raw_question_fallback: bool = True,
    include_raw_options: bool = False,
) -> str:
    if isinstance(record, VariableCard):
        label = (record.label or "").strip()
        question = (record.question or "").strip()
        opts = record.options or []
        missing = record.missing_codes or {}
        variable_kind = (record.variable_kind or "").strip()
        stem_template = (record.stem_template or "").strip()
        item_text = (record.item_text or "").strip()
        scale_summary = (record.scale_summary or "").strip()
        semantic_summary = (record.semantic_summary or "").strip()
    else:
        label = (record.get("label") or "").strip()
        question = (record.get("question") or "").strip()
        opts = record.get("options") or []
        missing = record.get("missing_codes") or {}
        variable_kind = (record.get("variable_kind") or "").strip()
        stem_template = (record.get("stem_template") or "").strip()
        item_text = (record.get("item_text") or "").strip()
        scale_summary = (record.get("scale_summary") or "").strip()
        semantic_summary = (record.get("semantic_summary") or "").strip()

    if use_generalized:
        parts: List[str] = []
        if stem_template:
            parts.append(stem_template)
        if item_text and item_text.lower() not in stem_template.lower():
            parts.append(f"item={item_text}")
        if scale_summary:
            parts.append(f"scale={scale_summary}")
        if semantic_summary:
            parts.append(f"summary={semantic_summary}")
        if variable_kind and variable_kind != "standard_survey_item":
            parts.append(f"kind={variable_kind}")

        detail = " | ".join([p for p in parts if p])
        abstraction_is_weak = not semantic_summary or (not stem_template and not item_text)
        if abstraction_is_weak and include_raw_question_fallback:
            raw_parts: List[str] = []
            if label:
                raw_parts.append(f"raw_label={_truncate_glossary_text(label, 110)}")
            if question and question.lower() not in label.lower():
                raw_parts.append(f"raw_question={_truncate_glossary_text(question, 140)}")
            detail = " | ".join([p for p in [detail] + raw_parts if p])
        return detail

    chunk_parts = []
    if label:
        chunk_parts.append(label)
    if question and question.lower() not in label.lower():
        chunk_parts.append(question)

    extras: List[str] = []
    if include_raw_options and isinstance(opts, list) and opts and variable_kind not in {"country_specific", "annex"}:
        preview_items = []
        for o in opts[:6]:
            try:
                preview_items.append(f"{o.get('code')}={o.get('text')}")
            except Exception:
                continue
        if preview_items:
            extras.append("Options: " + "; ".join(preview_items))
    if isinstance(missing, dict) and missing:
        items = []
        for k, v in list(missing.items())[:6]:
            items.append(f"{k}={v}")
        if items:
            extras.append("Missing: " + "; ".join(items))

    detail = " ".join([p for p in chunk_parts if p])
    extra_text = " ".join(extras)
    if extra_text:
        detail = (detail + " " + extra_text).strip()
    return detail


def parse_q_range_from_theme(theme: str) -> Optional[Tuple[int, int]]:
    """
    Extract (low, high) Q-range from a theme string, e.g. "Happiness and wellbeing (Q46–Q56)" -> (46, 56).
    Handles en-dash and hyphen. Returns None if no range found.
    """
    if not theme or not theme.strip():
        return None
    m = _Q_RANGE_RE.search(theme.strip())
    if not m:
        return None
    try:
        lo, hi = int(m.group(1)), int(m.group(2))
        if lo <= hi:
            return (lo, hi)
    except (ValueError, IndexError):
        pass
    return None


def parse_module_from_theme(theme: str) -> Optional[str]:
    """Extract the module name from a concept menu item by stripping the trailing parenthetical.

    Works for any dataset:
      "Citation Impact (citation_count, C3, ...)" -> "Citation Impact"
      "Happiness and wellbeing (Q46-Q56)"         -> "Happiness and wellbeing"
      "Social values, norms, stereotypes (Q1-Q45)"-> "Social values, norms, stereotypes"

    Returns None if theme is empty or has no extractable name.
    """
    if not theme or not theme.strip():
        return None
    t = re.sub(r"\s*\([^)]*\)\s*$", "", theme.strip()).strip()
    return t if t else None


def q_var_number(var_name: str) -> Optional[int]:
    """Extract question number from a WVS variable name, e.g. Q57 -> 57. Returns None for non-Q vars."""
    if not var_name:
        return None
    m = _Q_VAR_RE.match(var_name.strip())
    return int(m.group(1)) if m else None


def _clean_line(s: str) -> str:
    # Remove common PDF-to-text artifacts and whitespace noise.
    s = s.replace("\x0c", " ").strip()  # form-feed
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_page_number(line: str) -> bool:
    return bool(line) and line.isdigit() and 0 < int(line) < 5000


def _looks_like_section_header(line: str) -> bool:
    """
    Best-effort: WVS codebook has headers like:
      'Happiness and Wellbeing (Q46-Q56)'
      'Technical Variables'
      'Political trust module (Q291-294)'
    """
    if not line:
        return False
    if line.lower() in {"contents"}:
        return True
    if "www.worldvaluessurvey.org" in line.lower():
        return False
    if "the world values survey association" in line.lower():
        return False
    if _OPTION_LIKE_START_RE.match(line):
        return False
    low = line.lower().strip()
    if low in _KNOWN_SECTION_HEADERS:
        return True
    if _Q_RANGE_RE.search(line):
        return True
    if line.endswith("Variables"):
        return True
    if low.endswith("module"):
        return True
    return False


def _parse_option_line(line: str) -> Optional[Tuple[int, str]]:
    m = _OPTION_RE.match(line)
    if not m:
        return None
    try:
        code = int(m.group(1))
    except Exception:
        return None
    text = _clean_line(m.group(2))
    if not text:
        return None
    return code, text


def _is_noise_line(line: str) -> bool:
    if not line:
        return True
    low = line.lower()
    return "www.worldvaluessurvey.org" in low or "the world values survey association" in low


def _parse_card_payload(lines: Sequence[str], start_idx: int) -> Tuple[str, str, List[Dict[str, Any]], Dict[str, str], int]:
    """
    Parse one variable payload from `lines[start_idx:]`.
    Returns: (label, question, options, missing_codes, next_idx).
    """
    i = start_idx
    # Seek first meaningful line for label.
    while i < len(lines):
        line = lines[i]
        if _is_noise_line(line) or _is_page_number(line) or not line:
            i += 1
            continue
        # Ignore leading option rows leaked from previous page/variable.
        if _parse_option_line(line) is not None:
            i += 1
            continue
        if _looks_like_section_header(line) or _looks_like_var_code(line):
            return "", "", [], {}, i
        break

    if i >= len(lines):
        return "", "", [], {}, i

    label = lines[i]
    i += 1

    question_lines: List[str] = []
    options: List[Dict[str, Any]] = []
    missing_codes: Dict[str, str] = {}
    in_options = False

    while i < len(lines):
        nxt = lines[i]
        if _is_noise_line(nxt) or _is_page_number(nxt) or not nxt:
            i += 1
            continue
        if _looks_like_section_header(nxt) or _looks_like_var_code(nxt):
            break

        opt = _parse_option_line(nxt)
        if opt is not None:
            in_options = True
            code, text = opt
            options.append({"code": code, "text": text})
            if code < 0 or any(
                k in text.lower()
                for k in ["missing", "not asked", "no answer", "don't know", "don´t know", "not available"]
            ):
                missing_codes[str(code)] = text
            i += 1
            continue

        # After options begin, a non-option line is usually the next variable's label in matrix-style sections.
        if in_options:
            break

        question_lines.append(nxt)
        i += 1

    question = _clean_line(" ".join(question_lines))
    return label, question, options, missing_codes, i


def _looks_like_var_code(token: str) -> bool:
    """
    Return True for valid WVS-style variable codes and reject common false positives.
    """
    if not token or not _VAR_CODE_RE.match(token):
        return False
    # Guard against country-like 3-letter tokens (AFG, ARG, ...), which are frequent noise in PDFs.
    if _AMBIGUOUS_THREE_LETTER_RE.match(token):
        return token in _KNOWN_THREE_LETTER_VAR_CODES
    return True


@dataclass(frozen=True)
class VariableCard:
    var_name: str
    module: str
    label: str
    question: str
    options: List[Dict[str, Any]]
    missing_codes: Dict[str, str]
    section_theme: str = ""
    variable_kind: str = ""
    battery_id: str = ""
    stem_template: str = ""
    item_text: str = ""
    scale_id: str = ""
    scale_summary: str = ""
    semantic_summary: str = ""

    def __post_init__(self) -> None:
        abstraction = _infer_variable_abstraction(
            var_name=self.var_name,
            module=self.module,
            label=self.label,
            question=self.question,
            options=self.options,
            missing_codes=self.missing_codes,
        )
        for field_name, inferred_value in abstraction.items():
            if not getattr(self, field_name, ""):
                object.__setattr__(self, field_name, inferred_value)

    def to_index_text(
        self,
        *,
        include_raw_question_fallback: bool = True,
        include_raw_options: bool = False,
    ) -> str:
        # Deterministic compact description for retrieval.
        parts: List[str] = []
        if self.module:
            parts.append(f"Module: {self.module}")
        if self.section_theme:
            parts.append(f"Section theme: {self.section_theme}")
        if self.variable_kind:
            parts.append(f"Variable kind: {self.variable_kind}")
        parts.append(f"Variable: {self.var_name}")
        if self.semantic_summary:
            parts.append(f"Meaning: {self.semantic_summary}")
        if self.battery_id:
            parts.append(f"Battery: {self.battery_id}")
        if self.stem_template:
            parts.append(f"Stem: {self.stem_template}")
        if self.item_text:
            parts.append(f"Item: {self.item_text}")
        if self.scale_id:
            parts.append(f"Scale ID: {self.scale_id}")
        if self.scale_summary:
            parts.append(f"Scale: {self.scale_summary}")
        if include_raw_question_fallback:
            if self.label:
                parts.append(f"Raw label: {_truncate_text(self.label)}")
            if self.question and _normalize_text_for_match(self.question) not in _normalize_text_for_match(self.label):
                parts.append(f"Raw question: {_truncate_text(self.question)}")
        if include_raw_options and self.options and self.scale_id != "country_specific_codes":
            opt_preview = "; ".join([f"{o.get('code')}={o.get('text')}" for o in _non_missing_options(self.options)[:6]])
            if opt_preview:
                parts.append(f"Option preview: {opt_preview}")
        if self.missing_codes:
            miss_preview = "; ".join([f"{k}={v}" for k, v in list(self.missing_codes.items())[:10]])
            parts.append(f"Missing codes: {miss_preview}")
        return "\n".join(parts)

    def to_generalized_dict(self) -> Dict[str, Any]:
        return {
            "var_name": self.var_name,
            "module": self.module,
            "section_theme": self.section_theme,
            "variable_kind": self.variable_kind,
            "battery_id": self.battery_id,
            "stem_template": self.stem_template,
            "item_text": self.item_text,
            "scale_id": self.scale_id,
            "scale_summary": self.scale_summary,
            "semantic_summary": self.semantic_summary,
            "original_label": self.label,
            "original_question": self.question,
        }


def parse_wvs_codebook_txt(codebook_txt_path: str | Path) -> List[VariableCard]:
    """
    Parse `pdftotext` output from WVS7 Variables Report (V6.0).
    Produces one VariableCard per variable in the codebook.
    """
    p = Path(codebook_txt_path)
    if not p.exists():
        raise FileNotFoundError(f"Codebook text not found: {p}")

    raw_lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    lines = [_clean_line(l) for l in raw_lines]

    cards: List[VariableCard] = []
    current_module = ""

    i = 0
    while i < len(lines):
        line = lines[i]
        if not line or _is_page_number(line):
            i += 1
            continue

        if _looks_like_section_header(line):
            # Avoid setting module to very generic headers
            low = line.lower()
            if low not in {"contents", "core variables"}:
                current_module = line
            i += 1
            continue

        # Variable block begins with one or more var codes on their own lines.
        if _looks_like_var_code(line):
            pending_vars: List[str] = []
            while i < len(lines):
                cur = lines[i]
                if not cur or _is_page_number(cur) or _is_noise_line(cur):
                    i += 1
                    continue
                if _looks_like_var_code(cur):
                    pending_vars.append(cur)
                    i += 1
                    continue
                break

            for var_name in pending_vars:
                label, question, options, missing_codes, i = _parse_card_payload(lines, i)
                cards.append(
                    VariableCard(
                        var_name=var_name,
                        module=current_module,
                        label=label,
                        question=question,
                        options=options,
                        missing_codes=missing_codes,
                    )
                )
            continue

        i += 1

    # De-duplicate by var_name (keep first)
    seen = set()
    uniq: List[VariableCard] = []
    for c in cards:
        if c.var_name in seen:
            continue
        seen.add(c.var_name)
        uniq.append(c)
    return uniq


def read_csv_columns(csv_path: str | Path) -> List[str]:
    """Read only the header row from a CSV file."""
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
    return [str(c).strip() for c in header if str(c).strip()]


def filter_cards_by_columns(cards: Sequence[VariableCard], columns: Iterable[str]) -> List[VariableCard]:
    """Keep cards whose var_name appears in the provided CSV columns."""
    colset = set(columns)
    if not colset:
        return []
    return [c for c in cards if c.var_name in colset]


def parse_wvs_codebook_txt_for_columns(
    codebook_txt_path: str | Path,
    csv_path: str | Path,
) -> List[VariableCard]:
    """
    Parse WVS codebook text and retain only variables present in the CSV header.
    """
    cards = parse_wvs_codebook_txt(codebook_txt_path)
    columns = read_csv_columns(csv_path)
    return filter_cards_by_columns(cards, columns)


def save_variable_cards_jsonl(cards: Sequence[VariableCard], out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for c in cards:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    return out


def load_variable_cards_jsonl(path: str | Path) -> List[VariableCard]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Variable catalog not found: {p}")
    cards: List[VariableCard] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cards.append(
                VariableCard(
                    var_name=obj.get("var_name", ""),
                    module=obj.get("module", ""),
                    label=obj.get("label", ""),
                    question=obj.get("question", ""),
                    options=list(obj.get("options", []) or []),
                    missing_codes=dict(obj.get("missing_codes", {}) or {}),
                    section_theme=obj.get("section_theme", "") or "",
                    variable_kind=obj.get("variable_kind", "") or "",
                    battery_id=obj.get("battery_id", "") or "",
                    stem_template=obj.get("stem_template", "") or "",
                    item_text=obj.get("item_text", "") or "",
                    scale_id=obj.get("scale_id", "") or "",
                    scale_summary=obj.get("scale_summary", "") or "",
                    semantic_summary=obj.get("semantic_summary", "") or "",
                )
            )
    return cards


class VariableRetriever:
    """
    Lightweight TF-IDF retriever over variable cards.
    """

    def __init__(
        self,
        cards: Sequence[VariableCard],
        *,
        min_df: int = 1,
        max_features: int = 50000,
        include_raw_question_fallback: bool = True,
        include_raw_options_in_index_text: bool = False,
    ):
        self.cards = list(cards)
        self.card_map = {c.var_name: c for c in self.cards}
        self.include_raw_question_fallback = bool(include_raw_question_fallback)
        self.include_raw_options_in_index_text = bool(include_raw_options_in_index_text)
        self._texts = [
            c.to_index_text(
                include_raw_question_fallback=self.include_raw_question_fallback,
                include_raw_options=self.include_raw_options_in_index_text,
            )
            for c in self.cards
        ]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=min_df,
            max_features=max_features,
        )
        self.matrix = self.vectorizer.fit_transform(self._texts)

    def get_cards_by_names(self, var_names: Iterable[str]) -> List[VariableCard]:
        out: List[VariableCard] = []
        for name in var_names:
            card = self.card_map.get(name)
            if card is not None:
                out.append(card)
        return out

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 15,
        module_contains: Optional[str] = None,
        only_vars: Optional[Iterable[str]] = None,
        allowed_q_ranges: Optional[Sequence[Tuple[int, int]]] = None,
        allowed_modules: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []
        q = query.strip()

        allowed: Optional[set[str]] = None
        if only_vars is not None:
            allowed = set(only_vars)

        # Precompute lowercased allowed modules for substring matching
        _allowed_mod_lower = [m.lower() for m in allowed_modules] if allowed_modules else []

        # Build candidate indices (optionally filter by modules or Q-ranges)
        candidate_idx = []
        for idx, c in enumerate(self.cards):
            # Skip non-substantive variables: technical metadata, contextual
            # indicators, and annex items should not be used as X/Y predictors.
            # For index variables, keep Q-recoded items (Q94R, Q95R, etc.) but
            # skip computed composites (I_TRUSTPOLICE, RESEMAVAL, etc.) whose
            # correlation with component items is tautological.
            if c.variable_kind in ('technical', 'contextual', 'annex'):
                continue
            if c.variable_kind == 'index' and not c.var_name.startswith('Q'):
                continue
            if module_contains and module_contains.lower() not in (c.module or "").lower():
                continue
            if allowed is not None and c.var_name not in allowed:
                continue
            # Module-based filtering (works for any dataset)
            if _allowed_mod_lower:
                card_mod = (c.module or "").lower()
                if card_mod and not any(
                    am in card_mod or card_mod in am for am in _allowed_mod_lower
                ):
                    # Fallback: also check Q-range if available (WVS compat)
                    if allowed_q_ranges:
                        qn = q_var_number(c.var_name)
                        if qn is not None and not any(lo <= qn <= hi for lo, hi in allowed_q_ranges):
                            continue
                        # non-Q vars pass if only_vars allows them
                    else:
                        continue
            elif allowed_q_ranges:
                qn = q_var_number(c.var_name)
                if qn is not None:
                    if not any(lo <= qn <= hi for lo, hi in allowed_q_ranges):
                        continue
                # non-Q vars (e.g. B_COUNTRY, W_WEIGHT) always pass when only_vars allows them
            candidate_idx.append(idx)

        if not candidate_idx:
            return []

        qv = self.vectorizer.transform([q])
        sims = cosine_similarity(qv, self.matrix[candidate_idx]).ravel()
        order = np.argsort(-sims)[: max(1, int(top_k))]

        out: List[Dict[str, Any]] = []
        for rank, j in enumerate(order, 1):
            idx = candidate_idx[int(j)]
            c = self.cards[idx]
            out.append(
                {
                    "rank": int(rank),
                    "score": float(sims[int(j)]),
                    "var_name": c.var_name,
                    "module": c.module,
                    "label": c.label,
                    "question": c.question,
                    "options": c.options,
                    "missing_codes": c.missing_codes,
                    "section_theme": c.section_theme,
                    "variable_kind": c.variable_kind,
                    "battery_id": c.battery_id,
                    "stem_template": c.stem_template,
                    "item_text": c.item_text,
                    "scale_id": c.scale_id,
                    "scale_summary": c.scale_summary,
                    "semantic_summary": c.semantic_summary,
                    "index_text": c.to_index_text(
                        include_raw_question_fallback=self.include_raw_question_fallback,
                        include_raw_options=self.include_raw_options_in_index_text,
                    ),
                }
            )
        return out

