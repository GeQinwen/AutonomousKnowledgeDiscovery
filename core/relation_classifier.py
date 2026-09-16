"""Pairwise relation classifier for insight graph edges.

Determines the epistemic relationship between two insights based on their
metadata (columns, test_type, modules) — no LLM calls required.
"""

import re
from typing import Set, List, Optional

from core.types import Insight, RelationType

# ── Weight / structural columns to strip when computing core overlap ──
_STRUCTURAL_COLUMNS = frozenset({
    "W_WEIGHT", "PWGHT", "sampling_weight",
    "B_COUNTRY", "B_COUNTRY_ALPHA", "H_URBRURAL",
    "FW_START", "FW_END", "A_YEAR",
    "age", "sex", "gender", "income", "income_status",
    "education_level", "employment_status", "survey_year",
    "urban vs rural residence",
})

# ── Test-type epistemic depth ──
_TEST_TYPE_RANK = {
    "assoc": 1, "diff": 1, "correlation": 1,
    "interact": 2, "interaction": 2,
    "heterogeneity": 3,
}

# ── WVS Q-number → module mapping (fallback when no catalog modules) ──
_Q_RANGES = [
    ((1, 45), "Social values"), ((46, 56), "Wellbeing"),
    ((57, 105), "Trust"), ((106, 111), "Economic"),
    ((112, 120), "Corruption"), ((121, 130), "Migration"),
    ((131, 151), "Security"), ((152, 157), "Postmaterialism"),
    ((158, 163), "Science"), ((164, 175), "Religion"),
    ((176, 198), "Ethics"), ((199, 234), "Political participation"),
    ((235, 259), "Political culture"),
]

_SKIP_MODULES = {"Demographics", "demographics", "identifiers",
                 "demographics and ses"}

# ── Opposite-direction word pairs for contradiction detection ──
_OPPOSITE_PATTERNS = [
    ("positively", "negatively"),
    ("positive", "negative"),
    ("increases", "decreases"),
    ("more likely", "less likely"),
    ("higher", "lower"),
    ("greater", "smaller"),
]

_Q_NUMBER_RE = re.compile(r"Q(\d+)")


# ─────────────────────────── Helper functions ────────────────────────────

def test_type_rank(test_type: str) -> int:
    """Map test_type string to epistemic depth (0 = seed/unknown)."""
    return _TEST_TYPE_RANK.get(test_type, 0)


def core_columns(insight: Insight) -> Set[str]:
    """Return core research variables (X, Y, group, moderator), excluding controls.

    Uses parsed DSL spec if available (preferred: only core vars).
    Falls back to all columns minus structural for non-DSL insights.
    """
    meta = insight.metadata or {}
    parsed = meta.get("parsed")
    if parsed:
        core = set()
        for key in ("x", "y", "group", "moderator"):
            val = parsed.get(key)
            if val:
                core.add(val)
        if core:
            return core - _STRUCTURAL_COLUMNS
    # Fallback: all columns minus structural
    cols = set(meta.get("columns", []) or [])
    return cols - _STRUCTURAL_COLUMNS


def module_namespace(insight: Insight) -> str:
    """Which naming scheme insight_modules() used: 'catalog', 'qrange', or 'none'.

    Module names from the variable catalog (e.g. "Social Capital, Trust and
    Organizational Mem (Q57-Q105)") and from the Q-range fallback (e.g. "Trust")
    are different vocabularies for the same concepts.  Comparing across them
    always reports "disjoint", which silently turns EXTENDS into a constant, so
    callers must check that both sides share a namespace before comparing.
    """
    meta = insight.metadata or {}
    if meta.get("card_modules"):
        return "catalog"
    for c in meta.get("columns", []) or []:
        if _Q_NUMBER_RE.match(str(c)):
            return "qrange"
    return "none"


def insight_modules(insight: Insight) -> Set[str]:
    """Return the set of thematic module names for an insight's columns.

    Checks ``card_modules`` in metadata first (populated by the variable
    catalog for any dataset).  Falls back to WVS Q-number ranges.
    """
    meta = insight.metadata or {}

    # Fast path: catalog-provided module mapping
    card_modules = meta.get("card_modules")
    if card_modules:
        mods = set()
        for mod in card_modules.values():
            if mod and mod.lower() not in _SKIP_MODULES:
                mods.add(mod)
        return mods

    # Fallback: WVS Q-number ranges
    cols = meta.get("columns", [])
    mods: Set[str] = set()
    for c in cols:
        m = _Q_NUMBER_RE.match(str(c))
        if m:
            qn = int(m.group(1))
            for (lo, hi), name in _Q_RANGES:
                if lo <= qn <= hi and name not in _SKIP_MODULES:
                    mods.add(name)
                    break
    return mods


def _jaccard(a: Set[str], b: Set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _readable(insight: Insight) -> str:
    """Prefer the rendered sentence; the raw DSL carries no direction words."""
    meta = insight.metadata or {}
    return str(meta.get("readable_sentence") or insight.sentence or "")


def _signed_effect(insight: Insight) -> Optional[float]:
    """Signed effect size, but only when the finding is statistically credible.

    Returns None when the effect or p-value is missing or the result is not
    significant, so a null result never counts as a contradiction.
    """
    meta = insight.metadata or {}
    eff, p = meta.get("effect_size"), meta.get("p_value")
    try:
        eff = float(eff)
        p = float(p)
    except (TypeError, ValueError):
        return None
    if p >= 0.05 or eff == 0.0:
        return None
    return eff


def estimand_key(insight: Insight) -> Optional[tuple]:
    """Identity of *what quantity was estimated*, or None when unknowable.

    Two effect sizes are only comparable when they estimate the same thing.
    The variable *set* is not enough: ``interact(Q57, Q34_3 | Q275R)`` and
    ``interact(Q57, Q34_3 | Q112)`` share two of three core variables (Jaccard
    0.5) but report how the Q57→Q34_3 association varies with two *different*
    moderators — opposite signs there are two ordinary findings, not a
    disagreement.  The estimand therefore includes the moderator and grouping
    variable, plus the test_type (a main effect and an interaction coefficient
    are different quantities even on identical variables).

    Returns None when the parsed DSL roles are missing, so callers treat the
    comparison as unknowable rather than asserting a relationship.
    """
    parsed = (insight.metadata or {}).get("parsed") or {}
    x, y = parsed.get("x"), parsed.get("y")
    if not (x and y):
        return None
    return (
        (insight.metadata or {}).get("test_type", ""),
        frozenset((x, y)),
        parsed.get("moderator") or None,
        parsed.get("group") or None,
    )


def contradicts(source: Insight, target: Insight) -> bool:
    """True when two insights estimate the *same* quantity and disagree on sign.

    Single source of truth for "contradiction", shared by the edge classifier
    (`classify_relation`) and the orchestrator's conflict detector, so the loop
    cannot chase conflicts the graph would never record — or vice versa.
    """
    src_key, tgt_key = estimand_key(source), estimand_key(target)
    if src_key is None or tgt_key is None or src_key != tgt_key:
        return False
    return _has_opposite_direction(source, target)


def _has_opposite_direction(source: Insight, target: Insight) -> bool:
    """True when two insights report opposing directions for the same relationship.

    Prefers stored effect-size signs: they are deterministic, language-free, and
    already persisted.  Direction words in the sentences are only a fallback for
    insights without effect metadata, since a sentence built from raw DSL
    (``assoc(Q1, Q2) controlling for ...``) contains none.
    """
    src_eff, tgt_eff = _signed_effect(source), _signed_effect(target)
    if src_eff is not None and tgt_eff is not None:
        return (src_eff > 0) != (tgt_eff > 0)

    # Fallback for insights lacking effect metadata (e.g. seed briefs).
    s1, s2 = _readable(source).lower(), _readable(target).lower()
    if not s1 or not s2:
        return False
    for pos, neg in _OPPOSITE_PATTERNS:
        if (pos in s1 and neg in s2) or (neg in s1 and pos in s2):
            return True
    return False


# ─────────────────────── Main classifier ─────────────────────────────────

def classify_relation(source: Insight, target: Insight) -> RelationType:
    """Classify the epistemic relationship between *source* (new) and
    *target* (existing) insights.

    Priority order: CONTRADICTS > DEEPENS > EXTENDS > NARROWS (fallback).
    All decisions are deterministic from metadata — no LLM calls.
    """
    src_cols = core_columns(source)
    tgt_cols = core_columns(target)

    src_rank = test_type_rank((source.metadata or {}).get("test_type", ""))
    tgt_rank = test_type_rank((target.metadata or {}).get("test_type", ""))

    # ── 1. CONTRADICTS: same estimand, opposing direction ───────────────
    if contradicts(source, target):
        return RelationType.CONTRADICTS

    # ── 2. DEEPENS: source investigates same core vars at higher depth ──
    #    e.g. assoc(X,Y) → interact(X,Y|Z) → heterogeneity(X,Y by group)
    if src_rank > tgt_rank > 0 and tgt_cols and tgt_cols <= src_cols:
        return RelationType.DEEPENS

    # ── 3. EXTENDS: disjoint thematic modules = cross-cutting bridge ────
    #    Only meaningful when both sides name their modules in the SAME
    #    vocabulary; comparing catalog names against Q-range names always
    #    reports "disjoint" and makes EXTENDS a constant.  On a namespace
    #    mismatch fall through to NARROWS rather than asserting a cross-module
    #    bridge we cannot verify.
    src_mods = insight_modules(source)
    tgt_mods = insight_modules(target)
    same_namespace = module_namespace(source) == module_namespace(target)
    if same_namespace and src_mods and tgt_mods and src_mods.isdisjoint(tgt_mods):
        return RelationType.EXTENDS

    # ── 4. NARROWS: fallback — same or overlapping theme ────────────────
    return RelationType.NARROWS
