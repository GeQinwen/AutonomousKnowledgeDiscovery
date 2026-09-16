"""Dataset profile: encapsulates all dataset-specific knowledge.

A profile is a plain dict loaded from YAML.  The rest of the codebase reads
from it via ``get_profile()`` instead of hard-coding WVS / Amazon constants.
When no profile is loaded the system falls back to legacy WVS defaults so
existing behaviour is preserved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_active_profile: Dict[str, Any] = {}


def load_profile(path: str | Path) -> Dict[str, Any]:
    """Load a dataset profile YAML and set it as the active profile."""
    global _active_profile
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset profile not found: {p}")
    with open(p, "r") as f:
        _active_profile = yaml.safe_load(f) or {}
    return _active_profile


def set_profile(profile: Dict[str, Any]) -> None:
    """Programmatically set the active profile (e.g. from a run script)."""
    global _active_profile
    _active_profile = dict(profile)


def get_profile() -> Dict[str, Any]:
    """Return the currently active profile (may be empty)."""
    return _active_profile


# ---------------------------------------------------------------------------
# Convenience accessors with sensible fallbacks
# ---------------------------------------------------------------------------

def profile_get(key: str, default: Any = None) -> Any:
    """Dot-path lookup on the active profile, e.g. ``profile_get("name")``."""
    keys = key.split(".")
    val: Any = _active_profile
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k)
            if val is None:
                return default
        else:
            return default
    return val


# --- Frequently used accessors -------------------------------------------

def profile_name() -> str:
    return profile_get("name", "unknown")

def domain_description() -> str:
    return profile_get("domain_description", "")

def use_two_phase() -> bool:
    return bool(profile_get("use_two_phase", False))

def use_dsl() -> bool:
    return bool(profile_get("use_dsl", False))

def missing_codes() -> Set[int]:
    raw = profile_get("missing_codes", [])
    return set(int(c) for c in raw) if raw else set()

def derived_pairs() -> List[frozenset]:
    """Return pairs of columns that are mechanically derived from each other."""
    raw = profile_get("derived_pairs", [])
    return [frozenset(pair) for pair in raw] if raw else []

def weight_columns() -> List[str]:
    return list(profile_get("weight_columns", []) or [])

def control_only_columns() -> List[str]:
    return list(profile_get("control_only_columns", []) or [])

def id_columns() -> List[str]:
    return list(profile_get("id_columns", []) or [])

def text_columns() -> List[str]:
    return list(profile_get("text_columns", []) or [])

def known_categorical() -> List[str]:
    return list(profile_get("known_categorical", []) or [])

def drop_columns() -> List[str]:
    return list(profile_get("drop_columns", []) or [])

def grouping_excluded() -> List[str]:
    return list(profile_get("grouping_excluded", []) or [])

def nlp_features() -> List[str]:
    return list(profile_get("nlp_features", []) or [])

def column_labels() -> Dict[str, str]:
    return dict(profile_get("column_labels", {}) or {})

def prompt_hints() -> List[str]:
    return list(profile_get("prompt_hints", []) or [])

def demographic_synonyms() -> Dict[str, str]:
    return dict(profile_get("demographic_synonyms", {}) or {})

def sign_semantics() -> Dict[str, Any]:
    return dict(profile_get("sign_semantics", {}) or {})

# --- Accessors for the two-phase DSL pipeline -----------------------------

def phase1_persona() -> str:
    """Persona string for Phase 1 (general idea) prompt."""
    return str(profile_get("phase1_persona", "") or "")

def phase1_wording_hint() -> str:
    """Domain-specific wording hint for Phase 1 prompt."""
    return str(profile_get("phase1_wording_hint", "") or "")

def dsl_rules() -> List[str]:
    """Dataset-specific rules for Phase 2 (DSL hypothesis) prompt."""
    return list(profile_get("dsl_rules", []) or [])

def concept_menu_profile() -> List[str]:
    """Concept menu items from the profile."""
    return list(profile_get("concept_menu", []) or [])

def always_keep_profile() -> List[str]:
    """Columns to always include in schema selection."""
    return list(profile_get("always_keep", []) or [])

def theme_keywords() -> Dict[str, List[str]]:
    """Module name -> keyword list for secondary theme inference."""
    return dict(profile_get("theme_keywords", {}) or {})
