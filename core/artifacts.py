"""Run artifact logging for AutoKD.

This module provides a lightweight, structured way to persist "middle outputs"
needed for debugging/auditing (prompts, generated code, parsed specs, results,
scores, errors) without bloating the insight graph JSON.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import traceback
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json(obj: Any) -> Any:
    """Best-effort JSON conversion for common project objects."""
    if obj is None:
        return None
    if is_dataclass(obj):
        return _safe_json(asdict(obj))
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_safe_json(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _safe_json(v) for k, v in obj.items()}
    # common types that are not JSON serializable
    try:
        return float(obj)  # e.g., numpy scalars
    except Exception:
        return str(obj)


def _fingerprint_file(path: Optional[str], max_bytes: int = 1_000_000) -> Optional[Dict[str, Any]]:
    """Cheap, reproducible-ish fingerprint for large files (no full hash)."""
    if not path:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return {"path": str(p), "exists": False}

    stat = p.stat()
    fp: Dict[str, Any] = {
        "path": str(p),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }
    try:
        import hashlib

        h = hashlib.sha256()
        with p.open("rb") as f:
            chunk = f.read(max_bytes)
        h.update(chunk)
        fp["sha256_first_bytes"] = h.hexdigest()
        fp["hashed_bytes"] = len(chunk)
    except Exception as e:
        fp["hash_error"] = f"{type(e).__name__}: {e}"
    return fp


class ArtifactWriter:
    """Writes per-run / per-round / per-hypothesis artifacts under a run directory."""

    def __init__(
        self,
        enabled: bool,
        base_dir: str = "./data/runs",
        run_id: Optional[str] = None,
        save_llm_prompts: bool = True,
        save_llm_responses: bool = True,
        save_generated_code: bool = True,
        save_data_fingerprint: bool = True,
    ):
        self.enabled = bool(enabled)
        self.base_dir = Path(base_dir)
        self.save_llm_prompts = bool(save_llm_prompts)
        self.save_llm_responses = bool(save_llm_responses)
        self.save_generated_code = bool(save_generated_code)
        self.save_data_fingerprint = bool(save_data_fingerprint)

        if not self.enabled:
            self.run_id = run_id or "disabled"
            self.run_dir = None
            return

        if run_id is None:
            run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}"
        self.run_id = run_id
        self.run_dir = (self.base_dir / self.run_id).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)

        (self.run_dir / "rounds").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "events").mkdir(parents=True, exist_ok=True)

    def path(self) -> Optional[Path]:
        return self.run_dir

    def _write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _write_json(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_safe_json(data), indent=2, ensure_ascii=False), encoding="utf-8")

    def _append_jsonl(self, path: Path, record: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_safe_json(record), ensure_ascii=False) + "\n")

    def log_event(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled or self.run_dir is None:
            return
        record = {"ts": _utc_now_iso(), "type": event_type, "payload": payload or {}}
        self._append_jsonl(self.run_dir / "events" / "events.jsonl", record)

    def write_run_manifest(
        self,
        *,
        config_snapshot: Optional[Dict[str, Any]] = None,
        goal: Optional[str] = None,
        data_path: Optional[str] = None,
        memory_path: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled or self.run_dir is None:
            return
        manifest: Dict[str, Any] = {
            "run_id": self.run_id,
            "created_at_utc": _utc_now_iso(),
            "goal": goal,
            "data_path": data_path,
            "memory_path": memory_path,
            "system": {
                "python": sys.version,
                "platform": platform.platform(),
            },
            "config": config_snapshot or {},
            "data_fingerprint": _fingerprint_file(data_path) if self.save_data_fingerprint else None,
        }
        if extra:
            manifest["extra"] = extra
        self._write_json(self.run_dir / "run.json", manifest)

    def round_dir(self, round_id: int) -> Path:
        if not self.enabled or self.run_dir is None:
            raise RuntimeError("ArtifactWriter is disabled")
        d = self.run_dir / "rounds" / f"round_{round_id:03d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def hypothesis_dir(self, round_id: int, hypothesis_id: str) -> Path:
        d = self.round_dir(round_id) / "hypotheses" / hypothesis_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_round_context(self, round_id: int, context: Any, data_schema: Optional[str] = None) -> None:
        if not self.enabled or self.run_dir is None:
            return
        payload = {
            "round_id": round_id,
            "context": context,
            "data_schema": data_schema,
        }
        self._write_json(self.round_dir(round_id) / "round.json", payload)

    def save_round_note(self, round_id: int, filename: str, text: str) -> None:
        if not self.enabled or self.run_dir is None:
            return
        self._write_text(self.round_dir(round_id) / filename, text)

    def save_round_json(self, round_id: int, filename: str, data: Any) -> None:
        if not self.enabled or self.run_dir is None:
            return
        self._write_json(self.round_dir(round_id) / filename, data)

    def save_generation_attempt(
        self,
        round_id: int,
        attempt_idx: int,
        prompt: Optional[str],
        system_prompt: Optional[str],
        temperature: Optional[float],
        response_text: Optional[str],
        parsed_hypotheses: Any,
    ) -> None:
        if not self.enabled or self.run_dir is None:
            return
        d = self.round_dir(round_id) / "generation" / f"attempt_{attempt_idx:02d}"
        d.mkdir(parents=True, exist_ok=True)
        self._write_json(
            d / "generation.json",
            {
                "attempt": attempt_idx,
                "temperature": temperature,
                "parsed_hypotheses": parsed_hypotheses,
            },
        )
        if self.save_llm_prompts and prompt is not None:
            self._write_text(d / "prompt.txt", prompt)
        if self.save_llm_prompts and system_prompt is not None:
            self._write_text(d / "system_prompt.txt", system_prompt)
        if self.save_llm_responses and response_text is not None:
            self._write_text(d / "response.txt", response_text)

    def save_hypothesis_statuses(self, round_id: int, statuses: Any) -> None:
        if not self.enabled or self.run_dir is None:
            return
        self._write_json(self.round_dir(round_id) / "hypothesis_statuses.json", statuses)

    def save_exception(self, path: Path, stage: str, exc: BaseException) -> None:
        if not self.enabled or self.run_dir is None:
            return
        self._write_json(
            path / "error.json",
            {
                "stage": stage,
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "ts": _utc_now_iso(),
            },
        )

    def save_hypothesis_text(self, round_id: int, hypothesis_id: str, filename: str, text: str) -> None:
        if not self.enabled or self.run_dir is None:
            return
        self._write_text(self.hypothesis_dir(round_id, hypothesis_id) / filename, text)

    def save_hypothesis_json(self, round_id: int, hypothesis_id: str, filename: str, data: Any) -> None:
        if not self.enabled or self.run_dir is None:
            return
        self._write_json(self.hypothesis_dir(round_id, hypothesis_id) / filename, data)

