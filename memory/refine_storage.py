"""Refine Storage - manages cross-round refinement candidates."""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class RefineRecord:
    """Record of a hypothesis that needs refinement."""
    sentence: str
    validity_score: float
    novelty_score: float
    overall_score: float
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    n_observations: Optional[int] = None
    critique: Optional[str] = None  # Critic's reasoning for why it needs refinement
    columns: Optional[List[str]] = None  # Columns used in the hypothesis
    test_type: Optional[str] = None
    round_id: Optional[int] = None
    timestamp: Optional[str] = None
    refinement_count: int = 0  # How many times this has been refined
    # Original DSL sentence (not readable form), needed to re-execute a
    # refinement candidate later as a fresh Hypothesis.
    dsl_sentence: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.columns is None:
            self.columns = []


class RefineStorage:
    """Manages cross-round refinement candidates."""
    
    def __init__(self, storage_path: str = "./data/memory/refinements.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.refinements: List[RefineRecord] = []
        self._load()
    
    def _load(self):
        """Load refinements from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    # Tolerate unknown keys from older/newer formats.
                    valid_fields = set(RefineRecord.__dataclass_fields__.keys())
                    records = []
                    for record in data.get('refinements', []):
                        filtered = {k: v for k, v in record.items() if k in valid_fields}
                        records.append(RefineRecord(**filtered))
                    self.refinements = records
            except Exception as e:
                print(f"[WARN] Failed to load refinement history: {e}")
                self.refinements = []
        else:
            self.refinements = []
    
    def save(self):
        """Save refinements to storage."""
        try:
            # Convert dataclass to dict and ensure all values are JSON-serializable
            refinements_data = []
            for r in self.refinements:
                record_dict = asdict(r)
                # Ensure all numeric values are native Python types
                for key in ['validity_score', 'novelty_score', 'overall_score', 'p_value', 'effect_size']:
                    if record_dict.get(key) is not None:
                        record_dict[key] = float(record_dict[key])
                if record_dict.get('n_observations') is not None:
                    record_dict['n_observations'] = int(record_dict['n_observations'])
                if record_dict.get('round_id') is not None:
                    record_dict['round_id'] = int(record_dict['round_id'])
                if record_dict.get('refinement_count') is not None:
                    record_dict['refinement_count'] = int(record_dict['refinement_count'])
                refinements_data.append(record_dict)
            
            data = {
                'refinements': refinements_data,
                'last_updated': datetime.now().isoformat(),
                'total_count': len(self.refinements)
            }
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[WARN] Failed to save refinement history: {e}")
            import traceback
            traceback.print_exc()
    
    def add_refinement(
        self,
        sentence: str,
        validity_score: float,
        novelty_score: float,
        overall_score: float,
        p_value: Optional[float] = None,
        effect_size: Optional[float] = None,
        n_observations: Optional[int] = None,
        critique: Optional[str] = None,
        columns: Optional[List[str]] = None,
        test_type: Optional[str] = None,
        round_id: Optional[int] = None,
        dsl_sentence: Optional[str] = None,
        initial_refinement_count: int = 0,
    ):
        """Add a refinement record.

        ``initial_refinement_count`` lets the caller seed the count when
        the record is new — used by the graduation path to carry forward
        the count from a record that was popped, re-executed, and rejected
        again.  Without this the cap (``pop_top_for_graduation``'s
        ``max_refinement_count``) would reset every time we graduated.
        """
        # Convert numpy types to native Python types for JSON serialization
        validity_score = float(validity_score)
        novelty_score = float(novelty_score)
        overall_score = float(overall_score)
        if p_value is not None:
            p_value = float(p_value)
        if effect_size is not None:
            effect_size = float(effect_size)
        if n_observations is not None:
            n_observations = int(n_observations)
        if round_id is not None:
            round_id = int(round_id)

        # Check if this sentence already exists (same hypothesis refined multiple times)
        existing = self.find_by_sentence(sentence)
        if existing:
            # Update existing record: increment refinement count
            existing.refinement_count = max(
                existing.refinement_count + 1,
                initial_refinement_count,
            )
            existing.validity_score = validity_score
            existing.novelty_score = novelty_score
            existing.overall_score = overall_score
            existing.p_value = p_value
            existing.effect_size = effect_size
            existing.n_observations = n_observations
            existing.critique = critique
            existing.round_id = round_id
            existing.timestamp = datetime.now().isoformat()
            if dsl_sentence is not None:
                existing.dsl_sentence = dsl_sentence
        else:
            # New refinement record
            record = RefineRecord(
                sentence=sentence,
                validity_score=validity_score,
                novelty_score=novelty_score,
                overall_score=overall_score,
                p_value=p_value,
                effect_size=effect_size,
                n_observations=n_observations,
                critique=critique,
                columns=columns or [],
                test_type=test_type,
                round_id=round_id,
                dsl_sentence=dsl_sentence,
                refinement_count=initial_refinement_count,
            )
            self.refinements.append(record)

        # Limit storage to last 200 refinements to avoid bloat
        if len(self.refinements) > 200:
            self.refinements = self.refinements[-200:]
        self.save()
    
    def find_by_sentence(self, sentence: str) -> Optional[RefineRecord]:
        """Find a refinement record by sentence (exact match)."""
        sentence_lower = sentence.lower().strip()
        for r in self.refinements:
            if r.sentence.lower().strip() == sentence_lower:
                return r
        return None
    
    def get_recent_refinements(self, max_count: int = 50) -> List[RefineRecord]:
        """Get recent refinements, sorted by overall_score (highest first)."""
        recent = self.refinements[-max_count:]
        # Sort by overall_score descending (best candidates first)
        return sorted(recent, key=lambda x: x.overall_score, reverse=True)
    
    def get_high_potential_refinements(
        self,
        min_overall_score: float = 0.3,
        max_count: int = 20
    ) -> List[RefineRecord]:
        """Get refinements with high potential (good scores but need improvement)."""
        candidates = [
            r for r in self.refinements
            if r.overall_score >= min_overall_score
        ]
        # Sort by overall_score descending
        candidates.sort(key=lambda x: x.overall_score, reverse=True)
        return candidates[:max_count]

    def pop_top_for_graduation(
        self,
        max_count: int = 3,
        min_overall_score: float = 0.50,
        max_refinement_count: int = 2,
    ) -> List[RefineRecord]:
        """Pop top refinement candidates for re-execution ("graduation").

        Returns up to ``max_count`` records with the highest overall_score that:
          - Have a stored ``dsl_sentence`` (records without it cannot be re-executed).
          - Have ``overall_score >= min_overall_score`` (high-potential only).
          - Have been graduated fewer than ``max_refinement_count`` times
            (prevents infinite retry loops for stubbornly borderline items).

        Popped records are REMOVED from storage and the file is saved.
        If the retry still ends in "refine", the critic's add_refinement
        call will re-insert them as a fresh record with refinement_count
        incremented; otherwise they either become accepted insights or
        graduate out of the refinement queue entirely.
        """
        eligible = [
            r for r in self.refinements
            if r.dsl_sentence
            and r.overall_score >= min_overall_score
            and r.refinement_count < max_refinement_count
        ]
        eligible.sort(key=lambda x: x.overall_score, reverse=True)
        popped = eligible[:max_count]
        if not popped:
            return []
        popped_ids = {id(r) for r in popped}
        self.refinements = [r for r in self.refinements if id(r) not in popped_ids]
        self.save()
        return popped
    
    def get_all_sentences(self) -> List[str]:
        """Get all refinement candidate sentences."""
        return [r.sentence for r in self.refinements]
    
    def clear(self):
        """Clear all refinements."""
        self.refinements = []
        self.save()
    
    def get_statistics(self) -> Dict:
        """Get statistics about refinements."""
        if not self.refinements:
            return {
                'total': 0,
                'avg_overall_score': 0.0,
                'avg_validity_score': 0.0,
                'avg_novelty_score': 0.0,
                'high_potential_count': 0
            }
        
        avg_overall = sum(r.overall_score for r in self.refinements) / len(self.refinements)
        avg_validity = sum(r.validity_score for r in self.refinements) / len(self.refinements)
        avg_novelty = sum(r.novelty_score for r in self.refinements) / len(self.refinements)
        high_potential = len([r for r in self.refinements if r.overall_score >= 0.3])
        
        return {
            'total': len(self.refinements),
            'avg_overall_score': avg_overall,
            'avg_validity_score': avg_validity,
            'avg_novelty_score': avg_novelty,
            'high_potential_count': high_potential
        }



