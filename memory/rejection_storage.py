"""Rejection Storage - manages cross-round rejection history."""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class RejectionRecord:
    """Record of a rejected hypothesis."""
    sentence: str
    reason: str  # "PRE-EXISTING", "SELF-DUPLICATE", "STRUCTURAL_DUPLICATE", "REJECTED_BY_CRITIC"
    similarity_score: Optional[float] = None  # For PRE-EXISTING
    round_id: Optional[int] = None
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class RejectionStorage:
    """Manages cross-round rejection history."""
    
    def __init__(self, storage_path: str = "./data/memory/rejections.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.rejections: List[RejectionRecord] = []
        self._load()
    
    def _load(self):
        """Load rejections from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self.rejections = [
                        RejectionRecord(**record) for record in data.get('rejections', [])
                    ]
            except Exception as e:
                print(f"[WARN] Failed to load rejection history: {e}")
                self.rejections = []
        else:
            self.rejections = []
    
    def save(self):
        """Save rejections to storage."""
        try:
            # Convert dataclass to dict and ensure all values are JSON-serializable
            rejections_data = []
            for r in self.rejections:
                record_dict = asdict(r)
                # Ensure all numeric values are native Python types
                if record_dict.get('similarity_score') is not None:
                    record_dict['similarity_score'] = float(record_dict['similarity_score'])
                if record_dict.get('round_id') is not None:
                    record_dict['round_id'] = int(record_dict['round_id'])
                rejections_data.append(record_dict)
            
            data = {
                'rejections': rejections_data,
                'last_updated': datetime.now().isoformat(),
                'total_count': len(self.rejections)
            }
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[WARN] Failed to save rejection history: {e}")
            import traceback
            traceback.print_exc()
    
    def add_rejection(
        self,
        sentence: str,
        reason: str,
        similarity_score: Optional[float] = None,
        round_id: Optional[int] = None
    ):
        """Add a rejection record."""
        # Convert numpy types to native Python types for JSON serialization
        if similarity_score is not None:
            # Handle numpy.float32, numpy.float64, etc.
            similarity_score = float(similarity_score)
        if round_id is not None:
            round_id = int(round_id)
        
        record = RejectionRecord(
            sentence=sentence,
            reason=reason,
            similarity_score=similarity_score,
            round_id=round_id
        )
        self.rejections.append(record)
        # Limit storage to last 500 rejections to avoid bloat
        if len(self.rejections) > 500:
            self.rejections = self.rejections[-500:]
        self.save()
    
    def get_recent_rejections(self, max_count: int = 100) -> List[RejectionRecord]:
        """Get recent rejections."""
        return self.rejections[-max_count:]
    
    def get_rejections_by_reason(self, reason: str) -> List[RejectionRecord]:
        """Get rejections filtered by reason."""
        return [r for r in self.rejections if r.reason == reason]
    
    def get_all_sentences(self) -> List[str]:
        """Get all rejected sentences."""
        return [r.sentence for r in self.rejections]
    
    def clear(self):
        """Clear all rejections."""
        self.rejections = []
        self.save()
    
    def get_statistics(self) -> Dict:
        """Get statistics about rejections."""
        if not self.rejections:
            return {
                'total': 0,
                'by_reason': {},
                'recent_count': 0
            }
        
        by_reason = {}
        for r in self.rejections:
            by_reason[r.reason] = by_reason.get(r.reason, 0) + 1
        
        return {
            'total': len(self.rejections),
            'by_reason': by_reason,
            'recent_count': len(self.rejections[-50:])  # Last 50
        }

