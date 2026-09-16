"""Type definitions for AutoKD system."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime


class ResultStatus(str, Enum):
    """Quality state for ExperimentResult.

    VALID            – code ran AND all result scalars pass strict validation.
    VALID_WITH_WARNING – code ran, scalars are valid, but material warnings were raised.
    RESULT_INVALID   – code ran but produced non-scalar / NaN / out-of-range values.
    EXECUTION_ERROR  – code failed to execute (syntax, runtime, timeout).
    """
    VALID = "valid"
    VALID_WITH_WARNING = "valid_with_warning"
    RESULT_INVALID = "result_invalid"
    EXECUTION_ERROR = "execution_error"


class RelationType(str, Enum):
    """Types of relations between insights.

    All edges are directed: new_insight ──TYPE──▶ existing_insight.
    Types are determined by *pairwise* metadata comparison, not the new
    insight's novelty score alone.

    - DEEPENS:     new insight investigates the same core variables at a higher
                   test-type level (assoc → interact → heterogeneity chain).
    - EXTENDS:     new insight bridges to a different thematic module.
    - NARROWS:     new insight refines within the same theme — adds controls,
                   conditions, or subgroups.
    - CONTRADICTS: new insight opposes the existing one on shared variables.
    """
    DEEPENS = "deepens"
    EXTENDS = "extends"
    NARROWS = "narrows"
    CONTRADICTS = "contradicts"


@dataclass
class Insight:
    """Represents a single insight in the Discovery Memory."""
    id: str
    sentence: str
    source: str  # e.g. "autokd", "seed_prescan", "statistical_profiling"
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class InsightRelation:
    """Represents a relation between two insights."""
    source_id: str
    target_id: str
    relation_type: RelationType
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Hypothesis:
    """A hypothesis proposed by the Generator."""
    id: str
    sentence: str
    context_insight_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Results from running an experiment (Coder/Runner output)."""
    hypothesis_id: str
    effect_size: float
    p_value: float
    confidence_interval: tuple  # (lower, upper)
    n_observations: int
    coverage: float  # Proportion of data this applies to
    replication_status: Optional[str] = None
    raw_results: Dict[str, Any] = field(default_factory=dict)
    # DSL strict-validation fields (defaults keep non-DSL results valid)
    result_status: str = "valid"
    execution_warnings: List[str] = field(default_factory=list)
    test_method: Optional[str] = None
    formula: Optional[str] = None
    warning_severity: Optional[str] = None
    dsl_family: Optional[str] = None  # assoc | diff | interact | heterogeneity
    review_flags: List[str] = field(default_factory=list)
    summary_estimand: Optional[str] = None
    summary_term: Optional[str] = None
    sign_semantics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Evaluation:
    """Statistical evaluation of an experiment result."""
    hypothesis_id: str
    validity_score: float  # 0-1
    coverage: float
    effect_size: float
    p_value: float
    is_replicated: bool
    flags: List[str] = field(default_factory=list)


@dataclass
class InsightScore:
    """Final score for an insight from the Critic."""
    hypothesis_id: str
    overall_score: float  # 0-1
    novelty_score: float
    usefulness_score: float
    validity_score: float
    interpretability_score: float
    explanation: str
    decision: str  # "accept", "refine", "reject"
    related_insight_ids: List[str] = field(default_factory=list)


@dataclass
class RoundContext:
    """Context for a discovery round."""
    round_id: int
    focus_area: str
    anchor_insight_ids: List[str]
    retrieved_insights: List[Insight]
    goal: str
    mode: str  # "refinement", "exploration", "global_exploration", "conflict_resolution"
    data_schema: Optional[str] = None  # Data schema constraints for hypothesis generation


@dataclass
class RoundResult:
    """Results from a complete discovery round."""
    round_id: int
    hypotheses: List[Hypothesis]
    experiment_results: List[ExperimentResult]
    evaluations: List[Evaluation]
    insight_scores: List[InsightScore]
    accepted_insights: List[Insight]
    summary: str





