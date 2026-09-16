"""Type definitions for knowledge extraction (retrieval, synthesis, verification)."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class Query:
    """A natural-language research question posed to the insight graph."""
    query_id: str
    query_text: str
    gold_claim_ids: List[str] = field(default_factory=list)  # gold claims the query should cover


@dataclass
class RetrievedNode:
    """A node retrieved from the graph."""
    node_id: str
    insight_sentence: str
    retrieval_rank: int
    similarity_score: Optional[float] = None
    hop_distance: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Citation:
    """A citation in the synthesized answer."""
    node_id: str
    evidence_id: Optional[str] = None
    claim_text: Optional[str] = None  # The specific claim that cites this


@dataclass
class SynthesisOutput:
    """Output from synthesis layer."""
    query_id: str
    answer: str
    citations: List[Citation]
    retrieved_subgraph: List[RetrievedNode]
    synthesis_tokens: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    """Result from verification/critic layer."""
    synthesis_output: SynthesisOutput
    supported_claims: List[str]  # Claim texts that are supported
    unsupported_claims: List[str]  # Claim texts that lack evidence
    faithfulness_score: float  # 0-1, proportion of claims with evidence
    hallucination_rate: float  # 1 - faithfulness_score


@dataclass
class EvaluationConfig:
    """Synthesis settings (sampling temperature and answer token budget)."""
    synthesis_token_limit: int = 600
    synthesis_temperature: float = 0.3
