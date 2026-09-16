"""Verification layer: checks that a synthesized answer is grounded in the insights it cites.

A judge LLM decomposes the answer into individual claims, and each claim is
checked against the insights it cites via binary entailment.  Faith(q) is the
fraction of supported claims and Halluc(q) = 1 - Faith(q).  Unsupported claims
are flagged; the answer is not rewritten.
"""

import re
from typing import Dict, List

from evaluation.eval_types import SynthesisOutput, VerificationResult
from core.llm_client import create_llm_client
from core.config import config

_CITATION = re.compile(r"\[([^\[\]]+?)\]")


class AnswerCritic:
    """Verify that synthesized answers are supported by the evidence they cite."""

    def __init__(self):
        self.llm = create_llm_client(config.get_llm_config())

    def verify(self, synthesis_output: SynthesisOutput) -> VerificationResult:
        """Decompose the answer into claims and check each against its cited insights."""
        claims = self._extract_claims(synthesis_output.answer)
        evidence = {
            node.node_id: node.insight_sentence
            for node in synthesis_output.retrieved_subgraph
        }

        supported: List[str] = []
        unsupported: List[str] = []
        for claim in claims:
            if self._is_claim_supported(claim, evidence):
                supported.append(claim)
            else:
                unsupported.append(claim)

        faithfulness = len(supported) / len(claims) if claims else 0.0
        return VerificationResult(
            synthesis_output=synthesis_output,
            supported_claims=supported,
            unsupported_claims=unsupported,
            faithfulness_score=faithfulness,
            hallucination_rate=1.0 - faithfulness,
        )

    def _extract_claims(self, answer: str) -> List[str]:
        """Ask the judge LLM to split the answer into individual claims."""
        prompt = f"""Decompose the following answer into its individual factual claims.

Answer:
{answer}

Rules:
- Output one claim per line, starting with "- ".
- Keep each claim self-contained and keep its citation markers (e.g. [node_id]) exactly as written.
- Do not add, merge, or reinterpret claims.

Claims:"""
        response = self.llm.generate(prompt=prompt, temperature=0.0, max_tokens=800)
        claims = []
        for line in (response or "").splitlines():
            line = line.strip()
            if line.startswith("- ") and line[2:].strip():
                claims.append(line[2:].strip())
        return claims

    def _is_claim_supported(self, claim: str, evidence: Dict[str, str]) -> bool:
        """Binary entailment of one claim against the insights that claim cites."""
        cited = [
            token.strip()
            for group in _CITATION.findall(claim)
            for token in group.split(",")
        ]
        cited = list(dict.fromkeys(c for c in cited if c in evidence))
        if not cited:
            return False

        evidence_text = "\n".join(f"[{node_id}] {evidence[node_id]}" for node_id in cited)
        prompt = f"""Verify if the following claim can be supported by the provided evidence.

Claim: {claim}

Evidence:
{evidence_text}

Respond with only "YES" if the claim is supported by the evidence, or "NO" if it is not supported or goes beyond the evidence."""
        response = self.llm.generate(prompt=prompt, temperature=0.0, max_tokens=10)
        return (response or "").strip().upper().startswith("YES")
