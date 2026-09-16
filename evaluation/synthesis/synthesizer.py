"""Synthesis layer for generating answers from retrieved nodes."""

from typing import List
import re

from evaluation.eval_types import (
    Query, RetrievedNode, SynthesisOutput, Citation, EvaluationConfig
)
from core.llm_client import create_llm_client
from core.config import config as global_config


class Synthesizer:
    """Synthesize answers from retrieved graph nodes."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.llm = create_llm_client(global_config.get_llm_config())

    def synthesize(
        self,
        query: Query,
        retrieved_nodes: List[RetrievedNode]
    ) -> SynthesisOutput:
        """Synthesize answer from retrieved nodes.

        Args:
            query: Query object
            retrieved_nodes: List of retrieved nodes

        Returns:
            SynthesisOutput with answer and citations
        """
        if not retrieved_nodes:
            return SynthesisOutput(
                query_id=query.query_id,
                answer="No relevant insights found in the knowledge graph.",
                citations=[],
                retrieved_subgraph=retrieved_nodes,
                synthesis_tokens=0
            )

        # Build prompt
        prompt = self._build_synthesis_prompt(query, retrieved_nodes)

        # Generate answer
        answer_text = self.llm.generate(
            prompt=prompt,
            temperature=self.config.synthesis_temperature,
            max_tokens=self.config.synthesis_token_limit
        )

        # Extract citations from answer
        citations = self._extract_citations(answer_text, retrieved_nodes)

        # Estimate token usage (rough estimate)
        synthesis_tokens = len(prompt.split()) + len(answer_text.split())

        return SynthesisOutput(
            query_id=query.query_id,
            answer=answer_text,
            citations=citations,
            retrieved_subgraph=retrieved_nodes,
            synthesis_tokens=synthesis_tokens,
            metadata={"prompt_length": len(prompt)}
        )

    def _build_synthesis_prompt(
        self,
        query: Query,
        retrieved_nodes: List[RetrievedNode]
    ) -> str:
        """Build prompt for synthesis."""
        nodes_section = "\n".join(
            f"[{node.node_id}] {node.insight_sentence}" for node in retrieved_nodes
        )

        prompt = f"""You are a scientific research assistant. Based on the following insights from a knowledge graph, synthesize a clear and evidence-based answer to the query.

Query: {query.query_text}

Retrieved Insights:
{nodes_section}

Instructions:
1. Synthesize a comprehensive answer that directly addresses the query
2. For each claim you make, cite the relevant insight using [node_id] format
3. Be specific: include variables, directions, and conditions when available
4. Structure your answer with 2-5 key points
5. If insights conflict, acknowledge the conflict
6. If information is insufficient, state what is missing

Answer:"""

        return prompt

    def _extract_citations(
        self,
        answer_text: str,
        retrieved_nodes: List[RetrievedNode]
    ) -> List[Citation]:
        """Extract citations from answer text.

        Looks for patterns like [node_id] or [node_id, evidence_id]
        """
        citations = []

        # Pattern: [node_id] or [node_id, evidence_id]
        citation_pattern = r'\[([^\]]+)\]'
        matches = re.finditer(citation_pattern, answer_text)

        node_ids = {node.node_id for node in retrieved_nodes}

        for match in matches:
            content = match.group(1)
            parts = [p.strip() for p in content.split(',')]

            node_id = parts[0]
            if node_id in node_ids:
                evidence_id = parts[1] if len(parts) > 1 else None

                # Find the claim text that cites this (context around citation)
                start = max(0, match.start() - 100)
                end = min(len(answer_text), match.end() + 100)
                context = answer_text[start:end]

                citations.append(Citation(
                    node_id=node_id,
                    evidence_id=evidence_id,
                    claim_text=context[:200]  # Truncate long context
                ))

        return citations


def create_synthesizer(config: EvaluationConfig) -> Synthesizer:
    """Factory function to create Synthesizer."""
    return Synthesizer(config)
