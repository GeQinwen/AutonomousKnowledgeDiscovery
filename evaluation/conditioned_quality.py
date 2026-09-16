#!/usr/bin/env python3
"""Conditioned-quality evaluation (Section 5.2).

Each literature-derived research question is answered from the insight graph
by the knowledge-extraction pipeline (Section 4.2): seed-and-expand retrieval
of the top-K insights, a synthesized answer that cites them, and faithfulness
verification.  The output is a review sheet on which a human rater scores each
query against its gold claim:

  MC  More-than-covered: the graph contains the claim plus additional consistent
      evidence or detail that extends it.
  C   Covered: a directional match on the same variables.
  P   Partially covered: a related finding sharing some but not all of the
      claim's variables or conditions, or matching direction on a proxy.
  NC  Not covered: no matching or related finding is retrievable.

Queries are read from JSONL, one per line:
  {"query_id": "wvs_q1", "query": "<research question>", "gold_claim": "<published finding>"}

Usage:
  python evaluation/conditioned_quality.py --graph data/memory_wvs/insight_graph.json \\
      --queries queries_wvs.jsonl --out conditioned_wvs.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import config  # noqa: E402
from memory.insight_graph import InsightGraph  # noqa: E402
from evaluation.eval_types import EvaluationConfig, Query, RetrievedNode  # noqa: E402
from evaluation.retrieval.seed_expand_retriever import SeedExpandConfig, SeedExpandRetriever  # noqa: E402
from evaluation.synthesis.synthesizer import Synthesizer  # noqa: E402
from evaluation.verification.critic import AnswerCritic  # noqa: E402

RUBRIC = [
    ("MC", "More-than-covered", "the graph contains the claim plus additional consistent evidence or detail that extends it"),
    ("C", "Covered", "a directional match on the same variables"),
    ("P", "Partially covered", "a related finding sharing some but not all of the claim's variables or conditions, "
                               "or matching direction on a proxy"),
    ("NC", "Not covered", "no matching or related finding is retrievable"),
]


def load_queries(path: str) -> List[dict]:
    queries = []
    for line_no, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        missing = {"query_id", "query", "gold_claim"} - set(record)
        if missing:
            raise SystemExit(f"{path}:{line_no} is missing {sorted(missing)}")
        queries.append(record)
    return queries


def main() -> None:
    ap = argparse.ArgumentParser(description="AutoKD conditioned-quality evaluation")
    ap.add_argument("--graph", required=True, help="Insight graph JSON")
    ap.add_argument("--queries", required=True, help="JSONL with query_id, query, gold_claim")
    ap.add_argument("--out", required=True, help="Markdown review sheet to write")
    ap.add_argument("--k", type=int, default=5, help="Insights retrieved per query")
    ap.add_argument("--jsonl", help="Also write per-query results as JSONL")
    ap.add_argument("--placeholder", action="store_true", help="Use placeholder LLM (no Ollama)")
    args = ap.parse_args()

    if not Path(args.graph).exists():
        raise SystemExit(f"graph not found: {args.graph}")
    if args.placeholder:
        config.config.setdefault("llm", {})["provider"] = "placeholder"

    graph = InsightGraph(storage_path=args.graph)
    retriever = SeedExpandRetriever(graph, SeedExpandConfig())
    synthesizer = Synthesizer(EvaluationConfig())
    critic = AnswerCritic()

    lines = [f"# Conditioned quality review: {Path(args.graph).parent.name}", "",
             "Rate each query against its gold claim:", ""]
    lines += [f"- **{code} ({name})**: {text}." for code, name, text in RUBRIC]
    results = []

    for q in load_queries(args.queries):
        items = retriever.retrieve(q["query"], k=args.k)
        nodes = [
            RetrievedNode(
                node_id=item.insight.id,
                insight_sentence=item.insight.sentence,
                retrieval_rank=item.rank,
                similarity_score=item.similarity,
                metadata={"validity_score": item.validity, "retrieval_score": item.score},
            )
            for item in items
        ]
        synthesis = synthesizer.synthesize(Query(query_id=q["query_id"], query_text=q["query"]), nodes)
        verification = critic.verify(synthesis)
        n_claims = len(verification.supported_claims) + len(verification.unsupported_claims)

        lines += ["", f"## {q['query_id']}", "",
                  f"**Query:** {q['query']}", "",
                  f"**Gold claim:** {q['gold_claim']}", "",
                  "**Retrieved insights:**", "",
                  "| Rank | Insight | Similarity | Validity |", "|---:|---|---:|---:|"]
        lines += [f"| {n.retrieval_rank} | [{n.node_id}] {n.insight_sentence} | "
                  f"{n.similarity_score:.3f} | {n.metadata['validity_score']:.3f} |" for n in nodes]
        lines += ["", "**Answer:**", "", synthesis.answer.strip(), "",
                  f"**Faithfulness:** {verification.faithfulness_score:.2f} "
                  f"({len(verification.supported_claims)}/{n_claims} claims supported)"]
        if verification.unsupported_claims:
            lines += ["", "Unsupported claims:"] + [f"- {c}" for c in verification.unsupported_claims]
        lines += ["", "**Rating:** " + "  ".join(f"☐ {code}" for code, _, _ in RUBRIC)]

        results.append({
            "query_id": q["query_id"],
            "query": q["query"],
            "gold_claim": q["gold_claim"],
            "retrieved": [{"id": n.node_id, "sentence": n.insight_sentence, "rank": n.retrieval_rank,
                           "similarity": n.similarity_score, **n.metadata} for n in nodes],
            "answer": synthesis.answer,
            "faithfulness": verification.faithfulness_score,
            "hallucination_rate": verification.hallucination_rate,
            "unsupported_claims": verification.unsupported_claims,
        })
        print(f"[{q['query_id']}] retrieved={len(nodes)} faithfulness={verification.faithfulness_score:.2f}")

    Path(args.out).write_text("\n".join(lines) + "\n")
    if args.jsonl:
        Path(args.jsonl).write_text("".join(json.dumps(r) + "\n" for r in results))
    print(f"Review sheet -> {args.out}")


if __name__ == "__main__":
    main()
