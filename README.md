# Autonomous Knowledge Discovery

A multi-agent AI scientist for tabular datasets. It generates statistical
hypotheses with an LLM, executes them deterministically through a
domain-specific language (DSL) with no LLM in the execution path, scores the
results for validity and novelty, and accumulates accepted findings in a
persistent **insight graph**.

Six coordinated agents run the discovery loop:

| Agent | Role |
|---|---|
| **Orchestrator** | Picks the round mode and selects anchor insights from the graph. |
| **Historian** | Reads/writes the insight graph, retrieves context around anchors, detects duplicates. |
| **Generator** | Proposes hypotheses via LLM, with a two-phase concept → DSL flow for wide datasets. |
| **CoderRunner** | Compiles and executes hypotheses through the deterministic DSL. |
| **Evaluator** | Scores validity from p-value, effect size, sample size, and replication. |
| **Critic** | Scores novelty (structural + semantic + epistemic surprise) and accepts, refines, or rejects. |

## The DSL

Hypotheses are written in a small statistical DSL and compiled to `statsmodels`
tests, so every result is reproducible:

| Family | Syntax |
|---|---|
| `assoc` | `assoc(X, Y) controlling for C1, C2 weights=W` |
| `diff` | `diff(Y, by=G) controlling for C weights=W` |
| `interact` | `interact(X * Z -> Y) controlling for C weights=W` |
| `heterogeneity` | `heterogeneity(Y ~ X \| G) controlling for C weights=W` |

The compiler enforces pre-flight feasibility checks, strict result validation
(NaN/Inf rejection, finite confidence intervals), and warning-severity
classification for rank deficiency and convergence issues.

## Datasets

The system has been evaluated at three scales:

- **World Values Survey (Wave 7)** — 200+ socio-political variables, 95k respondents
- **SciSciNet CS** — 5M+ papers with 88 bibliometric variables
- **Amazon Books Reviews** — 27M+ reviews with product and reviewer metadata

## Status

Code and documentation are being prepared for release and will be added to this
repository.
