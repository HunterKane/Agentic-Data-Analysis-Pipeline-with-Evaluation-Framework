# Agentic Data Analysis Pipeline with Evaluation Framework

> A Claude-powered autonomous agent that performs end-to-end exploratory data analysis — generating hypotheses, selecting and executing statistical tests, and synthesizing insights — evaluated against a structured benchmark that scores statistical accuracy, hallucination rate, reasoning coherence, and test selection appropriateness.

---

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Evaluation Framework](#evaluation-framework)
- [Results Summary](#results-summary)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Notebook Walkthrough](#notebook-walkthrough)
- [Key Findings & Failure Modes](#key-findings--failure-modes)
- [Design Decisions](#design-decisions)
- [Limitations & Future Work](#limitations--future-work)
- [References](#references)

---

## Overview

This project builds and rigorously evaluates an **agentic LLM pipeline** for automated exploratory data analysis (EDA). Rather than using Claude as a text generator, it is deployed as a tool-calling agent operating in a closed reasoning loop — selecting statistical tests, executing code, observing results, and iterating until a set of evidenced findings can be synthesized.

The core contribution is not just the agent itself, but the **evaluation framework** used to benchmark its outputs against known ground truths. Every analytical claim made by the agent is scored across four dimensions, enabling systematic identification of failure modes and measurable improvement via prompt engineering and validation layers.

**What the agent does autonomously:**
- Profiles a raw dataset (types, distributions, cardinality, nulls)
- Generates a ranked list of exploratory hypotheses
- Selects and executes appropriate statistical tests
- Interprets results with explicit uncertainty quantification
- Synthesizes structured insights with supporting evidence

**What the evaluation framework measures:**
- Whether selected tests are statistically appropriate for the data and hypothesis
- Whether stated findings are supported by actual test results (hallucination detection)
- Whether the reasoning chain is internally consistent
- Whether conclusions correctly match known ground truths

---

## Motivation

LLMs are increasingly used for data analysis, but their outputs are rarely held to the same standard of evidence as traditional statistical workflows. Two failure patterns are particularly dangerous in practice:

1. **Hallucinated insights** — the model asserts a finding (e.g., "variable X strongly predicts Y") without having run a test to support it
2. **Test misapplication** — the model selects a parametric test for non-normal data, or runs a correlation on ordinal variables without acknowledging the limitation

Both failures are hard to detect without a systematic evaluation layer, since the outputs are fluent and often plausible-sounding. This project treats those failure modes as first-class engineering problems and documents the before/after impact of targeted fixes.

The broader goal is to establish a repeatable methodology for evaluating agentic analytical systems — applicable beyond this specific dataset or use case.

---

## Project Structure

```
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── agentic_eda_pipeline.ipynb       # Main notebook — full walkthrough
├── src/
│   ├── agent/
│   │   ├── tools.py                     # Tool schema definitions (5 tools)
│   │   ├── loop.py                      # Agent loop controller (max_turns, reflection)
│   │   └── prompts.py                   # System prompt versions (v1 baseline, v2 fixed)
│   ├── evaluation/
│   │   ├── scorer.py                    # Rubric-based scoring across 4 metrics
│   │   ├── hallucination_checker.py     # Cross-references claims vs tool_results
│   │   └── ground_truth.py             # Injected anomalies + known correlations
│   ├── data/
│   │   ├── loader.py                    # Dataset loading and metadata generation
│   │   └── inject_ground_truth.py      # Injects artificial signals into dataset
│   └── utils/
│       ├── stats.py                     # Shared statistical helpers (BH correction, etc.)
│       └── visualization.py            # Plot generation utilities
├── data/
│   └── README.md                        # Dataset provenance and injection log
├── results/
│   ├── baseline_runs/                   # Raw JSON outputs from N=30 baseline runs
│   ├── fixed_runs/                      # Raw JSON outputs from N=30 fixed runs
│   └── metrics_summary.csv             # Aggregated before/after scores
└── tests/
    ├── test_tools.py
    ├── test_scorer.py
    └── test_hallucination_checker.py
```

---

## Architecture

The pipeline consists of four layers that operate in sequence:

### 1. Data Profiling Layer
Ingests a raw dataset and computes a structured metadata object: shape, per-column dtype, null density, cardinality, descriptive statistics, and detected semantic type (continuous, categorical, ordinal, datetime, identifier). This metadata is passed as structured context to the agent — keeping prompts compact and avoiding raw data injection.

### 2. Agentic EDA Loop
Claude receives the metadata and enters a tool-calling loop. At each turn it may call one of five tools:

| Tool | Purpose |
|------|---------|
| `get_column_stats(col, stat_type)` | Structured column analysis; returns JSON |
| `run_statistical_test(test, col_a, col_b, params)` | Validated test execution; returns p-value, effect size, CI |
| `plot_distribution(col, plot_type, group_by)` | Distribution visualization |
| `check_assumptions(col, test_type)` | Normality, variance homogeneity, sample size checks |
| `summarize_findings(findings_list)` | Terminal tool; forces structured output, ends the loop |

The loop runs for a maximum of 12 turns. A reflection prompt is injected at turn 6, asking the agent to audit coverage and identify any hypotheses not yet addressed.

### 3. Structured Output Layer
All findings are emitted via `summarize_findings` in a strict JSON schema:
```json
{
  "finding": "string",
  "supporting_test": "test_name",
  "test_result": { "statistic": float, "p_value": float, "effect_size": float },
  "confidence": "Low | Medium | High",
  "implication": "string"
}
```
JSON-mode enforcement prevents free-form text findings that evade the evaluation layer.

### 4. Validation Layer
A post-hoc parser cross-references every `finding` field against the `tool_result` history. Any finding asserted without a matching `run_statistical_test` call in the same session is flagged as an unsupported claim (hallucination). This layer was added in v2 and feeds back into the evaluation loop.

---

## Evaluation Framework

### Dataset & Ground Truth

The [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing) is used as the base: 45,211 rows, 16 features (mix of numeric and categorical), a binary target, known class imbalance, and a date feature — all properties that stress-test the agent's test selection logic.

Before evaluation, three artificial anomalies and two deliberate correlations are injected into the dataset using `inject_ground_truth.py`. The agent is never told what was injected; its job is to find them. This creates an objective, unambiguous scoring basis.

### Scoring Rubric (per run)

**Statistical accuracy** — For each test the agent runs: does the selected test family match the data types and hypothesis? Is the p-value computed correctly (within floating-point tolerance)? Is multiple comparison correction applied when n_tests > 3? Score: fraction of tests fully correct.

**Hallucination rate** — Fraction of stated findings with no supporting `tool_result` in the session. Computed deterministically by the validation layer — no human judgment required.

**Reasoning coherence** — LLM-as-judge score (1–5) on the internal consistency of the reasoning chain: do conclusions follow from observations? Are assumptions stated? Are contradictions acknowledged? Averaged across a panel of 3 judge calls per run.

**Test selection appropriateness** — For each statistical test call, the selected test is compared against an expert-labeled ground truth decision tree. Score: fraction of calls where the correct test family was chosen for the given column types and hypothesis structure.

### Experimental Protocol
- N=30 independent runs per condition (baseline prompt v1, fixed prompt v2)
- Each run uses the same dataset with the same injected ground truths
- All raw outputs stored in `results/` for reproducibility
- Statistical significance of before/after differences tested with a paired Wilcoxon signed-rank test

---

## Results Summary

| Metric | Baseline (v1) | Fixed (v2) | Δ | p-value |
|--------|--------------|------------|---|---------|
| Statistical accuracy | 52% | 87% | +35pp | < 0.001 |
| Hallucination rate | 34% | 8% | −26pp | < 0.001 |
| Reasoning coherence (1–5) | 2.8 | 4.4 | +1.6 | < 0.001 |
| Test selection appropriateness | 61% | 93% | +32pp | < 0.001 |

The largest single-source improvement came from adding the normality check gate before test selection (+18pp on statistical accuracy alone). Hallucination rate was almost entirely resolved by the JSON validation layer — demonstrating that structural enforcement is more reliable than prompt-level instruction for this failure mode.

---

## Setup & Installation

**Requirements:** Python 3.10+, an Anthropic API key.

```bash
# Clone the repository
git clone https://github.com/your-username/agentic-eda-pipeline.git
cd agentic-eda-pipeline

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your API key
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Key dependencies:**
```
anthropic>=0.40.0
pandas>=2.0.0
scipy>=1.11.0
statsmodels>=0.14.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
```

---

## Usage

### Run the full pipeline on a dataset
```python
from src.agent.loop import run_eda_agent
from src.data.loader import load_and_profile

metadata = load_and_profile("data/bank-marketing.csv")
result = run_eda_agent(metadata, max_turns=12)

print(result.findings)        # Structured findings list
print(result.reasoning_trace) # Full tool call history
```

### Run the evaluation harness
```python
from src.evaluation.scorer import evaluate_run
from src.evaluation.ground_truth import load_ground_truths

ground_truths = load_ground_truths()
scores = evaluate_run(result, ground_truths)

print(scores)
# {
#   "statistical_accuracy": 0.87,
#   "hallucination_rate": 0.06,
#   "reasoning_coherence": 4.3,
#   "test_selection_appropriateness": 0.91
# }
```

### Reproduce the full N=30 benchmark
```bash
python scripts/run_benchmark.py --n 30 --prompt-version v2 --output results/fixed_runs/
python scripts/aggregate_results.py --input results/ --output results/metrics_summary.csv
```

---

## Notebook Walkthrough

The primary deliverable is `notebooks/agentic_eda_pipeline.ipynb`. Sections:

1. **Objective & Motivation** — Problem framing, why autonomous EDA is hard, what rigorous evaluation requires
2. **Dataset & Ground Truth Injection** — Dataset selection rationale, injection methodology, verification that injected signals are detectable
3. **Agent Architecture** — Tool schema walkthrough, system prompt design choices, loop control logic
4. **Baseline Agent Run** — Live agent run with annotated reasoning trace and visualizations
5. **Evaluation Framework** — Rubric definitions, scoring implementation, N=30 baseline results with distribution plots
6. **Failure Mode Analysis** — Taxonomy of failure modes with representative examples from the run logs
7. **Fixes Applied** — Prompt v1 → v2 diff with rationale, validation layer implementation, component-level attribution of improvement
8. **Before vs. After Results** — Full metric comparison, statistical significance testing, residual failure analysis
9. **Conclusion** — Key learnings, production readiness assessment, open problems

---

## Key Findings & Failure Modes

**Failure 1: Parametric test applied to skewed data**
The baseline agent applied a t-test to right-skewed income data without checking normality. Fix: a mandatory `check_assumptions` gate before any parametric test call.

**Failure 2: Correlation interpreted as causation**
The agent stated "X causes Y" based solely on a Pearson correlation coefficient. Fix: explicit language constraint in the system prompt prohibiting causal language without explicit caveating.

**Failure 3: Hallucinated insights**
In 34% of baseline runs, the agent asserted findings before calling the relevant test — the reasoning loop short-circuited to a conclusion. Fix: post-hoc JSON validation layer that rejects any finding without a matching tool result. Prompt-level instruction alone reduced this to ~22%; the structural validation layer brought it to 8%.

**Failure 4: Multiple comparison inflation**
The agent ran 8 independent tests at α=0.05 without correction, producing an expected 0.4 false positives per run by chance. Fix: automatic Benjamini-Hochberg correction injected when n_tests > 3.

**Failure 5: Premature loop termination**
The agent stopped at 3 findings and missed 4 of the 5 injected ground truths. Fix: midpoint reflection prompt at turn 6 that explicitly asks the agent to audit its coverage and identify unaddressed hypotheses.

---

## Design Decisions

**Why UCI Bank Marketing?** It has known statistical properties, a mix of numeric and categorical features, meaningful class imbalance, and has been used in published ML papers — making its properties verifiable. Datasets without known properties make ground-truth evaluation impossible.

**Why JSON-mode enforcement for findings?** Prose findings are not machine-parseable for the hallucination checker. Structured schema was the minimal intervention that made evaluation deterministic without requiring human review of each run.

**Why BH correction rather than Bonferroni?** Bonferroni is conservative to the point of masking real signals at n_tests=8. BH controls the false discovery rate at a more appropriate operating point for exploratory analysis where some false positives are tolerable but false negatives are expensive.

**Why max_turns=12 with a midpoint reflection?** Uncapped loops produce diminishing returns after ~8 turns and occasionally enter cycles. A hard cap at 12 was set empirically from pilot runs. The turn-6 reflection prompt addresses premature termination without requiring a higher cap.

---

## Limitations & Future Work

**Current limitations:**
- Evaluation relies on a single dataset; generalization to other domains is not validated
- The LLM-as-judge coherence scorer introduces variance — inter-run agreement is ~0.4 Spearman correlation
- The agent has no memory across sessions; long EDA tasks that require revisiting earlier findings are not well-supported

**Future work:**
- Multi-dataset evaluation across at least 3 domain types (financial, clinical, operational)
- Replace LLM coherence judge with a deterministic logic-checker where possible
- Extend the tool suite with causal inference support (e.g., propensity score matching, IV estimation)
- Evaluate prompt sensitivity: how much do results change across N=10 different prompt phrasings of the same constraints?

---

## References

- Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems*, 62, 22–31.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*, 57(1), 289–300.
- Anthropic. (2024). [Tool use documentation](https://docs.anthropic.com/en/docs/build-with-claude/tool-use).
- Shankar, S. et al. (2024). Who validates the validators? Towards evaluating LLM-based evaluation. *arXiv:2404.12272*.
