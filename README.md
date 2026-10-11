# Hiver AI Customer Support Agent

A reproducible, grounded AI customer-support agent and rigorous evaluation pipeline built on Kaggle's Customer Support on Twitter dataset.

## Architecture Overview

```
Customer Message
       │
       ▼
┌───────────────────────────────┐
│ 1. Intent Classification      │  ◄── Trivial Majority, TF-IDF + Logistic Regression, & LLM Classifier
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│ 2. Historical Retrieval       │  ◄── FAISS Semantic Index with Relevance Threshold & Leakage Exclusion
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│ 3. Deterministic Escalation   │  ◄── Rules Engine: High-Risk, Low Retrieval, Out-of-Scope Intents
└──────────────┬────────────────┘
               │
         ┌─────┴─────┐
         ▼           ▼
   [auto_handle]   [escalate]
         │           │
         ▼           ▼
  Grounded Reply  Human Triage Reason
```

## Features

- **Strict Leakage Prevention**: Held-out evaluation and golden sets are strictly segregated before building retrieval indices or training classifiers.
- **Deterministic Escalation Engine**: Independent safety and policy layer; the LLM does not decide its own safety.
- **Multi-Tier Baseline Comparisons**: Evaluated against Majority-class baseline and TF-IDF + Logistic Regression baseline.
- **LLM-as-Judge & Human Agreement**: Documented rubric measuring groundedness, relevance, helpfulness, consistency, and safety with Cohen's Kappa agreement tracking.
- **Zero-Secret Architecture**: Full test suite and baseline evaluations run offline without external API keys or proprietary servers.

## Quickstart

### 1. Environment Setup

```powershell
# Create and activate virtual environment (Python 3.11 recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Copy `.env.example` to `.env` and supply your OpenAI or compatible LLM credentials:
```powershell
cp .env.example .env
```

### 3. Run Tests

```powershell
pytest -v
```

## Checkpoint Status

- [x] **Checkpoint 0**: Workspace and Git preflight
- [x] **Checkpoint 1**: Project skeleton, configuration, logging, and smoke tests
- [ ] **Checkpoint 2**: Dataset ingestion, preprocessing, and brand selection
- [ ] **Checkpoint 3**: Intent taxonomy and golden evaluation set
- [ ] **Checkpoint 4**: Baselines (Majority-Class & TF-IDF + Logistic Regression)
- [ ] **Checkpoint 5**: Historical retrieval with FAISS and relevance thresholding
- [ ] **Checkpoint 6**: AI agent pipeline with structured outputs and deterministic escalation
- [ ] **Checkpoint 7**: Evaluation harness and LLM judge agreement
- [ ] **Checkpoint 8**: Failure analysis, headline number critique, and decision log
- [ ] **Checkpoint 9**: Final reproducibility verification and release
