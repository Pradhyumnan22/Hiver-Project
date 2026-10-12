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
- [x] **Checkpoint 2**: Dataset ingestion, preprocessing, and brand selection (`AmazonHelp`)
- [x] **Checkpoint 3**: Intent taxonomy and golden evaluation set (200 examples)
- [x] **Checkpoint 4**: Baselines (Majority-Class & TF-IDF + Logistic Regression)
- [x] **Checkpoint 5**: Historical retrieval with FAISS and relevance thresholding
- [ ] **Checkpoint 6**: AI agent pipeline with structured outputs and deterministic escalation
- [ ] **Checkpoint 7**: Evaluation harness and LLM judge agreement
- [ ] **Checkpoint 8**: Failure analysis, headline number critique, and decision log
- [ ] **Checkpoint 9**: Final reproducibility verification and release

## Baseline Performance Results

Evaluated on held-out **Golden Evaluation Set** (`data/processed/golden_eval_set.json`):

| Model | Classification Accuracy | Macro F1 | Escalation Accuracy | Escalation F1 | Avg Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Majority-Class Baseline** | 10.81% | 0.0279 | 65.95% | 0.0000 | 0.002 ms |
| **TF-IDF + Logistic Regression** | **77.84%** | **0.7497** | **82.16%** | **0.7519** | **4.20 ms** |
| **AI Support Agent (Proposed)** | *(CP 6 & 7)* | *(CP 6 & 7)* | *(CP 6 & 7)* | *(CP 6 & 7)* | *(CP 6 & 7)* |

- Per-class F1 for TF-IDF + LogReg:
  - `subscription_digital`: **0.9796**
  - `returns_refunds`: **0.9167**
  - `order_status_delivery`: **0.8636**
  - `account_access_security`: **0.7778**
  - `damaged_defective_item`: **0.6667**
  - `product_inquiry_availability`: **0.5217**
  - `other_unknown`: **0.5217**
- Detailed metrics saved to [`reports/results.json`](reports/results.json).


## Intent Taxonomy & Golden Evaluation Benchmark

Derived from customer interaction patterns on `AmazonHelp`:
1. `order_status_delivery`: Shipping status, late packages, courier inquiries.
2. `returns_refunds`: Return policies, refund timelines, locker drop-offs.
3. `damaged_defective_item`: Broken, missing, expired, or incorrect products.
4. `account_access_security`: Login issues, unauthorized charges, OTP authentication.
5. `subscription_digital`: Prime memberships, video streaming, digital billing.
6. `product_inquiry_availability`: Item specifications, stock availability, compatibility.
7. `other_unknown`: Unclear, ambiguous, or out-of-domain inquiries.

- **Annotation Guide**: Documented in [`docs/annotation_guide.md`](docs/annotation_guide.md).
- **Golden Set Benchmark**: 200 hand-verified examples stored in `data/processed/golden_eval_set.json` strictly held out from training/dev data.


## Data Preparation & Preprocessing

The preprocessing pipeline targets **AmazonHelp** from the 3M-tweet dataset:
- **Conversation Linkage**: Matches brand response back to parent customer tweet using `in_response_to_tweet_id`.
- **Privacy & Anonymization**: Masks user handles (`@user`), URLs (`<URL>`), emails (`<EMAIL>`), and phone numbers (`<PHONE>`).
- **Leakage Prevention**: Strictly splits partitions by customer `author_id` (80% dev, 20% eval) ensuring zero customer or conversation overlap between the knowledge base and evaluation set.

To reproduce dataset preparation:
```powershell
python scripts/prepare_dataset.py --max-rows 150000 --sample-size 100
```
- Sample fixture: `data/sample/sample_pairs.json` (committed to repository)
- Processed splits: `data/processed/dev_pairs.json` (5,124 pairs) and `data/processed/eval_pairs.json` (1,292 pairs)

