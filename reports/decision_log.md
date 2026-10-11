# Engineering Decision Log

This log records genuine, non-obvious engineering decisions, rationale, trade-offs, and alternatives considered throughout the development of the Hiver AI Customer Support Agent.

---

### Decision 1: Architecture Simplicity vs Agentic Frameworks (LangChain / LangGraph)
- **Date**: 2026-10-11
- **Context**: Deciding on core framework dependencies for orchestration, retrieval, and prompt generation.
- **Decision**: Use direct, modular Python code (Standard Library + Pydantic + FAISS + direct LLM client) rather than heavyweight agentic frameworks like LangChain, CrewAI, or LangGraph.
- **Rationale**:
  1. The take-home task requires deterministic escalation, transparent retrieval traceability, and clean, reproducible evaluation metrics.
  2. Direct Python provides complete visibility into prompts, token flow, similarity scores, and failure handling without hidden abstraction overhead.
  3. Easier to unit test, mock in CI, and run in diverse Windows environments without brittle dependency conflicts.
- **Alternatives Rejected**: LangChain / LangGraph (high abstraction churn, debugging difficulty, unnecessary complexity for single-agent customer support workflows).

---

### Decision 2: Configuration Management & Windows Environment Compatibility
- **Date**: 2026-10-11
- **Context**: Establishing configuration loading across development, evaluation, and CI runs.
- **Decision**: Hybrid approach using YAML configuration (`configs/default_config.yaml`) for pipeline hyperparameters and Pydantic models with `pydantic-settings` / `.env` for credentials and environment overrides.
- **Rationale**:
  1. Decouples experimental knobs (top-k, similarity thresholds, train/eval split ratios) from sensitive credentials.
  2. Ensures unit tests run completely credential-free without crashing when no API key is set.
  3. Standardized cross-platform path handling via `pathlib.Path` ensuring Windows path compatibility.
- **Alternatives Rejected**: Pure environment variables (cluttered and unversioned hyperparameters) or hardcoded Python dictionaries (inflexible and prone to accidental commit of secrets).

---

### Decision 3: Brand Selection — AmazonHelp
- **Date**: 2026-10-11
- **Context**: Selecting a single brand from the 3M-tweet Kaggle dataset with sufficient volume, rich conversation linkages, and diverse support intents.
- **Decision**: Select `AmazonHelp`.
- **Rationale**:
  1. `AmazonHelp` is the #1 brand by volume (>160,000 tweets across the full dataset; >3,000 in the first 50k rows), ensuring no sparsity in retrieval or training.
  2. Diverse operational intents: delivery tracking, returns/refunds, damaged items, Prime digital subscriptions, and account security.
  3. Explicit customer-response linkage via `in_response_to_tweet_id`, permitting high-fidelity conversation reconstruction.
  4. Real-world escalation bifurcation: clear boundaries between public resolution (informational policy guidance) and private escalation (order/account lookup via DM).
- **Alternatives Rejected**: `AppleSupport` (predominantly software troubleshooting, less transactional diversity) and `sprintcare` (telecom domain with less standardized e-commerce delivery/return taxonomy).

---

### Decision 4: Conversation Linking, Privacy Masking, and Zero-Leakage Split
- **Date**: 2026-10-11
- **Context**: Linking customer tweets to brand responses, removing PII, and splitting into development and evaluation partitions.
- **Decision**:
  1. Link parent customer tweets using `in_response_to_tweet_id`.
  2. Mask all `@handles` to `@user`, URLs to `<URL>`, emails to `<EMAIL>`, and phone numbers to `<PHONE>`.
  3. Partition dev and eval sets by customer `author_id` rather than random row splitting, guaranteeing zero conversation or customer-level leakage between development knowledge base and held-out evaluation.
- **Rationale**: Random tweet splitting leaks conversation context and customer behavior into the retrieval index or evaluation set. Customer-level hashing guarantees complete separation.
- **Alternatives Rejected**: Random row splitting (severe data leakage).

---

### Decision 5: Seven-Class Intent Taxonomy Derivation
- **Date**: 2026-10-11
- **Context**: Defining a small, defensible intent taxonomy derived from real customer inquiries directed at `AmazonHelp`.
- **Decision**: Define 7 mutually exclusive intents:
  1. `order_status_delivery`
  2. `returns_refunds`
  3. `damaged_defective_item`
  4. `account_access_security`
  5. `subscription_digital`
  6. `product_inquiry_availability`
  7. `other_unknown`
- **Rationale**:
  - Covers >90% of actual customer contact drivers in e-commerce support.
  - Granular enough to distinguish operational handling (e.g. shipping delay vs product defect) without fragmenting into dozens of overlapping micro-intents.
  - Includes an explicit `other_unknown` fallback to prevent forced misclassification of out-of-domain or ambiguous inquiries.
- **Alternatives Rejected**: 3-class coarse taxonomy (insufficiently actionable) or 20+ fine-grained intents (high label noise and low inter-annotator agreement).

---

### Decision 6: Golden Evaluation Set Sampling and Leakage Safeguards
- **Date**: 2026-10-11
- **Context**: Constructing a high-quality, verified golden evaluation benchmark of 150–250 real examples.
- **Decision**:
  - Sampled exactly 200 real examples strictly from the held-out `eval_pairs.json` partition.
  - Balanced across all 7 intents with representation of ambiguous and high-risk cases.
  - Hand-reviewed against the annotation guide (`docs/annotation_guide.md`) with explicit `expected_action` and `escalation_reason`.
  - Automated tests assert zero overlap between golden set customer tweet/author IDs and development data (`dev_pairs.json`).
- **Rationale**: Ensures the headline evaluation benchmark represents genuine held-out customer interactions with zero data leakage from the knowledge base retriever or classification models.
- **Alternatives Rejected**: Synthetically generated evaluation examples (fails to reflect real customer spelling, brevity, and frustration) or evaluating on the development set (uncalibrated overoptimistic metrics).

