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
  - Prepare exactly 200 real examples strictly from the held-out
    `eval_pairs.json` partition using deterministic seed 42.
  - De-duplicate by customer tweet ID before stratified sampling because one
    customer tweet can have multiple historical brand replies.
  - Treat keyword-generated labels as preliminary. Review all 200 rows
    semantically in four separate batches, preserve whether review was AI or
    human, and require explicit review provenance before publishing JSON.
  - Preserve ambiguous-case notes and reviewer metadata in a committed CSV
    audit trail.
  - Automated tests assert zero overlap between golden set customer
    tweet/author IDs and development data.
- **Rationale**: The original draft contained 185 rows but only 167 unique
  customer tweets and incorrectly described heuristic labels as hand
  verified. The corrected workflow prevents duplicate weighting and makes the
  provenance of every final label auditable. The current set is explicitly
  AI-reviewed; independent human validation remains a documented limitation.
- **Alternatives Rejected**: Synthetically generated evaluation examples (fails to reflect real customer spelling, brevity, and frustration) or evaluating on the development set (uncalibrated overoptimistic metrics).

---

### Decision 7: Baseline Model Selection (Majority Class & TF-IDF + Logistic Regression)
- **Date**: 2026-10-12
- **Context**: Establishing credible, reproducible baselines against which the proposed AI pipeline will be judged.
- **Decision**:
  1. **Majority Class Baseline**: Serves as the minimal empirical anchor
     (Accuracy: 19.50%, Macro F1: 0.0466 on the corrected 200-example set).
  2. **TF-IDF + Multinomial Logistic Regression**: Traditional linear text
     baseline with sublinear TF scaling and balanced class weighting
     (Accuracy: 56.00%, Macro F1: 0.5178).
  3. Development labels are deterministic weak supervision, while evaluation
     labels received separate semantic review. Results therefore measure
     transfer away from the labelling rules rather than supervised learning
     from equivalent-quality human labels.
- **Rationale**: Demonstrates how much improvement modern LLM semantic representations and retrieval grounding actually provide over a fast (4.2ms) classical ML model, providing an honest comparison for interview defense.
- **Alternatives Rejected**: Evaluating solely against zero-shot LLM without standard machine learning baselines (violates good scientific practice and conceals actual incremental value).

---

### Decision 8: Simple Retrieval Reply Baseline and Similarity Thresholding Need
- **Date**: 2026-10-12
- **Context**: Evaluating naive historical reply reuse prior to building the semantic FAISS index.
- **Decision**: Implemented `SimpleRetrievalReplyBaseline` using TF-IDF sparse vector cosine similarity over 5,124 historical pairs.
- **Rationale**: On all 200 corrected evaluation queries, mean top-1 similarity
  was 0.3361 and only 31.0% met the 0.35 relevance threshold. Traceable
  examples show contextually misleading replies below and sometimes above
  this cutoff. This provides empirical justification for thresholding and
  separate response-quality evaluation rather than treating retrieval as
  successful merely because a nearest neighbour exists.
- **Alternatives Rejected**: Unconditioned template generation.

---

### Decision 9: FAISS Vector Indexing and Metric Inner-Product Cosine Similarity
- **Date**: 2026-10-12
- **Context**: Selecting vector search backend for retrieving top-k historical support interactions.
- **Decision**: Implemented `FAISSRetriever` with `IndexFlatIP` on L2-normalized embeddings, guaranteeing exact cosine similarity computation with sub-millisecond query latency.
- **Rationale**:
  - `faiss-cpu` provides deterministic, production-grade vector search with zero external service dependencies.
  - Inner Product of normalized vectors is mathematically identical to cosine similarity, making relevance scores directly interpretable between 0.0 and 1.0.
  - Encapsulated within `DenseEmbedder` with offline resilience for testing environments.
- **Alternatives Rejected**: Heavy managed vector databases (Pinecone, Qdrant, Milvus) which introduce unnecessary infrastructure complexity and external latency for an embedded support agent.

---

### Decision 10: Leakage Exclusion Filter and Relevance Thresholding
- **Date**: 2026-10-12
- **Context**: Ensuring held-out evaluation pairs are never indexed or retrieved, and preventing hallucinated grounding on low-similarity queries.
- **Decision**:
  1. **Strict Exclusion Filter**: Filter held-out author IDs, tweet IDs, and
     normalized text fingerprints before indexing. De-duplicate remaining
     development text by the same fingerprint.
  2. **Relevance Thresholding**: Calibrate on a development-only 80/20 author
     split using 500 queries. Select the highest-coverage candidate meeting
     80% top-1 same-intent proxy precision and 10% minimum coverage.
  3. The resulting threshold is **0.65**, with 81.51% proxy precision and
     71.40% coverage. Persist the complete threshold curve in
     `reports/retrieval_calibration.json`.
  4. If no result meets 0.65, return an explicit empty result so the later
     escalation layer cannot treat weak evidence as grounding.
- **Rationale**: Replaces the original unvalidated 0.45 default with a
  reproducible development-only operating point and reduces evaluation-text
  leakage. Same-intent agreement is acknowledged as a weak proxy rather than
  a human relevance label.
- **Alternatives Rejected**: Unchecked top-k retrieval (always forcing k results regardless of relevance).

---

### Decision 11: Structured LLM Boundary Without Confidence Probabilities
- **Date**: 2026-10-13
- **Context**: Constraining intent classification and reply generation while
  handling malformed responses and provider failures.
- **Decision**: Use direct OpenAI-compatible structured JSON output validated
  by Pydantic. Intent output contains a taxonomy value, rationale, and optional
  uncertainty—but no numerical confidence. Reply output contains a bounded
  draft, evidence IDs, and optional uncertainty.
- **Rationale**: Provider-generated confidence is not calibrated probability.
  Schema validation provides a testable boundary, and any invalid output or
  API error fails closed to human escalation.
- **Alternatives Rejected**: Free-form parsing, trusting self-reported LLM
  confidence, and adding an orchestration framework for two model calls.

---

### Decision 12: Deterministic Escalation Before and After Generation
- **Date**: 2026-10-13
- **Context**: Preventing the model from deciding whether its own output is
  safe to send.
- **Decision**: Apply explicit rules before generation for high-risk language,
  unsupported intent, account verification, transaction verification,
  repeated unresolved history, and insufficient retrieval evidence. After
  generation, require cited IDs to be a subset of retrieved evidence and
  reject phrases that claim completed actions, private account knowledge, or
  guaranteed outcomes.
- **Rationale**: Safety decisions remain deterministic, auditable, and
  credential-free in tests. Escalated cases return a stable reason code plus a
  plain-language explanation.
- **Alternatives Rejected**: Asking the same LLM to approve its own draft or
  auto-handling when retrieval is empty.


