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
