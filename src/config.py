"""Configuration loading and validation."""
from pathlib import Path
from typing import Any, Dict, Optional
import os
import yaml
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Automatically load .env if present
load_dotenv()


class ProjectSettings(BaseModel):
    name: str = "hiver-support-agent"
    seed: int = 42
    environment: str = "development"


class DataSettings(BaseModel):
    selected_brand: str = "AmazonHelp"
    raw_dataset_path: str = "data/raw/twcs.csv"
    processed_dir: str = "data/processed"
    sample_dir: str = "data/sample"
    golden_set_path: str = "data/processed/golden_eval_set.json"
    dev_split_ratio: float = 0.8
    eval_split_ratio: float = 0.2


class RetrievalSettings(BaseModel):
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    faiss_index_path: str = "data/processed/knowledge_base.index"
    metadata_path: str = "data/processed/knowledge_metadata.json"
    top_k: int = 3
    relevance_threshold: float = 0.65


class AgentSettings(BaseModel):
    llm_model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    temperature: float = 0.0
    max_tokens: int = 300
    escalation_confidence_threshold: float = 0.70
    api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    base_url: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))


class EvaluationSettings(BaseModel):
    results_path: str = "reports/results.json"
    judge_model: str = Field(default_factory=lambda: os.getenv("JUDGE_MODEL", "gpt-4o-mini"))
    max_eval_latency_seconds: float = 15.0


class AppConfig(BaseModel):
    project: ProjectSettings = Field(default_factory=ProjectSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)


def load_config(config_path: Optional[str | Path] = None) -> AppConfig:
    """Load and validate configuration from YAML file and environment variables."""
    if config_path is None:
        default_yaml = Path(__file__).parent.parent / "configs" / "default_config.yaml"
        if default_yaml.exists():
            config_path = default_yaml

    data_dict: Dict[str, Any] = {}
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            raw_yaml = yaml.safe_load(f)
            if isinstance(raw_yaml, dict):
                data_dict = raw_yaml

    return AppConfig(**data_dict)
