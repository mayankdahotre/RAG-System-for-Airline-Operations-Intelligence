"""
Enterprise Configuration for Multimodal RAG Airline Operations System
"""
import os
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class EmbeddingConfig:
    model_name: str = "text-embedding-3-small"
    dimension: int = 1536
    batch_size: int = 100
    max_retries: int = 3


@dataclass
class LLMConfig:
    model_name: str = "gpt-4-turbo-preview"
    temperature: float = 0.1  # Low temp for factual ops responses
    max_tokens: int = 2048
    timeout_seconds: int = 30
    streaming_enabled: bool = True


@dataclass
class RetrievalConfig:
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    top_k: int = 10
    rerank_top_k: int = 5
    similarity_threshold: float = 0.75
    metadata_boost: float = 1.2  # Boost for fleet/airport matches


@dataclass
class GroundingConfig:
    confidence_threshold: float = 0.78
    min_citations_required: int = 1
    abstention_message: str = "Insufficient operational evidence to provide a reliable answer."
    require_exact_match: bool = False


@dataclass
class LatencyBudget:
    retrieval_ms: int = 200
    generation_ms: int = 3000
    total_ms: int = 5000
    warn_threshold_pct: float = 0.8


@dataclass
class VectorStoreConfig:
    faiss_index_path: str = "vectorstore/faiss_index"
    bm25_index_path: str = "vectorstore/bm25_index"
    chunk_size: int = 512
    chunk_overlap: int = 50


@dataclass
class Config:
    environment: Environment = Environment.DEVELOPMENT
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    grounding: GroundingConfig = field(default_factory=GroundingConfig)
    latency: LatencyBudget = field(default_factory=LatencyBudget)
    vectorstore: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    
    # API Keys (from environment)
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    
    # Airline-specific settings
    supported_fleets: list = field(
        default_factory=lambda: ["B737", "B777", "B787", "A320", "A350"]
    )
    supported_airports: list = field(
        default_factory=lambda: ["ORD", "EWR", "LAX", "SFO", "IAH", "DEN"]
    )
    
    def validate(self) -> bool:
        """Validate configuration for production readiness."""
        if self.environment == Environment.PRODUCTION:
            assert self.openai_api_key, "OpenAI API key required in production"
            assert self.grounding.confidence_threshold >= 0.75, "Confidence threshold too low for production"
        return True


# Global config instance
settings = Config()

