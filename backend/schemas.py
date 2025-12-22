"""
Pydantic Schemas for Airline Operations RAG System
Enterprise-grade type definitions with validation
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime


# ==================== QUERY SCHEMAS ====================

class QueryType(str, Enum):
    SOP_LOOKUP = "sop_lookup"
    MAINTENANCE_REASONING = "maintenance_reasoning"
    DELAY_ANALYSIS = "delay_analysis"
    CREW_OPERATIONS = "crew_operations"
    GENERAL = "general"


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=1000)
    session_id: Optional[str] = None
    fleet_filter: Optional[List[str]] = None
    airport_filter: Optional[List[str]] = None
    include_figures: bool = True
    stream: bool = False
    
    @validator("query")
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class SubQuery(BaseModel):
    """Decomposed atomic sub-query for multi-step reasoning"""
    query_text: str
    query_type: QueryType
    priority: int = 1
    depends_on: Optional[List[str]] = None


# ==================== DOCUMENT SCHEMAS ====================

class DocumentMetadata(BaseModel):
    source_file: str
    page_number: Optional[int] = None
    fleet_type: Optional[str] = None
    airport_code: Optional[str] = None
    document_type: str = "sop"  # sop, maintenance, training, safety
    effective_date: Optional[datetime] = None
    revision: Optional[str] = None
    section: Optional[str] = None


class DocumentChunk(BaseModel):
    chunk_id: str
    content: str
    metadata: DocumentMetadata
    embedding: Optional[List[float]] = None
    has_figure: bool = False
    figure_description: Optional[str] = None


class RetrievedDocument(BaseModel):
    chunk_id: str
    content: str
    metadata: DocumentMetadata
    score: float
    retrieval_method: str  # "dense", "sparse", "hybrid"
    

# ==================== RESPONSE SCHEMAS ====================

class Citation(BaseModel):
    citation_id: str
    source_file: str
    page_number: Optional[int]
    excerpt: str
    relevance_score: float


class ConfidenceMetrics(BaseModel):
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    retrieval_confidence: float = Field(..., ge=0.0, le=1.0)
    grounding_score: float = Field(..., ge=0.0, le=1.0)
    citation_coverage: float = Field(..., ge=0.0, le=1.0)
    should_abstain: bool = False
    abstention_reason: Optional[str] = None


class LatencyMetrics(BaseModel):
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    within_budget: bool


class QueryResponse(BaseModel):
    answer: str
    query_type: QueryType
    citations: List[Citation]
    confidence: ConfidenceMetrics
    latency: LatencyMetrics
    session_id: str
    related_figures: Optional[List[str]] = None
    follow_up_suggestions: Optional[List[str]] = None


# ==================== INGEST SCHEMAS ====================

class IngestRequest(BaseModel):
    file_path: str
    document_type: str = "sop"
    fleet_type: Optional[str] = None
    airport_code: Optional[str] = None
    extract_figures: bool = True


class IngestResponse(BaseModel):
    success: bool
    document_id: str
    chunks_created: int
    figures_extracted: int
    processing_time_ms: float
    errors: Optional[List[str]] = None


# ==================== EVALUATION SCHEMAS ====================

class FactualityResult(BaseModel):
    is_factual: bool
    unsupported_claims: List[str]
    supported_claims: List[str]
    factuality_score: float


class EvaluationResult(BaseModel):
    query: str
    response: str
    factuality: FactualityResult
    coverage_score: float
    latency_compliance: bool

