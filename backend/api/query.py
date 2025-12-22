"""
Query API for RAG Operations
Main endpoint for question answering
"""
from fastapi import APIRouter, HTTPException
from typing import Optional
import time
import uuid
import structlog
from openai import OpenAI

from backend.schemas import (
    QueryRequest, QueryResponse, QueryType,
    LatencyMetrics, Citation, ConfidenceMetrics
)
from backend.agents.query_classifier import classifier
from backend.agents.decomposer import decomposer
from backend.retrieval.hybrid import HybridRetriever
from backend.grounding.citation_enforcer import citation_enforcer
from backend.grounding.confidence_scorer import confidence_scorer
from backend.evaluation.factuality import factuality_evaluator
from backend.config import settings

logger = structlog.get_logger()
router = APIRouter(prefix="/query", tags=["query"])

# Initialize components
retriever = HybridRetriever()
_openai_client = None

def get_openai_client():
    """Lazy initialization of OpenAI client."""
    global _openai_client
    if _openai_client is None:
        api_key = settings.openai_api_key
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


SYSTEM_PROMPT = """You are an expert airline operations assistant. Your role is to provide accurate, 
grounded answers based ONLY on the provided context. 

Rules:
1. ONLY use information from the provided context
2. If the context doesn't contain the answer, say so explicitly
3. For procedures, list steps in order
4. For safety-critical information, be especially precise
5. Cite the source when providing specific information
6. Never make up information not in the context

Context:
{context}

Question: {question}

Provide a clear, accurate answer based only on the above context."""


@router.post("/", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Process a query against the airline operations knowledge base.
    
    Pipeline:
    1. Classify query type
    2. Decompose if complex
    3. Hybrid retrieval
    4. LLM generation
    5. Citation enforcement
    6. Confidence scoring
    7. Optional abstention
    """
    total_start = time.time()
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        # 1. Classify query
        query_type, classification_confidence = classifier.classify(request)
        entities = classifier.extract_entities(request.query)
        
        # 2. Decompose if needed
        decomposition = decomposer.decompose(
            request.query, query_type, entities
        )
        
        # 3. Retrieve for all sub-queries
        retrieval_start = time.time()
        
        metadata_filter = {}
        if request.fleet_filter:
            metadata_filter["fleet_type"] = request.fleet_filter[0]
        if request.airport_filter:
            metadata_filter["airport_code"] = request.airport_filter[0]
        
        all_docs = []
        for sub_query in decomposition.sub_queries:
            result = retriever.search(
                sub_query.query_text,
                k=settings.retrieval.top_k,
                query_type=sub_query.query_type,
                metadata_filter=metadata_filter if metadata_filter else None
            )
            all_docs.extend(result.documents)
        
        # Deduplicate by chunk_id
        seen = set()
        unique_docs = []
        for doc in all_docs:
            if doc.chunk_id not in seen:
                seen.add(doc.chunk_id)
                unique_docs.append(doc)
        
        # Sort by score and take top-k
        unique_docs.sort(key=lambda x: x.score, reverse=True)
        final_docs = unique_docs[:settings.retrieval.rerank_top_k]
        
        retrieval_time = (time.time() - retrieval_start) * 1000
        
        # 4. Generate answer
        generation_start = time.time()
        
        context = "\n\n".join([
            f"[Source: {doc.metadata.source_file}, p.{doc.metadata.page_number}]\n{doc.content}"
            for doc in final_docs
        ])
        
        response = get_openai_client().chat.completions.create(
            model=settings.llm.model_name,
            messages=[{
                "role": "user",
                "content": SYSTEM_PROMPT.format(
                    context=context,
                    question=request.query
                )
            }],
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens
        )
        
        answer = response.choices[0].message.content
        generation_time = (time.time() - generation_start) * 1000
        
        # 5. Citation enforcement
        citation_result = citation_enforcer.enforce(answer, final_docs)
        
        # 6. Confidence scoring
        confidence_metrics = confidence_scorer.calculate_confidence(
            final_docs, citation_result, request.query
        )
        
        # 7. Handle abstention
        if confidence_metrics.should_abstain:
            answer = settings.grounding.abstention_message
            logger.info("query_abstained", query=request.query[:50])
        
        # Calculate total latency
        total_time = (time.time() - total_start) * 1000
        within_budget = total_time <= settings.latency.total_ms
        
        logger.info(
            "query_completed",
            query=request.query[:50],
            type=query_type.value,
            docs=len(final_docs),
            confidence=confidence_metrics.overall_confidence,
            time_ms=total_time
        )
        
        return QueryResponse(
            answer=answer,
            query_type=query_type,
            citations=citation_result.citations,
            confidence=confidence_metrics,
            latency=LatencyMetrics(
                retrieval_ms=retrieval_time,
                generation_ms=generation_time,
                total_ms=total_time,
                within_budget=within_budget
            ),
            session_id=session_id,
            follow_up_suggestions=_generate_follow_ups(query_type, entities)
        )
        
    except Exception as e:
        logger.error("query_failed", query=request.query[:50], error=str(e))
        raise HTTPException(500, f"Query failed: {str(e)}")


def _generate_follow_ups(query_type: QueryType, entities: dict) -> list:
    """Generate relevant follow-up suggestions."""
    suggestions = {
        QueryType.SOP_LOOKUP: [
            "What are the emergency procedures?",
            "Are there any exceptions to this procedure?"
        ],
        QueryType.MAINTENANCE_REASONING: [
            "What are the MEL requirements?",
            "What is the turnaround time?"
        ],
        QueryType.DELAY_ANALYSIS: [
            "What are the mitigation procedures?",
            "How does this affect crew duty time?"
        ]
    }
    return suggestions.get(query_type, [])[:2]

