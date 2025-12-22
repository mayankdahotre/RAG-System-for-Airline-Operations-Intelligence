"""
Streaming API for Real-time Responses
Server-Sent Events for progressive answer generation
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
from typing import AsyncGenerator
import json
import time
import uuid
import structlog
from openai import OpenAI

from backend.schemas import QueryRequest, QueryType
from backend.agents.query_classifier import classifier
from backend.retrieval.hybrid import HybridRetriever
from backend.grounding.citation_enforcer import citation_enforcer
from backend.config import settings

logger = structlog.get_logger()
router = APIRouter(prefix="/stream", tags=["streaming"])

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


STREAMING_SYSTEM_PROMPT = """You are an expert airline operations assistant. 
Answer based ONLY on the provided context. Be precise and cite sources.

Context:
{context}

Question: {question}"""


@router.post("/query")
async def stream_query(request: QueryRequest):
    """
    Stream query response with Server-Sent Events.
    
    Events:
    - status: Pipeline stage updates
    - retrieval: Retrieved documents
    - token: Individual response tokens
    - citation: Citation information
    - complete: Final response metadata
    """
    
    async def generate_events() -> AsyncGenerator[str, None]:
        session_id = request.session_id or str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Status: Starting
            yield _format_sse("status", {"stage": "classifying", "message": "Analyzing query..."})
            
            # Classify query
            query_type, confidence = classifier.classify(request)
            yield _format_sse("status", {
                "stage": "classified",
                "query_type": query_type.value,
                "confidence": confidence
            })
            
            # Retrieve
            yield _format_sse("status", {"stage": "retrieving", "message": "Searching knowledge base..."})
            
            result = retriever.search(
                request.query,
                k=settings.retrieval.top_k,
                query_type=query_type
            )
            
            # Send retrieval summary
            yield _format_sse("retrieval", {
                "document_count": len(result.documents),
                "avg_score": result.avg_score,
                "time_ms": result.retrieval_time_ms
            })
            
            # Prepare context
            context = "\n\n".join([
                f"[{doc.metadata.source_file}]\n{doc.content[:500]}"
                for doc in result.documents[:5]
            ])
            
            # Stream LLM response
            yield _format_sse("status", {"stage": "generating", "message": "Generating response..."})
            
            full_response = ""
            stream = get_openai_client().chat.completions.create(
                model=settings.llm.model_name,
                messages=[{
                    "role": "user",
                    "content": STREAMING_SYSTEM_PROMPT.format(
                        context=context,
                        question=request.query
                    )
                }],
                temperature=settings.llm.temperature,
                max_tokens=settings.llm.max_tokens,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_response += token
                    yield _format_sse("token", {"content": token})
            
            # Citation enforcement
            citation_result = citation_enforcer.enforce(full_response, result.documents)
            
            # Send citations
            for citation in citation_result.citations:
                yield _format_sse("citation", {
                    "id": citation.citation_id,
                    "source": citation.source_file,
                    "page": citation.page_number,
                    "relevance": citation.relevance_score
                })
            
            # Complete
            total_time = (time.time() - start_time) * 1000
            
            yield _format_sse("complete", {
                "session_id": session_id,
                "query_type": query_type.value,
                "total_time_ms": total_time,
                "grounded": citation_result.is_sufficiently_grounded,
                "coverage_score": citation_result.coverage_score
            })
            
            logger.info(
                "stream_completed",
                session_id=session_id,
                time_ms=total_time
            )
            
        except Exception as e:
            logger.error("stream_failed", error=str(e))
            yield _format_sse("error", {"message": str(e)})
    
    return EventSourceResponse(generate_events())


def _format_sse(event: str, data: dict) -> str:
    """Format data as SSE event."""
    return json.dumps({"event": event, "data": data})


@router.get("/health")
async def streaming_health():
    """Health check for streaming endpoint."""
    return {"status": "healthy", "streaming": True}


class StreamBuffer:
    """
    Buffer for managing streaming output.
    Ensures tokens are sent at appropriate intervals.
    """
    
    def __init__(self, min_interval_ms: int = 50):
        self.min_interval = min_interval_ms / 1000
        self.buffer = ""
        self.last_send = 0
    
    def add(self, token: str) -> str | None:
        """Add token and return content to send if ready."""
        self.buffer += token
        
        now = time.time()
        if now - self.last_send >= self.min_interval or len(self.buffer) > 10:
            content = self.buffer
            self.buffer = ""
            self.last_send = now
            return content
        
        return None
    
    def flush(self) -> str:
        """Flush remaining buffer."""
        content = self.buffer
        self.buffer = ""
        return content

