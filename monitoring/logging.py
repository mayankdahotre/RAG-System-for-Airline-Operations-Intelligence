"""
Structured Logging for Production RAG
Comprehensive logging for debugging and monitoring
"""
import sys
import structlog
from typing import Optional, Dict, Any
from datetime import datetime
from functools import wraps
import json

from backend.config import settings


def configure_logging(
    level: str = "INFO",
    json_output: bool = True,
    include_timestamp: bool = True
):
    """
    Configure structured logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        json_output: Output logs as JSON (for production)
        include_timestamp: Include ISO timestamp in logs
    """
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    
    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure stdlib logging
    import logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper())
    )


class RAGLogger:
    """
    Specialized logger for RAG pipeline events.
    Provides structured logging with airline-ops context.
    """
    
    def __init__(self, component: str):
        self.component = component
        self.logger = structlog.get_logger(component)
    
    def query_start(
        self,
        query: str,
        session_id: str,
        query_type: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """Log query start."""
        self.logger.info(
            "query_start",
            query=query[:100],
            session_id=session_id,
            query_type=query_type,
            metadata=metadata
        )
    
    def query_complete(
        self,
        session_id: str,
        latency_ms: float,
        doc_count: int,
        confidence: float,
        abstained: bool = False
    ):
        """Log query completion."""
        self.logger.info(
            "query_complete",
            session_id=session_id,
            latency_ms=round(latency_ms, 2),
            documents_retrieved=doc_count,
            confidence=round(confidence, 3),
            abstained=abstained
        )
    
    def retrieval_event(
        self,
        method: str,
        query: str,
        doc_count: int,
        latency_ms: float,
        avg_score: float
    ):
        """Log retrieval event."""
        self.logger.info(
            "retrieval",
            method=method,
            query=query[:50],
            documents=doc_count,
            latency_ms=round(latency_ms, 2),
            avg_score=round(avg_score, 3)
        )
    
    def generation_event(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float
    ):
        """Log LLM generation event."""
        self.logger.info(
            "generation",
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=round(latency_ms, 2)
        )
    
    def grounding_event(
        self,
        citations: int,
        coverage: float,
        confidence: float,
        abstained: bool
    ):
        """Log grounding/citation event."""
        self.logger.info(
            "grounding",
            citations=citations,
            coverage_score=round(coverage, 3),
            confidence=round(confidence, 3),
            abstained=abstained
        )
    
    def error(
        self,
        error_type: str,
        message: str,
        session_id: Optional[str] = None,
        stack_trace: Optional[str] = None
    ):
        """Log error event."""
        self.logger.error(
            "error",
            error_type=error_type,
            message=message,
            session_id=session_id,
            stack_trace=stack_trace
        )
    
    def ingestion_event(
        self,
        document_id: str,
        file_name: str,
        chunks: int,
        latency_ms: float,
        success: bool
    ):
        """Log document ingestion event."""
        self.logger.info(
            "ingestion",
            document_id=document_id,
            file_name=file_name,
            chunks_created=chunks,
            latency_ms=round(latency_ms, 2),
            success=success
        )


def log_function_call(logger_name: str = "function"):
    """Decorator for logging function calls."""
    def decorator(func):
        logger = structlog.get_logger(logger_name)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = datetime.now()
            logger.debug(
                "function_call_start",
                function=func.__name__,
                args_count=len(args),
                kwargs_keys=list(kwargs.keys())
            )
            try:
                result = func(*args, **kwargs)
                latency = (datetime.now() - start).total_seconds() * 1000
                logger.debug(
                    "function_call_complete",
                    function=func.__name__,
                    latency_ms=round(latency, 2)
                )
                return result
            except Exception as e:
                logger.error(
                    "function_call_error",
                    function=func.__name__,
                    error=str(e)
                )
                raise
        return wrapper
    return decorator


# Initialize logging on import
configure_logging(
    level="INFO" if settings.environment.value == "production" else "DEBUG",
    json_output=settings.environment.value == "production"
)

