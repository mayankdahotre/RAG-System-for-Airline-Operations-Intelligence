"""
Multimodal RAG System for Airline Operations Intelligence
Enterprise-grade FastAPI Application

Features:
- Hybrid Retrieval (Dense + Sparse + Metadata)
- Multi-step Query Decomposition (Agentic RAG)
- Citation Enforcement & Confidence Scoring
- Streaming Responses with SSE
- Production Monitoring & Logging
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import structlog
import time

from backend.config import settings
from backend.api import ingest_router, query_router, stream_router
from monitoring.latency import latency_monitor
from monitoring.logging import configure_logging, RAGLogger

# Configure logging
configure_logging(
    level="INFO" if settings.environment.value == "production" else "DEBUG",
    json_output=settings.environment.value == "production"
)

logger = structlog.get_logger()
rag_logger = RAGLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info(
        "application_starting",
        environment=settings.environment.value,
        version="2.0.0"
    )
    
    # Validate configuration
    try:
        settings.validate()
        logger.info("configuration_validated")
    except AssertionError as e:
        logger.error("configuration_invalid", error=str(e))
    
    yield
    
    # Shutdown
    logger.info("application_shutting_down")
    
    # Log final latency stats
    stats = latency_monitor.get_all_stats()
    logger.info("final_latency_stats", stats=stats)


# Create FastAPI app
app = FastAPI(
    title="Airline Operations RAG API",
    description="""
    Enterprise Multimodal RAG System for Airline Operations Intelligence.
    
    ## Features
    - **Hybrid Retrieval**: Dense embeddings + BM25 sparse search
    - **Agentic RAG**: Multi-step query decomposition
    - **Grounding**: Citation enforcement and confidence scoring
    - **Streaming**: Real-time response streaming with SSE
    - **Multimodal**: PDF parsing with figure and table extraction
    
    ## Use Cases
    - SOP Lookup
    - Maintenance Reasoning  
    - Delay Root Cause Analysis
    - Crew Operations Queries
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def latency_middleware(request: Request, call_next):
    """Track request latency."""
    start_time = time.perf_counter()
    
    response = await call_next(request)
    
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    # Log request
    logger.info(
        "http_request",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        latency_ms=round(latency_ms, 2)
    )
    
    # Add latency header
    response.headers["X-Response-Time-Ms"] = str(round(latency_ms, 2))
    
    return response


# Include routers
app.include_router(ingest_router)
app.include_router(query_router)
app.include_router(stream_router)


@app.get("/")
async def root():
    """Root endpoint with system info."""
    return {
        "service": "Airline Operations RAG",
        "version": "2.0.0",
        "status": "operational",
        "environment": settings.environment.value,
        "endpoints": {
            "docs": "/docs",
            "query": "/query/",
            "stream": "/stream/query",
            "ingest": "/ingest/document"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "checks": {
            "api": True,
            "vectorstore": True,  # Add actual check in production
            "llm": settings.openai_api_key is not None
        }
    }


@app.get("/metrics")
async def get_metrics():
    """Get system metrics."""
    return {
        "latency": latency_monitor.get_all_stats(),
        "sla_compliance": latency_monitor.check_sla()
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(
        "unhandled_exception",
        path=request.url.path,
        error=str(exc),
        error_type=type(exc).__name__
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if settings.environment.value != "production" else "An error occurred"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.environment.value == "development"
    )

