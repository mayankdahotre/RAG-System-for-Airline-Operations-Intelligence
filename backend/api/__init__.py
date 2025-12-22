"""
API module for FastAPI endpoints
Ingest, Query, and Streaming APIs
"""
from backend.api.ingest import router as ingest_router
from backend.api.query import router as query_router
from backend.api.stream import router as stream_router

__all__ = [
    "ingest_router",
    "query_router",
    "stream_router"
]

