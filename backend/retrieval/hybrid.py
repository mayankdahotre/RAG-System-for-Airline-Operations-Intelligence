"""
Hybrid Retrieval Engine
Combines Dense (semantic) + Sparse (BM25) + Metadata filtering
Production-grade retrieval for airline operations
"""
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import structlog

from backend.config import settings
from backend.schemas import RetrievedDocument, DocumentChunk, QueryType
from backend.retrieval.dense import DenseRetriever
from backend.retrieval.sparse import SparseRetriever

logger = structlog.get_logger()


@dataclass
class RetrievalResult:
    """Complete retrieval result with metrics"""
    documents: List[RetrievedDocument]
    dense_count: int
    sparse_count: int
    total_candidates: int
    avg_score: float
    retrieval_time_ms: float


class HybridRetriever:
    """
    Production hybrid retrieval combining:
    1. Dense retrieval (semantic similarity)
    2. Sparse retrieval (BM25 keyword matching)
    3. Metadata-based filtering and boosting
    4. Reciprocal Rank Fusion for score combination
    
    This is exactly how enterprise search systems work at scale.
    """
    
    def __init__(
        self,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        rrf_k: int = 60  # RRF constant
    ):
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k
        
        self.dense_retriever = DenseRetriever()
        self.sparse_retriever = SparseRetriever()
    
    def add_documents(self, documents: List[DocumentChunk]) -> None:
        """Add documents to both indices."""
        self.dense_retriever.add_documents(documents)
        self.sparse_retriever.add_documents(documents)
        logger.info("hybrid_documents_indexed", count=len(documents))
    
    def search(
        self,
        query: str,
        k: int = 10,
        query_type: Optional[QueryType] = None,
        metadata_filter: Optional[dict] = None,
        dense_weight: Optional[float] = None,
        sparse_weight: Optional[float] = None
    ) -> RetrievalResult:
        """
        Hybrid search with configurable weights.
        
        Args:
            query: Search query
            k: Number of final results
            query_type: Query type for weight adjustment
            metadata_filter: Filter by metadata
            dense_weight: Override default dense weight
            sparse_weight: Override default sparse weight
            
        Returns:
            RetrievalResult with documents and metrics
        """
        import time
        start_time = time.time()
        
        # Adjust weights based on query type
        d_weight = dense_weight or self._get_weight_for_query(query_type, "dense")
        s_weight = sparse_weight or self._get_weight_for_query(query_type, "sparse")
        
        # Fetch from both retrievers (fetch more candidates for fusion)
        candidate_k = k * 3
        
        dense_results = self.dense_retriever.search(
            query, k=candidate_k, metadata_filter=metadata_filter
        )
        sparse_results = self.sparse_retriever.search(
            query, k=candidate_k, metadata_filter=metadata_filter
        )
        
        # Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(
            dense_results, sparse_results, d_weight, s_weight
        )
        
        # Apply metadata boosting
        if metadata_filter:
            fused_results = self._apply_metadata_boost(fused_results, metadata_filter)
        
        # Take top-k
        final_results = sorted(fused_results, key=lambda x: x.score, reverse=True)[:k]
        
        retrieval_time = (time.time() - start_time) * 1000
        
        avg_score = sum(r.score for r in final_results) / len(final_results) if final_results else 0
        
        logger.info(
            "hybrid_search",
            query=query[:30],
            dense_count=len(dense_results),
            sparse_count=len(sparse_results),
            final_count=len(final_results),
            time_ms=retrieval_time
        )
        
        return RetrievalResult(
            documents=final_results,
            dense_count=len(dense_results),
            sparse_count=len(sparse_results),
            total_candidates=len(dense_results) + len(sparse_results),
            avg_score=avg_score,
            retrieval_time_ms=retrieval_time
        )
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[RetrievedDocument, float]],
        sparse_results: List[Tuple[RetrievedDocument, float]],
        dense_weight: float,
        sparse_weight: float
    ) -> List[RetrievedDocument]:
        """
        Combine results using Reciprocal Rank Fusion.
        RRF is robust and doesn't require score normalization.
        """
        scores: Dict[str, float] = {}
        docs: Dict[str, RetrievedDocument] = {}
        
        # Score dense results
        for rank, (doc, _) in enumerate(dense_results, 1):
            rrf_score = dense_weight / (self.rrf_k + rank)
            scores[doc.chunk_id] = scores.get(doc.chunk_id, 0) + rrf_score
            docs[doc.chunk_id] = doc
        
        # Score sparse results
        for rank, (doc, _) in enumerate(sparse_results, 1):
            rrf_score = sparse_weight / (self.rrf_k + rank)
            scores[doc.chunk_id] = scores.get(doc.chunk_id, 0) + rrf_score
            if doc.chunk_id not in docs:
                docs[doc.chunk_id] = doc
        
        # Create final documents with fused scores
        results = []
        for chunk_id, score in scores.items():
            doc = docs[chunk_id]
            results.append(RetrievedDocument(
                chunk_id=doc.chunk_id,
                content=doc.content,
                metadata=doc.metadata,
                score=score,
                retrieval_method="hybrid"
            ))
        
        return results
    
    def _get_weight_for_query(self, query_type: Optional[QueryType], retriever: str) -> float:
        """Get retrieval weight based on query type."""
        if not query_type:
            return self.dense_weight if retriever == "dense" else self.sparse_weight
        
        # Query-type specific weights
        weights = {
            QueryType.SOP_LOOKUP: {"dense": 0.4, "sparse": 0.6},
            QueryType.MAINTENANCE_REASONING: {"dense": 0.5, "sparse": 0.5},
            QueryType.DELAY_ANALYSIS: {"dense": 0.7, "sparse": 0.3},
            QueryType.CREW_OPERATIONS: {"dense": 0.6, "sparse": 0.4},
            QueryType.GENERAL: {"dense": 0.6, "sparse": 0.4}
        }
        
        return weights.get(query_type, {}).get(retriever, 0.5)
    
    def _apply_metadata_boost(
        self,
        documents: List[RetrievedDocument],
        metadata_filter: dict
    ) -> List[RetrievedDocument]:
        """Boost documents that match metadata filters."""
        boost = settings.retrieval.metadata_boost
        
        boosted = []
        for doc in documents:
            match_count = 0
            for key, value in metadata_filter.items():
                if hasattr(doc.metadata, key) and getattr(doc.metadata, key) == value:
                    match_count += 1
            
            if match_count > 0:
                boosted_score = doc.score * (boost ** match_count)
                doc = RetrievedDocument(
                    chunk_id=doc.chunk_id,
                    content=doc.content,
                    metadata=doc.metadata,
                    score=boosted_score,
                    retrieval_method=doc.retrieval_method
                )
            boosted.append(doc)
        
        return boosted

