"""
Confidence Scoring and Abstention Logic
Determines when the system should abstain from answering
"""
from typing import List, Optional
from dataclasses import dataclass
import structlog

from backend.schemas import RetrievedDocument, ConfidenceMetrics
from backend.config import settings
from backend.grounding.citation_enforcer import CitationEnforcementResult

logger = structlog.get_logger()


@dataclass
class ConfidenceFactors:
    """Individual factors contributing to confidence"""
    retrieval_score: float  # Average retrieval similarity
    source_diversity: float  # Diversity of sources
    coverage_score: float   # Citation coverage
    query_specificity: float  # How specific the query is
    
    def to_dict(self):
        return {
            "retrieval_score": self.retrieval_score,
            "source_diversity": self.source_diversity,
            "coverage_score": self.coverage_score,
            "query_specificity": self.query_specificity
        }


class ConfidenceScorer:
    """
    Calculates confidence scores for RAG responses.
    Implements abstention logic to avoid hallucinations in safety-critical contexts.
    
    Confidence is based on:
    1. Retrieval quality (similarity scores)
    2. Source diversity (multiple corroborating sources)
    3. Citation coverage (% of claims grounded)
    4. Query specificity (well-defined queries score higher)
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.78,
        min_sources: int = 2,
        abstention_message: str = None
    ):
        self.confidence_threshold = confidence_threshold
        self.min_sources = min_sources
        self.abstention_message = (
            abstention_message or 
            settings.grounding.abstention_message
        )
    
    def calculate_confidence(
        self,
        retrieved_docs: List[RetrievedDocument],
        citation_result: CitationEnforcementResult,
        query: str
    ) -> ConfidenceMetrics:
        """
        Calculate comprehensive confidence metrics.
        
        Args:
            retrieved_docs: Documents retrieved for the query
            citation_result: Result from citation enforcement
            query: Original user query
            
        Returns:
            ConfidenceMetrics with scores and abstention decision
        """
        factors = self._calculate_factors(
            retrieved_docs, citation_result, query
        )
        
        # Weighted combination of factors
        weights = {
            "retrieval": 0.35,
            "diversity": 0.15,
            "coverage": 0.35,
            "specificity": 0.15
        }
        
        overall_confidence = (
            weights["retrieval"] * factors.retrieval_score +
            weights["diversity"] * factors.source_diversity +
            weights["coverage"] * factors.coverage_score +
            weights["specificity"] * factors.query_specificity
        )
        
        # Determine if we should abstain
        should_abstain = self._should_abstain(
            overall_confidence, retrieved_docs, citation_result
        )
        
        abstention_reason = None
        if should_abstain:
            abstention_reason = self._get_abstention_reason(
                overall_confidence, factors, retrieved_docs
            )
        
        logger.info(
            "confidence_calculated",
            overall=overall_confidence,
            factors=factors.to_dict(),
            should_abstain=should_abstain
        )
        
        return ConfidenceMetrics(
            overall_confidence=overall_confidence,
            retrieval_confidence=factors.retrieval_score,
            grounding_score=factors.coverage_score,
            citation_coverage=citation_result.coverage_score,
            should_abstain=should_abstain,
            abstention_reason=abstention_reason
        )
    
    def _calculate_factors(
        self,
        retrieved_docs: List[RetrievedDocument],
        citation_result: CitationEnforcementResult,
        query: str
    ) -> ConfidenceFactors:
        """Calculate individual confidence factors."""
        
        # Retrieval score (average similarity)
        if retrieved_docs:
            scores = [doc.score for doc in retrieved_docs]
            retrieval_score = sum(scores) / len(scores)
        else:
            retrieval_score = 0.0
        
        # Source diversity (unique source files)
        if retrieved_docs:
            unique_sources = len(set(doc.metadata.source_file for doc in retrieved_docs))
            source_diversity = min(1.0, unique_sources / self.min_sources)
        else:
            source_diversity = 0.0
        
        # Coverage from citation result
        coverage_score = citation_result.coverage_score
        
        # Query specificity (heuristic based on query characteristics)
        query_specificity = self._calculate_query_specificity(query)
        
        return ConfidenceFactors(
            retrieval_score=retrieval_score,
            source_diversity=source_diversity,
            coverage_score=coverage_score,
            query_specificity=query_specificity
        )
    
    def _calculate_query_specificity(self, query: str) -> float:
        """
        Estimate how specific/well-defined a query is.
        More specific queries = higher confidence potential.
        """
        import re
        
        score = 0.5  # Base score
        
        # Check for specific identifiers
        if re.search(r'\b[A-Z]{2}\d{3,4}\b', query):  # Flight number
            score += 0.15
        if re.search(r'\bB7[3-8]\d|A3[2-5]\d\b', query.upper()):  # Fleet type
            score += 0.15
        if re.search(r'\b[A-Z]{3}\b', query):  # Airport code
            score += 0.1
        
        # Penalize vague queries
        vague_terms = ["general", "overall", "typically", "usually", "any"]
        if any(term in query.lower() for term in vague_terms):
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _should_abstain(
        self,
        confidence: float,
        retrieved_docs: List[RetrievedDocument],
        citation_result: CitationEnforcementResult
    ) -> bool:
        """Determine if the system should abstain from answering."""
        
        # Abstain if confidence below threshold
        if confidence < self.confidence_threshold:
            return True
        
        # Abstain if no relevant documents
        if not retrieved_docs:
            return True
        
        # Abstain if all top documents have low scores
        top_scores = [doc.score for doc in retrieved_docs[:3]]
        if all(score < settings.retrieval.similarity_threshold for score in top_scores):
            return True
        
        # Abstain if too many ungrounded claims
        if len(citation_result.ungrounded_claims) > len(citation_result.grounded_claims):
            return True
        
        return False
    
    def _get_abstention_reason(
        self,
        confidence: float,
        factors: ConfidenceFactors,
        retrieved_docs: List[RetrievedDocument]
    ) -> str:
        """Generate human-readable abstention reason."""
        
        if not retrieved_docs:
            return "No relevant documents found in the knowledge base."
        
        if factors.retrieval_score < 0.5:
            return "Retrieved documents have low relevance to the query."
        
        if factors.coverage_score < 0.5:
            return "Unable to verify sufficient claims against source documents."
        
        if confidence < self.confidence_threshold:
            return self.abstention_message
        
        return "Insufficient confidence to provide a reliable answer."
    
    def should_answer(self, confidence_score: float) -> bool:
        """Simple check if confidence is sufficient to answer."""
        return confidence_score >= self.confidence_threshold


# Singleton instance
confidence_scorer = ConfidenceScorer()

