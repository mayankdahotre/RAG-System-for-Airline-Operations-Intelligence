"""
Coverage Evaluation for RAG Responses
Measures how well responses cover the retrieved context
"""
from typing import List, Dict, Set
from dataclasses import dataclass
import re
import structlog

from backend.schemas import RetrievedDocument

logger = structlog.get_logger()


@dataclass
class CoverageMetrics:
    """Detailed coverage metrics"""
    context_coverage: float    # How much context was used
    answer_density: float      # Information density of answer
    redundancy_score: float    # How much content is repeated
    key_term_coverage: float   # Coverage of important terms
    completeness_score: float  # Overall completeness
    missed_topics: List[str]   # Topics in context not in answer


class CoverageEvaluator:
    """
    Evaluates how well a RAG response covers the retrieved context.
    
    Important for ensuring:
    1. Key information isn't missed
    2. Response is comprehensive but not redundant
    3. All relevant aspects are addressed
    """
    
    # Important terms that should be preserved in airline context
    CRITICAL_TERMS = {
        "safety": ["warning", "caution", "danger", "emergency", "abort"],
        "procedures": ["step", "procedure", "checklist", "verify", "confirm"],
        "limits": ["maximum", "minimum", "limit", "threshold", "tolerance"],
        "requirements": ["must", "shall", "required", "mandatory", "prohibited"]
    }
    
    def __init__(self):
        pass
    
    def evaluate(
        self,
        answer: str,
        retrieved_docs: List[RetrievedDocument],
        query: str
    ) -> CoverageMetrics:
        """
        Evaluate coverage of response against retrieved context.
        
        Args:
            answer: Generated response
            retrieved_docs: Retrieved source documents
            query: Original query
            
        Returns:
            CoverageMetrics with detailed scores
        """
        # Combine context
        context = " ".join([doc.content for doc in retrieved_docs])
        
        # Calculate individual metrics
        context_coverage = self._calculate_context_coverage(answer, context)
        answer_density = self._calculate_answer_density(answer)
        redundancy = self._calculate_redundancy(answer)
        key_term_coverage = self._calculate_key_term_coverage(answer, context)
        missed_topics = self._find_missed_topics(answer, context)
        
        # Calculate completeness
        completeness = (
            0.3 * context_coverage +
            0.2 * answer_density +
            0.2 * (1 - redundancy) +
            0.3 * key_term_coverage
        )
        
        logger.info(
            "coverage_evaluated",
            context_coverage=context_coverage,
            key_terms=key_term_coverage,
            completeness=completeness
        )
        
        return CoverageMetrics(
            context_coverage=context_coverage,
            answer_density=answer_density,
            redundancy_score=redundancy,
            key_term_coverage=key_term_coverage,
            completeness_score=completeness,
            missed_topics=missed_topics[:5]  # Top 5 missed
        )
    
    def _calculate_context_coverage(self, answer: str, context: str) -> float:
        """
        Calculate how much of the context is reflected in the answer.
        Uses n-gram overlap.
        """
        answer_lower = answer.lower()
        context_lower = context.lower()
        
        # Extract significant phrases (3-grams)
        context_ngrams = self._extract_ngrams(context_lower, 3)
        answer_ngrams = self._extract_ngrams(answer_lower, 3)
        
        if not context_ngrams:
            return 0.0
        
        overlap = len(context_ngrams & answer_ngrams)
        coverage = overlap / len(context_ngrams)
        
        # Normalize to reasonable range
        return min(1.0, coverage * 5)
    
    def _extract_ngrams(self, text: str, n: int) -> Set[str]:
        """Extract n-grams from text."""
        words = re.findall(r'\b\w{3,}\b', text)
        ngrams = set()
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i:i+n])
            ngrams.add(ngram)
        return ngrams
    
    def _calculate_answer_density(self, answer: str) -> float:
        """
        Calculate information density of the answer.
        Higher = more informative per word.
        """
        words = answer.split()
        if not words:
            return 0.0
        
        # Count informative elements
        informative = 0
        
        # Numbers
        informative += len(re.findall(r'\b\d+(?:\.\d+)?', answer))
        
        # Technical terms (uppercase, codes, etc.)
        informative += len(re.findall(r'\b[A-Z]{2,}\b', answer))
        
        # Action verbs
        action_verbs = ["verify", "check", "ensure", "confirm", "perform", "execute"]
        informative += sum(1 for v in action_verbs if v in answer.lower())
        
        density = informative / len(words)
        return min(1.0, density * 10)
    
    def _calculate_redundancy(self, answer: str) -> float:
        """Calculate how much content is repeated."""
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        
        if len(sentences) < 2:
            return 0.0
        
        # Check for repeated content
        seen_content = set()
        redundant = 0
        
        for sentence in sentences:
            # Normalize
            normalized = " ".join(sorted(sentence.lower().split()))
            
            # Check similarity to previous sentences
            for seen in seen_content:
                words_sent = set(normalized.split())
                words_seen = set(seen.split())
                overlap = len(words_sent & words_seen)
                if words_sent and overlap / len(words_sent) > 0.7:
                    redundant += 1
                    break
            
            seen_content.add(normalized)
        
        return redundant / len(sentences)
    
    def _calculate_key_term_coverage(self, answer: str, context: str) -> float:
        """Calculate coverage of critical airline terms."""
        answer_lower = answer.lower()
        context_lower = context.lower()
        
        context_key_terms = set()
        answer_key_terms = set()
        
        for category, terms in self.CRITICAL_TERMS.items():
            for term in terms:
                if term in context_lower:
                    context_key_terms.add(term)
                if term in answer_lower:
                    answer_key_terms.add(term)
        
        if not context_key_terms:
            return 1.0  # No key terms to cover
        
        return len(context_key_terms & answer_key_terms) / len(context_key_terms)
    
    def _find_missed_topics(self, answer: str, context: str) -> List[str]:
        """Identify topics in context that aren't in the answer."""
        answer_lower = answer.lower()
        context_lower = context.lower()
        
        # Extract significant nouns/phrases from context
        context_phrases = re.findall(r'\b(?:[A-Z][a-z]+\s+)+[A-Z][a-z]+\b', context)
        context_phrases += re.findall(r'\b[A-Z]{2,}\s+\d+', context)  # Codes
        
        missed = []
        for phrase in set(context_phrases):
            if phrase.lower() not in answer_lower:
                missed.append(phrase)
        
        return missed


# Singleton instance
coverage_evaluator = CoverageEvaluator()

