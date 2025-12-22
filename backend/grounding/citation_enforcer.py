"""
Citation Enforcement for Grounded Responses
Ensures LLM outputs are traceable to source documents
"""
from typing import List, Optional, Tuple
from dataclasses import dataclass
import re
import structlog

from backend.schemas import Citation, RetrievedDocument
from backend.config import settings

logger = structlog.get_logger()


@dataclass 
class CitationEnforcementResult:
    """Result of citation enforcement"""
    citations: List[Citation]
    grounded_claims: List[str]
    ungrounded_claims: List[str]
    coverage_score: float
    is_sufficiently_grounded: bool


class CitationEnforcer:
    """
    Enforces citation requirements for LLM responses.
    Critical for safety-critical airline operations where every claim must be traceable.
    
    Key responsibilities:
    1. Extract claims from LLM response
    2. Match claims to source documents
    3. Generate citations with excerpts
    4. Calculate grounding coverage
    5. Flag ungrounded claims
    """
    
    def __init__(
        self,
        min_citations: int = 1,
        require_exact_match: bool = False,
        similarity_threshold: float = 0.7
    ):
        self.min_citations = min_citations
        self.require_exact_match = require_exact_match
        self.similarity_threshold = similarity_threshold
    
    def enforce(
        self,
        answer: str,
        sources: List[RetrievedDocument]
    ) -> CitationEnforcementResult:
        """
        Enforce citation requirements on an answer.
        
        Args:
            answer: LLM-generated answer
            sources: Retrieved source documents
            
        Returns:
            CitationEnforcementResult with citations and grounding analysis
        """
        # Extract claims from answer
        claims = self._extract_claims(answer)
        
        if not claims:
            logger.warning("no_claims_extracted", answer=answer[:50])
            return CitationEnforcementResult(
                citations=[],
                grounded_claims=[],
                ungrounded_claims=[],
                coverage_score=0.0,
                is_sufficiently_grounded=False
            )
        
        # Match claims to sources
        citations = []
        grounded_claims = []
        ungrounded_claims = []
        
        for claim in claims:
            matches = self._find_supporting_sources(claim, sources)
            
            if matches:
                grounded_claims.append(claim)
                for source, excerpt, score in matches:
                    # Avoid duplicate citations
                    if not any(c.chunk_id == source.chunk_id for c in citations):
                        citations.append(Citation(
                            citation_id=f"[{len(citations) + 1}]",
                            source_file=source.metadata.source_file,
                            page_number=source.metadata.page_number,
                            excerpt=excerpt[:200],
                            relevance_score=score
                        ))
            else:
                ungrounded_claims.append(claim)
        
        # Calculate coverage
        total_claims = len(claims)
        grounded_count = len(grounded_claims)
        coverage_score = grounded_count / total_claims if total_claims > 0 else 0.0
        
        is_grounded = (
            len(citations) >= self.min_citations and
            coverage_score >= settings.grounding.confidence_threshold
        )
        
        logger.info(
            "citation_enforcement",
            total_claims=total_claims,
            grounded=grounded_count,
            citations=len(citations),
            coverage=coverage_score
        )
        
        return CitationEnforcementResult(
            citations=citations,
            grounded_claims=grounded_claims,
            ungrounded_claims=ungrounded_claims,
            coverage_score=coverage_score,
            is_sufficiently_grounded=is_grounded
        )
    
    def _extract_claims(self, answer: str) -> List[str]:
        """
        Extract individual claims from an answer.
        Uses sentence splitting with special handling for procedural content.
        """
        # Split by sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            # Filter out short sentences and meta-statements
            if len(sentence) < 20:
                continue
            if any(phrase in sentence.lower() for phrase in [
                "i think", "i believe", "maybe", "perhaps",
                "here is", "here are", "the answer is"
            ]):
                continue
            claims.append(sentence)
        
        return claims
    
    def _find_supporting_sources(
        self,
        claim: str,
        sources: List[RetrievedDocument]
    ) -> List[Tuple[RetrievedDocument, str, float]]:
        """
        Find sources that support a claim.
        Returns list of (source, matching_excerpt, similarity_score).
        """
        matches = []
        claim_lower = claim.lower()
        
        # Extract key terms from claim
        claim_terms = set(re.findall(r'\b\w{4,}\b', claim_lower))
        
        for source in sources:
            content_lower = source.content.lower()
            
            # Check for exact substring match
            if self.require_exact_match:
                if claim_lower in content_lower:
                    matches.append((source, claim, 1.0))
                    continue
            
            # Calculate term overlap
            source_terms = set(re.findall(r'\b\w{4,}\b', content_lower))
            overlap = len(claim_terms & source_terms)
            similarity = overlap / len(claim_terms) if claim_terms else 0
            
            if similarity >= self.similarity_threshold:
                # Find best matching excerpt
                excerpt = self._find_best_excerpt(claim, source.content)
                matches.append((source, excerpt, similarity))
        
        # Sort by similarity score
        matches.sort(key=lambda x: x[2], reverse=True)
        return matches[:2]  # Return top 2 matches
    
    def _find_best_excerpt(self, claim: str, content: str, window: int = 200) -> str:
        """Find the most relevant excerpt from content for a claim."""
        claim_terms = set(re.findall(r'\b\w{4,}\b', claim.lower()))
        
        best_excerpt = ""
        best_score = 0
        
        # Sliding window over content
        words = content.split()
        for i in range(0, len(words), 10):
            window_text = " ".join(words[i:i+30])
            window_terms = set(re.findall(r'\b\w{4,}\b', window_text.lower()))
            score = len(claim_terms & window_terms)
            if score > best_score:
                best_score = score
                best_excerpt = window_text
        
        return best_excerpt if best_excerpt else content[:window]
    
    def format_with_citations(
        self,
        answer: str,
        citations: List[Citation]
    ) -> str:
        """Format answer with inline citations."""
        if not citations:
            return answer
        
        # Add citation references at end
        citation_section = "\n\n**Sources:**\n"
        for c in citations:
            citation_section += f"\n{c.citation_id} {c.source_file}"
            if c.page_number:
                citation_section += f", p.{c.page_number}"
        
        return answer + citation_section


# Singleton instance
citation_enforcer = CitationEnforcer()

