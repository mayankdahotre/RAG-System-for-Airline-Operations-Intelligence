"""
Sparse Retrieval using BM25
Exact keyword matching for airline terminology and codes
"""
from typing import List, Optional, Tuple, Dict
import re
import structlog
from rank_bm25 import BM25Okapi

from backend.schemas import RetrievedDocument, DocumentChunk, DocumentMetadata

logger = structlog.get_logger()


class SparseRetriever:
    """
    BM25-based sparse retrieval for exact terminology matching.
    Critical for airline operations where specific codes and terms matter.
    
    Examples where sparse excels:
    - "MEL 21-31" (exact code lookup)
    - "B737-MAX MCAS procedure" (specific aircraft/system)
    - "NOTAM KORD 12/2024" (airport notice reference)
    """
    
    # Airline-specific tokenization patterns
    AIRLINE_PATTERNS = [
        r"\b[A-Z]{2}\d{3,4}\b",          # Flight numbers: UA1234
        r"\b[A-Z]{3,4}\b",                # Airport codes: ORD, KORD
        r"\bB7[3-8]\d(?:-\w+)?\b",        # Boeing models: B737, B787-9
        r"\bA3[2-5]\d(?:-\w+)?\b",        # Airbus models: A320, A350-900
        r"\bMEL\s*\d+-\d+\b",             # MEL codes: MEL 21-31
        r"\bATA\s*\d+\b",                 # ATA chapters: ATA 21
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",  # Dates
    ]
    
    def __init__(self):
        self.documents: List[DocumentChunk] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.AIRLINE_PATTERNS
        ]
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Custom tokenization that preserves airline-specific terms.
        """
        # Extract special patterns first
        special_tokens = []
        for pattern in self._compiled_patterns:
            matches = pattern.findall(text)
            special_tokens.extend([m.upper() for m in matches])
        
        # Standard tokenization
        text_lower = text.lower()
        # Remove special characters but keep alphanumeric
        text_clean = re.sub(r'[^\w\s-]', ' ', text_lower)
        standard_tokens = text_clean.split()
        
        # Remove stopwords
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'as', 'or', 'and', 'but', 'if', 'then', 'than',
            'when', 'where', 'which', 'who', 'whom', 'this', 'that'
        }
        
        filtered_tokens = [t for t in standard_tokens if t not in stopwords and len(t) > 1]
        
        # Combine and deduplicate while preserving order
        all_tokens = special_tokens + filtered_tokens
        seen = set()
        unique_tokens = []
        for t in all_tokens:
            if t.lower() not in seen:
                seen.add(t.lower())
                unique_tokens.append(t)
        
        return unique_tokens
    
    def add_documents(self, documents: List[DocumentChunk]) -> None:
        """Add documents to BM25 index."""
        self.documents.extend(documents)
        
        # Tokenize all documents
        for doc in documents:
            tokens = self._tokenize(doc.content)
            self.tokenized_corpus.append(tokens)
        
        # Rebuild BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        logger.info("sparse_documents_indexed", count=len(documents), total=len(self.documents))
    
    def search(
        self,
        query: str,
        k: int = 10,
        metadata_filter: Optional[dict] = None
    ) -> List[Tuple[RetrievedDocument, float]]:
        """
        Search using BM25 scoring.
        
        Args:
            query: Search query
            k: Number of results
            metadata_filter: Filter by metadata fields
            
        Returns:
            List of (RetrievedDocument, score) tuples
        """
        if not self.documents or not self.bm25:
            logger.warning("sparse_search_empty_index")
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            logger.warning("sparse_search_empty_query", query=query)
            return []
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        # Build results with filtering
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
                
            doc = self.documents[idx]
            
            # Apply metadata filter
            if metadata_filter:
                if not self._matches_filter(doc.metadata, metadata_filter):
                    continue
            
            # Normalize score to 0-1 range
            max_score = max(scores) if max(scores) > 0 else 1
            normalized_score = scores[idx] / max_score
            
            results.append((
                RetrievedDocument(
                    chunk_id=doc.chunk_id,
                    content=doc.content,
                    metadata=doc.metadata,
                    score=normalized_score,
                    retrieval_method="sparse"
                ),
                normalized_score
            ))
            
            if len(results) >= k:
                break
        
        logger.info(
            "sparse_search",
            query=query[:30],
            tokens=query_tokens[:5],
            results=len(results)
        )
        return results
    
    def _matches_filter(self, metadata: DocumentMetadata, filters: dict) -> bool:
        """Check if metadata matches filter criteria."""
        for key, value in filters.items():
            if hasattr(metadata, key):
                if getattr(metadata, key) != value:
                    return False
        return True
    
    def get_term_frequencies(self, query: str) -> Dict[str, int]:
        """Get term frequencies across corpus for analysis."""
        tokens = self._tokenize(query)
        frequencies = {}
        for token in tokens:
            count = sum(1 for doc_tokens in self.tokenized_corpus if token.lower() in [t.lower() for t in doc_tokens])
            frequencies[token] = count
        return frequencies

