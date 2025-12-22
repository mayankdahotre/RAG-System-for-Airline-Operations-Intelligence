"""
BM25 Sparse Index Implementation
Keyword-based retrieval for exact terminology matching
"""
from typing import List, Optional, Tuple, Dict, Set
from dataclasses import dataclass
import pickle
import re
import math
import structlog

logger = structlog.get_logger()


@dataclass
class BM25Config:
    """BM25 algorithm parameters"""
    k1: float = 1.5  # Term frequency saturation
    b: float = 0.75  # Document length normalization


class BM25Index:
    """
    BM25 (Best Matching 25) sparse retrieval index.
    
    Critical for airline operations where exact terminology matters:
    - Flight codes (UA1234)
    - MEL references (MEL 21-31)
    - Aircraft types (B737-MAX)
    - Airport codes (KORD, ORD)
    
    BM25 Formula:
    score(D,Q) = Σ IDF(q) * (f(q,D) * (k1 + 1)) / (f(q,D) + k1 * (1 - b + b * |D|/avgdl))
    """
    
    def __init__(self, config: Optional[BM25Config] = None):
        self.config = config or BM25Config()
        
        # Document storage
        self.documents: Dict[str, str] = {}  # doc_id -> content
        self.metadata: Dict[str, Dict] = {}  # doc_id -> metadata
        
        # Inverted index
        self.inverted_index: Dict[str, Set[str]] = {}  # term -> set of doc_ids
        self.doc_term_freqs: Dict[str, Dict[str, int]] = {}  # doc_id -> {term: freq}
        self.doc_lengths: Dict[str, int] = {}  # doc_id -> length
        
        # Corpus statistics
        self.total_docs = 0
        self.avg_doc_length = 0.0
        self.idf_cache: Dict[str, float] = {}
        
        # Tokenization patterns for airline domain
        self._airline_patterns = [
            re.compile(r'\b[A-Z]{2}\d{3,4}\b'),  # Flight numbers
            re.compile(r'\bMEL\s*\d+-\d+\b', re.IGNORECASE),  # MEL codes
            re.compile(r'\bB7[3-8]\d(?:-\w+)?\b', re.IGNORECASE),  # Boeing
            re.compile(r'\bA3[2-5]\d(?:-\w+)?\b', re.IGNORECASE),  # Airbus
            re.compile(r'\b[A-Z]{3,4}\b'),  # Airport codes
        ]
    
    def add(self, doc_id: str, content: str, metadata: Optional[Dict] = None):
        """Add a document to the index."""
        # Tokenize
        tokens = self._tokenize(content)
        
        if not tokens:
            return
        
        # Store document
        self.documents[doc_id] = content
        self.metadata[doc_id] = metadata or {}
        
        # Calculate term frequencies
        term_freqs = {}
        for token in tokens:
            term_freqs[token] = term_freqs.get(token, 0) + 1
        
        self.doc_term_freqs[doc_id] = term_freqs
        self.doc_lengths[doc_id] = len(tokens)
        
        # Update inverted index
        for term in term_freqs:
            if term not in self.inverted_index:
                self.inverted_index[term] = set()
            self.inverted_index[term].add(doc_id)
        
        # Update corpus stats
        self.total_docs += 1
        total_length = sum(self.doc_lengths.values())
        self.avg_doc_length = total_length / self.total_docs
        
        # Invalidate IDF cache
        self.idf_cache.clear()
    
    def add_batch(
        self,
        doc_ids: List[str],
        contents: List[str],
        metadata_list: Optional[List[Dict]] = None
    ):
        """Add multiple documents."""
        for i, (doc_id, content) in enumerate(zip(doc_ids, contents)):
            meta = metadata_list[i] if metadata_list else None
            self.add(doc_id, content, meta)
        
        logger.info("bm25_batch_indexed", count=len(doc_ids), total=self.total_docs)
    
    def search(
        self,
        query: str,
        k: int = 10,
        filter_fn: Optional[callable] = None
    ) -> List[Tuple[str, float, Dict]]:
        """
        Search for relevant documents.
        
        Returns:
            List of (doc_id, score, metadata) tuples
        """
        if self.total_docs == 0:
            return []
        
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            return []
        
        # Calculate scores for documents containing query terms
        scores: Dict[str, float] = {}
        
        for term in query_tokens:
            if term not in self.inverted_index:
                continue
            
            idf = self._get_idf(term)
            
            for doc_id in self.inverted_index[term]:
                # Apply filter
                if filter_fn and not filter_fn(self.metadata.get(doc_id, {})):
                    continue
                
                tf = self.doc_term_freqs[doc_id].get(term, 0)
                doc_len = self.doc_lengths[doc_id]
                
                # BM25 scoring
                numerator = tf * (self.config.k1 + 1)
                denominator = tf + self.config.k1 * (
                    1 - self.config.b + self.config.b * doc_len / self.avg_doc_length
                )
                
                term_score = idf * (numerator / denominator)
                scores[doc_id] = scores.get(doc_id, 0) + term_score
        
        # Sort and return top-k
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        return [
            (doc_id, score, self.metadata.get(doc_id, {}))
            for doc_id, score in sorted_docs
        ]
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text with airline-specific handling."""
        # Extract special patterns first (preserve them)
        special_tokens = []
        for pattern in self._airline_patterns:
            matches = pattern.findall(text)
            special_tokens.extend([m.upper() for m in matches])
        
        # Standard tokenization
        text_lower = text.lower()
        text_clean = re.sub(r'[^\w\s-]', ' ', text_lower)
        tokens = text_clean.split()
        
        # Filter stopwords
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from'
        }
        
        filtered = [t for t in tokens if t not in stopwords and len(t) > 1]
        
        # Combine with special tokens
        return special_tokens + filtered
    
    def _get_idf(self, term: str) -> float:
        """Calculate IDF for a term."""
        if term in self.idf_cache:
            return self.idf_cache[term]
        
        doc_freq = len(self.inverted_index.get(term, set()))
        if doc_freq == 0:
            return 0.0
        
        # IDF with smoothing
        idf = math.log((self.total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
        self.idf_cache[term] = idf
        
        return idf
    
    def save(self, path: str):
        """Save index to disk."""
        data = {
            "documents": self.documents,
            "metadata": self.metadata,
            "inverted_index": {k: list(v) for k, v in self.inverted_index.items()},
            "doc_term_freqs": self.doc_term_freqs,
            "doc_lengths": self.doc_lengths,
            "total_docs": self.total_docs,
            "avg_doc_length": self.avg_doc_length
        }
        
        with open(f"{path}.bm25", "wb") as f:
            pickle.dump(data, f)
        
        logger.info("bm25_index_saved", path=path, docs=self.total_docs)
    
    def load(self, path: str):
        """Load index from disk."""
        with open(f"{path}.bm25", "rb") as f:
            data = pickle.load(f)
        
        self.documents = data["documents"]
        self.metadata = data["metadata"]
        self.inverted_index = {k: set(v) for k, v in data["inverted_index"].items()}
        self.doc_term_freqs = data["doc_term_freqs"]
        self.doc_lengths = data["doc_lengths"]
        self.total_docs = data["total_docs"]
        self.avg_doc_length = data["avg_doc_length"]
        self.idf_cache.clear()
        
        logger.info("bm25_index_loaded", path=path, docs=self.total_docs)
    
    @property
    def size(self) -> int:
        return self.total_docs

