"""
Dense Retrieval using FAISS and OpenAI Embeddings
Semantic similarity search for airline operations documents
"""
from typing import List, Optional, Tuple
import numpy as np
import structlog
from openai import OpenAI

from backend.config import settings
from backend.schemas import RetrievedDocument, DocumentChunk, DocumentMetadata

logger = structlog.get_logger()


class DenseRetriever:
    """
    Dense retrieval using vector embeddings.
    Uses OpenAI embeddings with FAISS for fast similarity search.
    """
    
    def __init__(
        self,
        index_path: Optional[str] = None,
        dimension: int = 1536
    ):
        self.dimension = dimension
        self.index_path = index_path or settings.vectorstore.faiss_index_path
        self._client = None  # Lazy initialization
        
        # Document storage (in production, use a proper database)
        self.documents: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self._index = None

    @property
    def client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            api_key = settings.openai_api_key
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self._client = OpenAI(api_key=api_key)
        return self._client

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        response = self.client.embeddings.create(
            model=settings.embedding.model_name,
            input=text
        )
        return response.data[0].embedding
    
    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        response = self.client.embeddings.create(
            model=settings.embedding.model_name,
            input=texts
        )
        return [item.embedding for item in response.data]
    
    def add_documents(self, documents: List[DocumentChunk]) -> None:
        """Add documents to the dense index."""
        try:
            import faiss
        except ImportError:
            logger.warning("FAISS not installed, using numpy fallback")
            faiss = None
        
        # Get embeddings for all documents
        texts = [doc.content for doc in documents]
        embeddings = self._get_embeddings_batch(texts)
        
        # Store documents
        self.documents.extend(documents)
        
        # Build/update index
        new_embeddings = np.array(embeddings, dtype=np.float32)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        # Create FAISS index if available
        if faiss:
            self._index = faiss.IndexFlatIP(self.dimension)  # Inner product
            faiss.normalize_L2(self.embeddings)
            self._index.add(self.embeddings)
        
        logger.info("documents_indexed", count=len(documents), total=len(self.documents))
    
    def search(
        self,
        query: str,
        k: int = 10,
        metadata_filter: Optional[dict] = None
    ) -> List[Tuple[RetrievedDocument, float]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results
            metadata_filter: Filter by metadata fields
            
        Returns:
            List of (RetrievedDocument, score) tuples
        """
        if not self.documents:
            logger.warning("search_empty_index")
            return []
        
        # Get query embedding
        query_embedding = np.array([self._get_embedding(query)], dtype=np.float32)
        
        try:
            import faiss
            if self._index:
                faiss.normalize_L2(query_embedding)
                scores, indices = self._index.search(query_embedding, k * 2)
                scores = scores[0]
                indices = indices[0]
            else:
                raise ImportError
        except ImportError:
            # Numpy fallback
            normalized_query = query_embedding / np.linalg.norm(query_embedding)
            normalized_docs = self.embeddings / np.linalg.norm(
                self.embeddings, axis=1, keepdims=True
            )
            scores = np.dot(normalized_docs, normalized_query.T).flatten()
            indices = np.argsort(scores)[::-1][:k * 2]
            scores = scores[indices]
        
        # Build results with metadata filtering
        results = []
        for idx, score in zip(indices, scores):
            if idx < 0 or idx >= len(self.documents):
                continue
                
            doc = self.documents[idx]
            
            # Apply metadata filter
            if metadata_filter:
                if not self._matches_filter(doc.metadata, metadata_filter):
                    continue
            
            results.append((
                RetrievedDocument(
                    chunk_id=doc.chunk_id,
                    content=doc.content,
                    metadata=doc.metadata,
                    score=float(score),
                    retrieval_method="dense"
                ),
                float(score)
            ))
            
            if len(results) >= k:
                break
        
        logger.info("dense_search", query=query[:30], results=len(results))
        return results
    
    def _matches_filter(self, metadata: DocumentMetadata, filters: dict) -> bool:
        """Check if metadata matches filter criteria."""
        for key, value in filters.items():
            if hasattr(metadata, key):
                if getattr(metadata, key) != value:
                    return False
        return True
    
    def save_index(self, path: Optional[str] = None) -> None:
        """Save index to disk."""
        path = path or self.index_path
        try:
            import faiss
            if self._index:
                faiss.write_index(self._index, f"{path}.faiss")
            np.save(f"{path}_embeddings.npy", self.embeddings)
            logger.info("index_saved", path=path)
        except Exception as e:
            logger.error("index_save_failed", error=str(e))
    
    def load_index(self, path: Optional[str] = None) -> None:
        """Load index from disk."""
        path = path or self.index_path
        try:
            import faiss
            self._index = faiss.read_index(f"{path}.faiss")
            self.embeddings = np.load(f"{path}_embeddings.npy")
            logger.info("index_loaded", path=path)
        except Exception as e:
            logger.error("index_load_failed", error=str(e))

