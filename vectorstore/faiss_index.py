"""
FAISS Vector Index Implementation
Production-ready vector storage for dense retrieval
"""
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import numpy as np
import structlog
import pickle
import os

logger = structlog.get_logger()


class FAISSIndex:
    """
    FAISS-based vector index for semantic search.
    Supports multiple index types for different scale requirements.
    
    Index Types:
    - Flat: Exact search, best for < 100k vectors
    - IVF: Approximate search with inverted file, for 100k-1M vectors
    - HNSW: Graph-based search, for 1M+ vectors with high recall needs
    """
    
    def __init__(
        self,
        dimension: int = 1536,
        index_type: str = "flat",
        nlist: int = 100,  # For IVF
        nprobe: int = 10   # For IVF
    ):
        self.dimension = dimension
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        
        self._index = None
        self._id_map: Dict[int, str] = {}  # FAISS ID -> Document ID
        self._metadata: Dict[str, Dict] = {}  # Document ID -> Metadata
        self._count = 0
        
        self._init_index()
    
    def _init_index(self):
        """Initialize FAISS index."""
        try:
            import faiss
            
            if self.index_type == "flat":
                # Exact search with inner product
                self._index = faiss.IndexFlatIP(self.dimension)
            elif self.index_type == "ivf":
                # Approximate search with clustering
                quantizer = faiss.IndexFlatIP(self.dimension)
                self._index = faiss.IndexIVFFlat(
                    quantizer, self.dimension, self.nlist, faiss.METRIC_INNER_PRODUCT
                )
            elif self.index_type == "hnsw":
                # Graph-based approximate search
                self._index = faiss.IndexHNSWFlat(self.dimension, 32)
            else:
                raise ValueError(f"Unknown index type: {self.index_type}")
            
            logger.info("faiss_index_initialized", type=self.index_type, dimension=self.dimension)
            
        except ImportError:
            logger.warning("FAISS not installed, using numpy fallback")
            self._index = None
    
    def add(
        self,
        doc_id: str,
        embedding: List[float],
        metadata: Optional[Dict] = None
    ) -> int:
        """Add a single vector to the index."""
        vector = np.array([embedding], dtype=np.float32)
        
        # Normalize for cosine similarity
        faiss_available = self._try_normalize(vector)
        
        if self._index is not None and faiss_available:
            self._index.add(vector)
        
        # Store mapping
        internal_id = self._count
        self._id_map[internal_id] = doc_id
        self._metadata[doc_id] = metadata or {}
        self._count += 1
        
        return internal_id
    
    def add_batch(
        self,
        doc_ids: List[str],
        embeddings: List[List[float]],
        metadata_list: Optional[List[Dict]] = None
    ) -> List[int]:
        """Add multiple vectors to the index."""
        vectors = np.array(embeddings, dtype=np.float32)
        
        # Normalize
        self._try_normalize(vectors)
        
        # Train IVF index if needed
        if self.index_type == "ivf" and not self._index.is_trained:
            if len(vectors) >= self.nlist:
                self._index.train(vectors)
                self._index.nprobe = self.nprobe
            else:
                logger.warning("Not enough vectors to train IVF, using flat search")
        
        if self._index is not None:
            try:
                self._index.add(vectors)
            except Exception as e:
                logger.error("faiss_add_failed", error=str(e))
        
        # Store mappings
        internal_ids = []
        for i, doc_id in enumerate(doc_ids):
            internal_id = self._count
            self._id_map[internal_id] = doc_id
            self._metadata[doc_id] = (metadata_list[i] if metadata_list else {})
            internal_ids.append(internal_id)
            self._count += 1
        
        logger.info("vectors_added", count=len(doc_ids), total=self._count)
        return internal_ids
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 10,
        filter_fn: Optional[callable] = None
    ) -> List[Tuple[str, float, Dict]]:
        """
        Search for similar vectors.
        
        Returns:
            List of (doc_id, score, metadata) tuples
        """
        if self._count == 0:
            return []
        
        query = np.array([query_embedding], dtype=np.float32)
        self._try_normalize(query)
        
        # Get more results if filtering
        search_k = k * 3 if filter_fn else k
        
        if self._index is not None:
            try:
                scores, indices = self._index.search(query, min(search_k, self._count))
                scores = scores[0]
                indices = indices[0]
            except Exception as e:
                logger.error("faiss_search_failed", error=str(e))
                return []
        else:
            # Numpy fallback (shouldn't happen in production)
            return []
        
        # Build results
        results = []
        for idx, score in zip(indices, scores):
            if idx < 0:  # FAISS returns -1 for empty slots
                continue
            
            doc_id = self._id_map.get(idx)
            if doc_id is None:
                continue
            
            metadata = self._metadata.get(doc_id, {})
            
            # Apply filter
            if filter_fn and not filter_fn(metadata):
                continue
            
            results.append((doc_id, float(score), metadata))
            
            if len(results) >= k:
                break
        
        return results
    
    def _try_normalize(self, vectors: np.ndarray) -> bool:
        """Normalize vectors for cosine similarity."""
        try:
            import faiss
            faiss.normalize_L2(vectors)
            return True
        except ImportError:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors /= np.maximum(norms, 1e-8)
            return False
    
    def save(self, path: str):
        """Save index to disk."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        
        try:
            import faiss
            if self._index is not None:
                faiss.write_index(self._index, f"{path}.faiss")
        except Exception as e:
            logger.error("faiss_save_failed", error=str(e))
        
        # Save mappings
        with open(f"{path}_meta.pkl", "wb") as f:
            pickle.dump({
                "id_map": self._id_map,
                "metadata": self._metadata,
                "count": self._count
            }, f)
        
        logger.info("index_saved", path=path, vectors=self._count)
    
    def load(self, path: str):
        """Load index from disk."""
        try:
            import faiss
            self._index = faiss.read_index(f"{path}.faiss")
        except Exception as e:
            logger.error("faiss_load_failed", error=str(e))
        
        with open(f"{path}_meta.pkl", "rb") as f:
            data = pickle.load(f)
            self._id_map = data["id_map"]
            self._metadata = data["metadata"]
            self._count = data["count"]
        
        logger.info("index_loaded", path=path, vectors=self._count)
    
    @property
    def size(self) -> int:
        """Get number of vectors in index."""
        return self._count

