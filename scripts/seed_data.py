"""
Data Seeding Script
Seeds the vector stores with sample airline operations data
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.schemas import DocumentChunk, DocumentMetadata
from backend.retrieval.hybrid import HybridRetriever
from assets.sample_data import get_all_sample_documents


def seed_knowledge_base():
    """Seed the knowledge base with sample data."""
    print("🚀 Seeding Airline Operations Knowledge Base...")
    
    # Initialize retriever
    retriever = HybridRetriever()
    
    # Get sample documents
    sample_docs = get_all_sample_documents()
    print(f"📄 Loading {len(sample_docs)} sample documents...")
    
    # Convert to DocumentChunk objects
    chunks = []
    for doc in sample_docs:
        metadata = doc["metadata"]
        chunks.append(DocumentChunk(
            chunk_id=doc["chunk_id"],
            content=doc["content"],
            metadata=DocumentMetadata(
                source_file=metadata["source_file"],
                document_type=metadata.get("document_type", "sop"),
                fleet_type=metadata.get("fleet_type"),
                section=metadata.get("title")
            )
        ))
    
    # Add to hybrid retriever
    print("📊 Indexing documents in hybrid retriever...")
    retriever.add_documents(chunks)
    
    print(f"✅ Successfully indexed {len(chunks)} documents!")
    print("\nSample queries you can try:")
    print("  - What is the pre-flight inspection procedure for B737?")
    print("  - What are the engine start steps for B787?")
    print("  - What is MEL 21-31?")
    print("  - Why was flight UA234 delayed?")
    print("  - What are the delay codes for maintenance issues?")
    
    return retriever


def test_retrieval(retriever):
    """Test retrieval with sample queries."""
    print("\n🔍 Testing Retrieval...\n")
    
    test_queries = [
        "B737 pre-flight checklist",
        "engine start procedure B787",
        "maintenance delay codes",
        "MEL pack valve"
    ]
    
    for query in test_queries:
        print(f"Query: {query}")
        result = retriever.search(query, k=2)
        print(f"  Found {len(result.documents)} documents")
        if result.documents:
            top = result.documents[0]
            print(f"  Top result: {top.chunk_id} (score: {top.score:.3f})")
        print()


if __name__ == "__main__":
    retriever = seed_knowledge_base()
    test_retrieval(retriever)
    print("✨ Seeding complete!")

