"""
Document Ingestion API
Handles PDF upload and processing for the knowledge base
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import Optional
import uuid
import time
import structlog
from pathlib import Path

from backend.schemas import IngestRequest, IngestResponse, DocumentChunk, DocumentMetadata
from backend.vision.layout_parser import LayoutParser
from backend.vision.table_extractor import TableExtractor
from backend.retrieval.hybrid import HybridRetriever
from backend.config import settings

logger = structlog.get_logger()
router = APIRouter(prefix="/ingest", tags=["ingestion"])

# Global retriever instance (in production, use dependency injection)
retriever = HybridRetriever()
layout_parser = LayoutParser()
table_extractor = TableExtractor()


@router.post("/document", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    document_type: str = "sop",
    fleet_type: Optional[str] = None,
    airport_code: Optional[str] = None,
    extract_figures: bool = True,
    background_tasks: BackgroundTasks = None
):
    """
    Ingest a document into the knowledge base.
    
    - Parses PDF structure
    - Extracts text, tables, and figures
    - Chunks content for retrieval
    - Indexes in both dense and sparse stores
    """
    start_time = time.time()
    
    # Validate file
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files are supported")
    
    # Save uploaded file
    doc_id = str(uuid.uuid4())
    temp_path = Path(f"temp/{doc_id}_{file.filename}")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        content = await file.read()
        with open(temp_path, 'wb') as f:
            f.write(content)
        
        # Parse document
        parsed_doc = layout_parser.parse(str(temp_path))
        
        # Extract tables
        tables = []
        if extract_figures:
            tables = table_extractor.extract_tables(str(temp_path))
        
        # Create chunks
        chunks = _create_chunks(
            parsed_doc,
            tables,
            document_type,
            fleet_type,
            airport_code,
            file.filename
        )
        
        # Index chunks
        retriever.add_documents(chunks)
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(
            "document_ingested",
            doc_id=doc_id,
            chunks=len(chunks),
            figures=len(parsed_doc.figures),
            time_ms=processing_time
        )
        
        return IngestResponse(
            success=True,
            document_id=doc_id,
            chunks_created=len(chunks),
            figures_extracted=len(parsed_doc.figures),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error("ingestion_failed", error=str(e))
        raise HTTPException(500, f"Ingestion failed: {str(e)}")
    
    finally:
        # Cleanup temp file
        if temp_path.exists():
            temp_path.unlink()


def _create_chunks(
    parsed_doc,
    tables,
    document_type: str,
    fleet_type: Optional[str],
    airport_code: Optional[str],
    source_file: str
) -> list:
    """Create document chunks for indexing."""
    chunks = []
    chunk_size = settings.vectorstore.chunk_size
    overlap = settings.vectorstore.chunk_overlap
    
    # Chunk main text
    text = parsed_doc.text_content
    words = text.split()
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        
        if len(chunk_text) < 50:  # Skip tiny chunks
            continue
        
        # Estimate page number
        char_pos = text.find(chunk_text[:50])
        page_est = (char_pos // 3000) + 1 if char_pos >= 0 else 1
        
        chunks.append(DocumentChunk(
            chunk_id=f"{source_file}:chunk_{len(chunks)}",
            content=chunk_text,
            metadata=DocumentMetadata(
                source_file=source_file,
                page_number=page_est,
                fleet_type=fleet_type,
                airport_code=airport_code,
                document_type=document_type
            ),
            has_figure=False
        ))
    
    # Add table chunks
    for table in tables:
        table_text = table_extractor.table_to_text(table)
        chunks.append(DocumentChunk(
            chunk_id=table.table_id,
            content=table_text,
            metadata=DocumentMetadata(
                source_file=source_file,
                page_number=table.page_number,
                fleet_type=fleet_type,
                airport_code=airport_code,
                document_type=document_type,
                section=f"Table: {table.table_type}"
            ),
            has_figure=False
        ))
    
    return chunks


@router.get("/status/{doc_id}")
async def get_ingestion_status(doc_id: str):
    """Get status of document ingestion."""
    # In production, track async job status
    return {"doc_id": doc_id, "status": "completed"}


@router.delete("/document/{doc_id}")
async def delete_document(doc_id: str):
    """Remove a document from the knowledge base."""
    # In production, remove from vector stores
    logger.info("document_deleted", doc_id=doc_id)
    return {"success": True, "doc_id": doc_id}

