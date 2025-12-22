"""
Document Layout Parser for Multimodal RAG
Extracts text, figures, and structural information from PDFs
"""
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import structlog
import re

logger = structlog.get_logger()


@dataclass
class PageElement:
    """Represents an element on a page"""
    element_type: str  # "text", "figure", "table", "heading", "list"
    content: str
    bounding_box: Optional[Tuple[float, float, float, float]] = None
    page_number: int = 1
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ParsedDocument:
    """Complete parsed document structure"""
    file_path: str
    total_pages: int
    elements: List[PageElement]
    figures: List[PageElement]
    tables: List[PageElement]
    text_content: str
    metadata: Dict[str, Any]


class LayoutParser:
    """
    Intelligent document layout parser for airline SOPs and manuals.
    Handles complex layouts with figures, tables, and procedural content.
    
    Key capabilities:
    1. PDF text extraction with layout preservation
    2. Figure detection and description
    3. Table boundary detection
    4. Section/heading hierarchy extraction
    5. Procedural step identification
    """
    
    # Patterns for airline document structure
    HEADING_PATTERNS = [
        r'^CHAPTER\s+\d+',
        r'^\d+\.\d+\s+[A-Z]',
        r'^SECTION\s+\d+',
        r'^PROCEDURE\s*:',
        r'^WARNING\s*:',
        r'^CAUTION\s*:',
        r'^NOTE\s*:'
    ]
    
    PROCEDURE_PATTERNS = [
        r'^\d+\)\s+',  # Numbered steps
        r'^Step\s+\d+',
        r'^[a-z]\)\s+',  # Lettered steps
        r'^\[\s*\]\s+',  # Checkbox items
    ]
    
    def __init__(self):
        self._compiled_heading = [re.compile(p, re.IGNORECASE) for p in self.HEADING_PATTERNS]
        self._compiled_procedure = [re.compile(p) for p in self.PROCEDURE_PATTERNS]
    
    def parse(self, file_path: str) -> ParsedDocument:
        """
        Parse a document and extract structured content.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            ParsedDocument with extracted elements
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        if path.suffix.lower() != '.pdf':
            raise ValueError(f"Unsupported file type: {path.suffix}")
        
        try:
            return self._parse_pdf(file_path)
        except Exception as e:
            logger.error("parse_failed", file=file_path, error=str(e))
            raise
    
    def _parse_pdf(self, file_path: str) -> ParsedDocument:
        """Parse PDF using pdfplumber."""
        try:
            import pdfplumber
        except ImportError:
            logger.warning("pdfplumber not installed, using basic extraction")
            return self._parse_pdf_basic(file_path)
        
        elements = []
        figures = []
        tables = []
        full_text = []
        
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract text
                text = page.extract_text() or ""
                full_text.append(text)
                
                # Parse text into elements
                page_elements = self._parse_text_elements(text, page_num)
                elements.extend(page_elements)
                
                # Detect tables
                page_tables = self._detect_tables(page, page_num)
                tables.extend(page_tables)
                
                # Detect figures
                page_figures = self._detect_figures(page, page_num)
                figures.extend(page_figures)
        
        metadata = {
            "file_name": Path(file_path).name,
            "total_pages": total_pages,
            "figure_count": len(figures),
            "table_count": len(tables)
        }
        
        logger.info(
            "document_parsed",
            file=file_path,
            pages=total_pages,
            elements=len(elements),
            figures=len(figures),
            tables=len(tables)
        )
        
        return ParsedDocument(
            file_path=file_path,
            total_pages=total_pages,
            elements=elements,
            figures=figures,
            tables=tables,
            text_content="\n".join(full_text),
            metadata=metadata
        )
    
    def _parse_pdf_basic(self, file_path: str) -> ParsedDocument:
        """Basic PDF parsing fallback using pypdf."""
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("No PDF library available. Install pypdf or pdfplumber.")
        
        reader = PdfReader(file_path)
        elements = []
        full_text = []
        
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            full_text.append(text)
            page_elements = self._parse_text_elements(text, page_num)
            elements.extend(page_elements)
        
        return ParsedDocument(
            file_path=file_path,
            total_pages=len(reader.pages),
            elements=elements,
            figures=[],
            tables=[],
            text_content="\n".join(full_text),
            metadata={"file_name": Path(file_path).name}
        )
    
    def _parse_text_elements(self, text: str, page_num: int) -> List[PageElement]:
        """Parse text into structural elements."""
        elements = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            element_type = self._classify_line(line)
            elements.append(PageElement(
                element_type=element_type,
                content=line,
                page_number=page_num
            ))
        
        return elements
    
    def _classify_line(self, line: str) -> str:
        """Classify a line of text."""
        for pattern in self._compiled_heading:
            if pattern.match(line):
                return "heading"
        
        for pattern in self._compiled_procedure:
            if pattern.match(line):
                return "procedure_step"
        
        return "text"
    
    def _detect_tables(self, page, page_num: int) -> List[PageElement]:
        """Detect tables in a page."""
        tables = []
        try:
            extracted_tables = page.extract_tables()
            for i, table in enumerate(extracted_tables):
                if table:
                    table_text = self._format_table(table)
                    tables.append(PageElement(
                        element_type="table",
                        content=table_text,
                        page_number=page_num,
                        metadata={"table_index": i}
                    ))
        except Exception as e:
            logger.debug("table_detection_failed", page=page_num, error=str(e))
        return tables
    
    def _format_table(self, table: List[List[str]]) -> str:
        """Format extracted table as text."""
        rows = []
        for row in table:
            cells = [str(cell or "").strip() for cell in row]
            rows.append(" | ".join(cells))
        return "\n".join(rows)
    
    def _detect_figures(self, page, page_num: int) -> List[PageElement]:
        """Detect figures/images in a page."""
        figures = []
        try:
            images = page.images
            for i, img in enumerate(images):
                figures.append(PageElement(
                    element_type="figure",
                    content=f"[Figure on page {page_num}]",
                    page_number=page_num,
                    bounding_box=(img['x0'], img['top'], img['x1'], img['bottom']),
                    metadata={"figure_index": i}
                ))
        except Exception as e:
            logger.debug("figure_detection_failed", page=page_num, error=str(e))
        return figures

