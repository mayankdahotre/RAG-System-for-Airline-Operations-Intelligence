"""
Table Extraction for Structured Data
Specialized extraction for airline operational tables
"""
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import structlog
import re

logger = structlog.get_logger()


@dataclass
class ExtractedTable:
    """Structured representation of an extracted table"""
    table_id: str
    headers: List[str]
    rows: List[List[str]]
    page_number: int
    table_type: str  # "checklist", "mel", "limits", "schedule", "general"
    source_file: str
    confidence: float = 1.0
    
    def to_markdown(self) -> str:
        """Convert to markdown table format."""
        if not self.headers:
            return ""
        
        md = "| " + " | ".join(self.headers) + " |\n"
        md += "| " + " | ".join(["---"] * len(self.headers)) + " |\n"
        for row in self.rows:
            md += "| " + " | ".join(row) + " |\n"
        return md
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for structured queries."""
        return {
            "table_id": self.table_id,
            "headers": self.headers,
            "rows": [dict(zip(self.headers, row)) for row in self.rows],
            "table_type": self.table_type
        }


class TableExtractor:
    """
    Specialized table extraction for airline documents.
    Handles common airline table formats:
    - Checklists (pre-flight, emergency, etc.)
    - MEL (Minimum Equipment Lists)
    - Operational limits tables
    - Crew schedules
    - Maintenance logs
    """
    
    # Patterns to identify table types
    TABLE_TYPE_PATTERNS = {
        "checklist": [r"checklist", r"check\s*list", r"verification", r"items to check"],
        "mel": [r"mel", r"minimum equipment", r"dispatch deviation", r"deferral"],
        "limits": [r"limits", r"maximum", r"minimum", r"operating limits"],
        "schedule": [r"schedule", r"duty time", r"crew", r"rotation"],
        "maintenance": [r"maintenance", r"inspection", r"mro", r"service"]
    }
    
    def __init__(self):
        self._compiled_patterns = {
            ttype: [re.compile(p, re.IGNORECASE) for p in patterns]
            for ttype, patterns in self.TABLE_TYPE_PATTERNS.items()
        }
    
    def extract_tables(
        self,
        file_path: str,
        page_numbers: Optional[List[int]] = None
    ) -> List[ExtractedTable]:
        """
        Extract all tables from a document.
        
        Args:
            file_path: Path to PDF
            page_numbers: Optional specific pages to extract
            
        Returns:
            List of ExtractedTable objects
        """
        try:
            import pdfplumber
        except ImportError:
            logger.warning("pdfplumber not available for table extraction")
            return []
        
        tables = []
        
        with pdfplumber.open(file_path) as pdf:
            pages = pdf.pages
            if page_numbers:
                pages = [pdf.pages[i-1] for i in page_numbers if i <= len(pdf.pages)]
            
            for page in pages:
                page_tables = self._extract_page_tables(
                    page, page.page_number, file_path
                )
                tables.extend(page_tables)
        
        logger.info("tables_extracted", file=file_path, count=len(tables))
        return tables
    
    def _extract_page_tables(
        self,
        page,
        page_num: int,
        file_path: str
    ) -> List[ExtractedTable]:
        """Extract tables from a single page."""
        tables = []
        
        try:
            raw_tables = page.extract_tables()
            page_text = page.extract_text() or ""
            
            for i, raw_table in enumerate(raw_tables):
                if not raw_table or len(raw_table) < 2:
                    continue
                
                # Clean the table data
                cleaned = self._clean_table(raw_table)
                if not cleaned:
                    continue
                
                headers, rows = cleaned
                
                # Classify table type
                table_type = self._classify_table(headers, page_text)
                
                # Generate unique ID
                table_id = f"{file_path}:p{page_num}:t{i}"
                
                tables.append(ExtractedTable(
                    table_id=table_id,
                    headers=headers,
                    rows=rows,
                    page_number=page_num,
                    table_type=table_type,
                    source_file=file_path
                ))
                
        except Exception as e:
            logger.error("table_extraction_failed", page=page_num, error=str(e))
        
        return tables
    
    def _clean_table(self, raw_table: List[List]) -> Optional[tuple]:
        """Clean and validate extracted table data."""
        if not raw_table:
            return None
        
        # Clean cells
        cleaned_rows = []
        for row in raw_table:
            cleaned_row = []
            for cell in row:
                if cell is None:
                    cleaned_row.append("")
                else:
                    # Clean whitespace and newlines
                    cleaned = str(cell).strip().replace('\n', ' ')
                    cleaned_row.append(cleaned)
            cleaned_rows.append(cleaned_row)
        
        if len(cleaned_rows) < 2:
            return None
        
        # First row is typically headers
        headers = cleaned_rows[0]
        rows = cleaned_rows[1:]
        
        # Validate headers
        if not any(h for h in headers):
            return None
        
        return headers, rows
    
    def _classify_table(self, headers: List[str], context_text: str) -> str:
        """Classify the type of table based on headers and context."""
        combined_text = " ".join(headers).lower() + " " + context_text.lower()
        
        for table_type, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(combined_text):
                    return table_type
        
        return "general"
    
    def table_to_text(self, table: ExtractedTable) -> str:
        """Convert table to searchable text for RAG."""
        text_parts = [f"Table Type: {table.table_type}"]
        text_parts.append(f"Headers: {', '.join(table.headers)}")
        
        for i, row in enumerate(table.rows):
            row_text = " | ".join(row)
            text_parts.append(f"Row {i+1}: {row_text}")
        
        return "\n".join(text_parts)
    
    def query_table(
        self,
        table: ExtractedTable,
        column: str,
        value: str
    ) -> List[Dict[str, str]]:
        """
        Query a table for specific values.
        Useful for structured lookups like MEL codes.
        """
        results = []
        
        try:
            col_idx = None
            for i, header in enumerate(table.headers):
                if column.lower() in header.lower():
                    col_idx = i
                    break
            
            if col_idx is None:
                return results
            
            for row in table.rows:
                if col_idx < len(row) and value.lower() in row[col_idx].lower():
                    results.append(dict(zip(table.headers, row)))
                    
        except Exception as e:
            logger.error("table_query_failed", error=str(e))
        
        return results


# Singleton instance
table_extractor = TableExtractor()

