"""
Query Decomposition Agent for Agentic RAG
Breaks complex operational questions into atomic sub-queries for multi-step reasoning
"""
from typing import List, Optional
from dataclasses import dataclass
import structlog

from backend.schemas import QueryType, SubQuery

logger = structlog.get_logger()


@dataclass
class DecompositionResult:
    """Result of query decomposition"""
    original_query: str
    sub_queries: List[SubQuery]
    reasoning_chain: List[str]
    requires_aggregation: bool


class QueryDecomposer:
    """
    Decomposes complex airline operational queries into atomic sub-queries.
    Enables multi-step reasoning for better answer quality.
    
    Example:
        "Why was flight UA234 delayed due to maintenance?"
        →
        1. "What is the standard maintenance procedure for UA234's aircraft type?"
        2. "What maintenance issues were reported for UA234?"
        3. "What are the operational constraints for delays due to maintenance?"
    """
    
    # Decomposition templates by query type
    DECOMPOSITION_TEMPLATES = {
        QueryType.DELAY_ANALYSIS: [
            "Standard operating procedure for: {topic}",
            "Known root causes for: {topic}",
            "Operational constraints related to: {topic}",
            "Historical patterns for: {topic}"
        ],
        QueryType.MAINTENANCE_REASONING: [
            "Maintenance procedure for: {topic}",
            "Required inspections for: {topic}",
            "MEL exceptions applicable to: {topic}",
            "Turnaround time requirements for: {topic}"
        ],
        QueryType.SOP_LOOKUP: [
            "Primary procedure for: {topic}",
            "Exception handling for: {topic}",
            "Safety considerations for: {topic}"
        ],
        QueryType.CREW_OPERATIONS: [
            "Crew duty regulations for: {topic}",
            "Rest requirements for: {topic}",
            "Staffing protocols for: {topic}"
        ]
    }
    
    # Complexity indicators that trigger decomposition
    COMPLEXITY_INDICATORS = [
        "why", "how", "explain", "analyze", "compare",
        "what are the factors", "root cause", "step by step",
        "and also", "in addition", "considering"
    ]
    
    def __init__(self, max_sub_queries: int = 4):
        self.max_sub_queries = max_sub_queries
    
    def should_decompose(self, query: str, query_type: QueryType) -> bool:
        """Determine if query is complex enough to warrant decomposition."""
        query_lower = query.lower()
        
        # Simple queries don't need decomposition
        if len(query.split()) < 8:
            return False
        
        # Check for complexity indicators
        has_complexity = any(
            indicator in query_lower 
            for indicator in self.COMPLEXITY_INDICATORS
        )
        
        # Delay and maintenance queries benefit most from decomposition
        if query_type in [QueryType.DELAY_ANALYSIS, QueryType.MAINTENANCE_REASONING]:
            return has_complexity or len(query.split()) > 12
        
        return has_complexity
    
    def decompose(
        self, 
        query: str, 
        query_type: QueryType,
        entities: Optional[dict] = None
    ) -> DecompositionResult:
        """
        Decompose query into atomic sub-queries.
        
        Args:
            query: Original user query
            query_type: Classified query type
            entities: Extracted entities (flights, airports, etc.)
            
        Returns:
            DecompositionResult with sub-queries and reasoning chain
        """
        if not self.should_decompose(query, query_type):
            # Return single query if no decomposition needed
            return DecompositionResult(
                original_query=query,
                sub_queries=[
                    SubQuery(
                        query_text=query,
                        query_type=query_type,
                        priority=1
                    )
                ],
                reasoning_chain=["Direct lookup - no decomposition required"],
                requires_aggregation=False
            )
        
        # Extract topic from query
        topic = self._extract_topic(query)
        
        # Get templates for query type
        templates = self.DECOMPOSITION_TEMPLATES.get(
            query_type, 
            self.DECOMPOSITION_TEMPLATES[QueryType.SOP_LOOKUP]
        )
        
        # Generate sub-queries
        sub_queries = []
        reasoning_chain = []
        
        for i, template in enumerate(templates[:self.max_sub_queries]):
            sub_query_text = template.format(topic=topic)
            
            # Determine sub-query type (may differ from parent)
            sub_type = self._infer_sub_query_type(template, query_type)
            
            sub_queries.append(SubQuery(
                query_text=sub_query_text,
                query_type=sub_type,
                priority=i + 1,
                depends_on=[f"sq_{j}" for j in range(i)] if i > 0 else None
            ))
            
            reasoning_chain.append(
                f"Step {i+1}: Retrieve information about '{sub_query_text}'"
            )
        
        logger.info(
            "query_decomposed",
            original=query[:50],
            sub_query_count=len(sub_queries)
        )
        
        return DecompositionResult(
            original_query=query,
            sub_queries=sub_queries,
            reasoning_chain=reasoning_chain,
            requires_aggregation=True
        )
    
    def _extract_topic(self, query: str) -> str:
        """Extract main topic from query for template filling."""
        # Remove common question words
        stopwords = ["what", "why", "how", "when", "where", "is", "are", "the", "a", "an"]
        words = query.lower().split()
        topic_words = [w for w in words if w not in stopwords]
        return " ".join(topic_words[:8])  # Limit topic length
    
    def _infer_sub_query_type(
        self, 
        template: str, 
        parent_type: QueryType
    ) -> QueryType:
        """Infer appropriate query type for sub-query."""
        template_lower = template.lower()
        if "procedure" in template_lower or "sop" in template_lower:
            return QueryType.SOP_LOOKUP
        if "maintenance" in template_lower or "inspection" in template_lower:
            return QueryType.MAINTENANCE_REASONING
        return parent_type


# Singleton instance
decomposer = QueryDecomposer()

