"""
Query Classifier Agent for Airline Operations
Classifies incoming queries to route to appropriate retrieval strategies
"""
import re
from typing import Tuple, Dict, List
from enum import Enum
import structlog

from backend.schemas import QueryType, QueryRequest

logger = structlog.get_logger()


class QueryClassifier:
    """
    Intelligent query classifier for airline operations.
    Uses rule-based classification with keyword matching and pattern recognition.
    Can be upgraded to LLM-based classification for complex queries.
    """
    
    # Domain-specific keyword patterns
    PATTERNS: Dict[QueryType, List[str]] = {
        QueryType.SOP_LOOKUP: [
            r"procedure", r"sop", r"checklist", r"step[s]?\s+for",
            r"how\s+to", r"protocol", r"guideline", r"standard\s+operating",
            r"before\s+takeoff", r"pre-?flight", r"post-?flight",
            r"emergency\s+procedure", r"abort\s+procedure"
        ],
        QueryType.MAINTENANCE_REASONING: [
            r"maintenance", r"repair", r"fix", r"mechanic",
            r"inspection", r"mro", r"apu", r"engine\s+issue",
            r"hydraulic", r"avionics", r"fault\s+code",
            r"troubleshoot", r"defect", r"mel\b", r"minimum\s+equipment"
        ],
        QueryType.DELAY_ANALYSIS: [
            r"delay", r"late", r"ground\s+stop", r"wait",
            r"why\s+was.*delayed", r"root\s+cause", r"on-?time",
            r"schedule\s+impact", r"atc\s+delay", r"weather\s+delay",
            r"crew\s+delay", r"mechanical\s+delay"
        ],
        QueryType.CREW_OPERATIONS: [
            r"crew", r"pilot", r"flight\s+attendant", r"captain",
            r"first\s+officer", r"duty\s+time", r"rest\s+requirement",
            r"fatigue", r"staffing", r"crew\s+schedule"
        ]
    }
    
    # Priority weights for classification confidence
    PRIORITY_WEIGHTS = {
        QueryType.SOP_LOOKUP: 1.0,
        QueryType.MAINTENANCE_REASONING: 0.95,
        QueryType.DELAY_ANALYSIS: 0.9,
        QueryType.CREW_OPERATIONS: 0.85,
        QueryType.GENERAL: 0.5
    }
    
    def __init__(self):
        # Compile regex patterns for efficiency
        self._compiled_patterns = {
            qtype: [re.compile(p, re.IGNORECASE) for p in patterns]
            for qtype, patterns in self.PATTERNS.items()
        }
    
    def classify(self, request: QueryRequest) -> Tuple[QueryType, float]:
        """
        Classify query and return type with confidence score.
        
        Returns:
            Tuple of (QueryType, confidence_score)
        """
        query = request.query.lower()
        
        scores: Dict[QueryType, float] = {}
        
        for qtype, patterns in self._compiled_patterns.items():
            matches = sum(1 for p in patterns if p.search(query))
            if matches > 0:
                # Calculate confidence based on match count and priority
                confidence = min(0.95, 0.6 + (matches * 0.1))
                scores[qtype] = confidence * self.PRIORITY_WEIGHTS[qtype]
        
        if not scores:
            logger.info("query_classified", query=query[:50], type="general")
            return QueryType.GENERAL, 0.5
        
        # Get highest scoring type
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]
        
        logger.info(
            "query_classified",
            query=query[:50],
            type=best_type.value,
            confidence=confidence
        )
        
        return best_type, confidence
    
    def extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract airline-specific entities from query."""
        entities = {
            "flight_numbers": re.findall(r"\b[A-Z]{2}\d{3,4}\b", query.upper()),
            "airports": re.findall(r"\b[A-Z]{3}\b", query.upper()),
            "fleet_types": re.findall(r"\b(?:B7[3-8]\d|A3[2-5]\d)\b", query.upper()),
            "dates": re.findall(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", query)
        }
        return {k: v for k, v in entities.items() if v}
    
    def get_retrieval_strategy(self, query_type: QueryType) -> Dict[str, any]:
        """Get optimal retrieval configuration for query type."""
        strategies = {
            QueryType.SOP_LOOKUP: {
                "dense_weight": 0.4,
                "sparse_weight": 0.6,  # Exact terminology matters
                "top_k": 5,
                "require_metadata_match": True
            },
            QueryType.MAINTENANCE_REASONING: {
                "dense_weight": 0.5,
                "sparse_weight": 0.5,
                "top_k": 10,
                "require_metadata_match": False
            },
            QueryType.DELAY_ANALYSIS: {
                "dense_weight": 0.7,
                "sparse_weight": 0.3,  # Semantic understanding needed
                "top_k": 8,
                "require_metadata_match": False
            },
            QueryType.GENERAL: {
                "dense_weight": 0.6,
                "sparse_weight": 0.4,
                "top_k": 10,
                "require_metadata_match": False
            }
        }
        return strategies.get(query_type, strategies[QueryType.GENERAL])


# Singleton instance
classifier = QueryClassifier()

