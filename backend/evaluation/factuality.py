"""
Factuality Evaluation for RAG Responses
Detects hallucinations and unsupported claims
"""
from typing import List, Optional, Dict
from dataclasses import dataclass
import re
import structlog
from openai import OpenAI

from backend.config import settings
from backend.schemas import FactualityResult

logger = structlog.get_logger()


@dataclass
class ClaimAnalysis:
    """Analysis of a single claim"""
    claim: str
    is_supported: bool
    supporting_evidence: Optional[str] = None
    confidence: float = 0.0


class FactualityEvaluator:
    """
    Evaluates factuality of RAG responses against source context.
    Uses both rule-based and LLM-as-a-Judge approaches.
    
    Critical for airline operations where hallucinations = safety risk.
    """
    
    # Claims that should always be verified more strictly
    CRITICAL_PATTERNS = [
        r"must\s+(?:always|never)",
        r"required\s+by\s+(?:law|regulation|faa)",
        r"safety\s+critical",
        r"emergency\s+procedure",
        r"abort\s+",
        r"maximum\s+\d+",
        r"minimum\s+\d+",
        r"within\s+\d+\s+(?:minutes|hours|feet|nm)",
    ]
    
    LLM_JUDGE_PROMPT = """You are a factuality judge for airline operations content. 
Your task is to verify if a claim is supported by the provided context.

CONTEXT:
{context}

CLAIM TO VERIFY:
{claim}

Instructions:
1. Check if the claim can be directly supported by the context
2. If the claim contains specific numbers, verify they match the context
3. If the claim is about procedures, verify the steps match
4. Be strict - if the context doesn't explicitly support the claim, mark it as unsupported

Respond with ONLY:
SUPPORTED: [yes/no]
EVIDENCE: [quote the supporting text if supported, or explain why not if unsupported]
CONFIDENCE: [0.0 to 1.0]"""

    def __init__(self, use_llm_judge: bool = True):
        self.use_llm_judge = use_llm_judge
        self._compiled_critical = [
            re.compile(p, re.IGNORECASE) for p in self.CRITICAL_PATTERNS
        ]
        self._client = None  # Lazy initialization

    @property
    def client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            api_key = settings.openai_api_key
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self._client = OpenAI(api_key=api_key)
        return self._client
    
    def evaluate(
        self,
        answer: str,
        context: str,
        strict_mode: bool = False
    ) -> FactualityResult:
        """
        Evaluate factuality of an answer against context.
        
        Args:
            answer: Generated answer to evaluate
            context: Source context to verify against
            strict_mode: Use stricter verification (for safety-critical content)
            
        Returns:
            FactualityResult with supported/unsupported claims
        """
        # Extract claims
        claims = self._extract_claims(answer)
        
        if not claims:
            return FactualityResult(
                is_factual=True,
                unsupported_claims=[],
                supported_claims=[],
                factuality_score=1.0
            )
        
        # Analyze each claim
        supported = []
        unsupported = []
        
        for claim in claims:
            is_critical = self._is_critical_claim(claim)
            
            if self.use_llm_judge and (is_critical or strict_mode):
                analysis = self._llm_verify_claim(claim, context)
            else:
                analysis = self._rule_verify_claim(claim, context)
            
            if analysis.is_supported:
                supported.append(claim)
            else:
                unsupported.append(claim)
        
        # Calculate score
        total = len(claims)
        score = len(supported) / total if total > 0 else 0.0
        
        # Factual only if >80% claims supported and no critical unsupported
        is_factual = score >= 0.8 and not any(
            self._is_critical_claim(c) for c in unsupported
        )
        
        logger.info(
            "factuality_evaluated",
            total_claims=total,
            supported=len(supported),
            unsupported=len(unsupported),
            score=score
        )
        
        return FactualityResult(
            is_factual=is_factual,
            unsupported_claims=unsupported,
            supported_claims=supported,
            factuality_score=score
        )
    
    def _extract_claims(self, answer: str) -> List[str]:
        """Extract verifiable claims from answer."""
        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            # Filter meta-statements
            if len(sentence) < 15:
                continue
            if any(skip in sentence.lower() for skip in [
                "based on", "according to", "the context shows",
                "i found", "here is", "below are"
            ]):
                continue
            claims.append(sentence)
        
        return claims
    
    def _is_critical_claim(self, claim: str) -> bool:
        """Check if a claim is safety-critical."""
        return any(p.search(claim) for p in self._compiled_critical)
    
    def _rule_verify_claim(self, claim: str, context: str) -> ClaimAnalysis:
        """Simple rule-based verification using text matching."""
        claim_lower = claim.lower()
        context_lower = context.lower()
        
        # Extract key terms (4+ char words)
        claim_terms = set(re.findall(r'\b\w{4,}\b', claim_lower))
        context_terms = set(re.findall(r'\b\w{4,}\b', context_lower))
        
        # Calculate overlap
        overlap = len(claim_terms & context_terms)
        coverage = overlap / len(claim_terms) if claim_terms else 0
        
        # Check for number consistency
        claim_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', claim)
        if claim_numbers:
            numbers_in_context = all(n in context for n in claim_numbers)
            if not numbers_in_context:
                coverage *= 0.5  # Penalize if numbers don't match
        
        is_supported = coverage >= 0.6
        
        return ClaimAnalysis(
            claim=claim,
            is_supported=is_supported,
            confidence=coverage
        )
    
    def _llm_verify_claim(self, claim: str, context: str) -> ClaimAnalysis:
        """LLM-as-a-Judge verification for critical claims."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{
                    "role": "user",
                    "content": self.LLM_JUDGE_PROMPT.format(
                        context=context[:3000],
                        claim=claim
                    )
                }],
                temperature=0,
                max_tokens=200
            )
            
            result = response.choices[0].message.content
            
            # Parse response
            is_supported = "supported: yes" in result.lower()
            confidence_match = re.search(r'confidence:\s*([\d.]+)', result.lower())
            confidence = float(confidence_match.group(1)) if confidence_match else 0.5
            
            return ClaimAnalysis(
                claim=claim,
                is_supported=is_supported,
                supporting_evidence=result,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error("llm_judge_failed", error=str(e))
            return self._rule_verify_claim(claim, context)


# Singleton instance
factuality_evaluator = FactualityEvaluator()

