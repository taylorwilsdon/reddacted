from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class AnalysisResult:
    """Holds the results of both sentiment and PII analysis."""

    comment_id: str
    sentiment_score: float
    sentiment_emoji: str
    pii_risk_score: float
    pii_matches: List[Any]
    permalink: str
    text: str
    upvotes: int = 0
    downvotes: int = 0
    llm_risk_score: float = 0.0
    llm_findings: Optional[Dict[str, Any]] = None
