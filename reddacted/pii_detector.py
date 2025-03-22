import re
from dataclasses import dataclass
from typing import List, Tuple
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from reddacted.utils.logging import get_logger, with_logging
from reddacted.utils.exceptions import handle_exception

logger = get_logger(__name__)


@dataclass
class PIIMatch:
    """Represents a PII match found in text"""

    type: str
    value: str
    confidence: float


class PIIDetector:
    """Detects various types of personally identifiable information in text"""

    # Common PII patterns
    PATTERNS = {
        "email": (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", 0.95),
        "phone": (r"\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b", 0.85),
        "ssn": (r"\b\d{3}-\d{2}-\d{4}\b", 0.97),
        "credit_card": (r"\b(?:\d{4}[- ]?){3}\d{4}\b", 0.95),
        "address": (
            r"\b\d{2,5}\s+(?:[A-Za-z]+\s)+(?:street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr)\.?\b",
            0.65,
        ),
        "name_pattern": (r"\b(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b", 0.7),
    }

    # Keywords that might indicate PII context
    CONTEXT_KEYWORDS = [
        "name is",
        "live at",
        "address",
        "reach me",
        "phone",
        "email",
        "contact",
        "call me",
        "ssn",
        "social security",
        "credit card",
        "driver license",
    ]

    COMMON_FALSE_POSITIVES = [
        r"\b\d+ (llm|ai|gpu|cpu|ram|mb|gb|ghz|mhz|api)\b",
        r"\b\d+ (times|years|days|hours|minutes|seconds)\b",
        r"\b\d+(?:st|nd|rd|th)\b",
        r"\b\d+[km]?b?\b",
    ]

    def __init__(self):
        self.compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE) for name, (pattern, _) in self.PATTERNS.items()
        }
        self.false_positive_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.COMMON_FALSE_POSITIVES
        ]

    @with_logging(logger)
    def analyze_text(self, text: str) -> List[PIIMatch]:
        """
        Analyze text for potential PII.
        Returns a list of PIIMatch objects for each PII instance found.
        """
        matches = []

        # First check for false positives
        if any(fp.search(text) for fp in self.false_positive_patterns):
            return []

        # Validate matches against known false positive contexts
        for pii_type, (_, confidence) in self.PATTERNS.items():
            pattern = self.compiled_patterns[pii_type]
            for match in pattern.finditer(text):
                full_match = match.group(0)

                # Additional validation per type
                if pii_type == "phone" and len(full_match.replace("-", "").replace(" ", "")) < 10:
                    continue

                if pii_type == "address" and not any(c.isalpha() for c in full_match.split()[-2]):
                    continue

                matches.append(PIIMatch(pii_type, full_match, confidence))

        # Contextual confidence boost with cap
        context_boost = (
            0.15
            if any(
                re.search(rf"\b{re.escape(kw)}\b", text, re.IGNORECASE)
                for kw in self.CONTEXT_KEYWORDS
            )
            else 0.0
        )

        for match in matches:
            match.confidence = min(1.0, match.confidence + context_boost)

        return matches

    @with_logging(logger)
    def get_pii_risk_score(self, text: str, progress=None) -> Tuple[float, List[PIIMatch]]:
        """
        Calculate overall PII risk score for a text and return matches.
        Returns a tuple of (risk_score, matches).
        """
        matches = self.analyze_text(text)
        if not matches:
            return 0.0, []

        # Weighted average with type weights
        type_weights = {
            "ssn": 1.2,
            "credit_card": 1.2,
            "email": 1.0,
            "phone": 0.9,
            "address": 0.7,
            "name_pattern": 0.6,
        }

        total_weight = sum(type_weights.get(match.type, 1.0) for match in matches)
        weighted_sum = sum(
            match.confidence * type_weights.get(match.type, 1.0) for match in matches
        )

        return min(1.0, weighted_sum / total_weight), matches
