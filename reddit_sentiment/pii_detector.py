import re
from dataclasses import dataclass
from typing import List, Tuple
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

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
        'email': (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 0.9),
        'phone': (r'\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b', 0.8),
        'ssn': (r'\b\d{3}-\d{2}-\d{4}\b', 0.95),
        'credit_card': (r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', 0.9),
        'address': (r'\b\d+\s+[A-Za-z\s]+(?:street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr)\b', 0.7),
        'name_pattern': (r'\b(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', 0.6),
    }

    # Keywords that might indicate PII context
    CONTEXT_KEYWORDS = [
        'my name is', 'i live', 'my address', 'you can reach me at',
        'my phone', 'my email', 'contact me', 'call me at'
    ]

    def __init__(self):
        self.compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, (pattern, _) in self.PATTERNS.items()
        }

    def analyze_text(self, text: str) -> List[PIIMatch]:
        """
        Analyze text for potential PII.
        Returns a list of PIIMatch objects for each PII instance found.
        """
        matches = []
        
        # Check for PII patterns
        for pii_type, (_, confidence) in self.PATTERNS.items():
            pattern = self.compiled_patterns[pii_type]
            found = pattern.findall(text)
            if found:
                for match in found:
                    match_str = match if isinstance(match, str) else match[0]
                    matches.append(PIIMatch(pii_type, match_str, confidence))

        # Adjust confidence based on context
        for match in matches:
            for keyword in self.CONTEXT_KEYWORDS:
                if keyword.lower() in text.lower():
                    # Increase confidence if contextual keywords are present
                    match.confidence = min(1.0, match.confidence + 0.1)

        return matches

    def get_pii_risk_score(self, text: str, progress=None) -> Tuple[float, List[PIIMatch]]:
        """
        Calculate overall PII risk score for a text and return matches.
        Returns a tuple of (risk_score, matches).
        """
        matches = self.analyze_text(text)
        if not matches:
            return 0.0, []
            
        # Calculate weighted average of confidence scores
        total_confidence = sum(match.confidence for match in matches)
        risk_score = min(1.0, total_confidence / len(matches))
        
        return risk_score, matches
