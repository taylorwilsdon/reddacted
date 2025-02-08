import os
from typing import Tuple, Dict, Any
import openai

class LLMDetector:
    """Uses LLM to detect potential PII and personal information in text"""

    DEFAULT_PROMPT = """
    Analyze the following text for any information that could potentially identify the author or reveal personal details about them.
    Consider both explicit PII (like names, addresses) and implicit personal information (like specific life events, locations, relationships).
    
    Respond in JSON format with these fields:
    - has_pii: boolean
    - confidence: float (0-1)
    - details: list of findings
    - reasoning: brief explanation
    
    Text to analyze: {text}
    """

    def __init__(self, api_key: str, api_base: str = None, model: str = "gpt-3.5-turbo"):
        self.model = model
        openai.api_key = api_key
        if api_base:
            openai.api_base = api_base

    def analyze_text(self, text: str) -> Tuple[float, Dict[str, Any]]:
        """
        Analyze text using LLM for potential personal information.
        Returns tuple of (risk_score, details).
        """
        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a privacy analysis assistant."},
                    {"role": "user", "content": self.DEFAULT_PROMPT.format(text=text)}
                ],
                temperature=0.1
            )
            
            result = response.choices[0].message.content
            # Note: In production, add proper JSON parsing with error handling
            import json
            analysis = json.loads(result)
            
            return (
                float(analysis.get('confidence', 0.0)),
                analysis
            )
            
        except Exception as e:
            print(f"LLM analysis failed: {str(e)}")
            return 0.0, {"error": str(e)}
