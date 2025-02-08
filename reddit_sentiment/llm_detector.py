import os
import asyncio
import logging
from typing import Tuple, Dict, Any, List
import openai

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)


class LLMDetector:
    """Uses LLM to detect potential PII and personal information in text"""

    DEFAULT_PROMPT = """
    Analyze the following text for any information that could potentially identify the author or reveal personal details about them.
    Consider both explicit PII (like names, addresses) and implicit personal information (like specific life events, locations, relationships).
    
    Respond in JSON format with these fields:
    - has_pii: boolean
    - confidence: float (0-1)
    - details: list of findings with specific examples from the text
    - reasoning: detailed explanation of why this content might identify the author
    - risk_factors: list of specific elements that contribute to the risk score
    
    Text to analyze: {text}
    """

    def __init__(self, api_key: str, api_base: str = None, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.client_config = {
            'api_key': api_key,
        }
        if api_base:
            self.client_config['base_url'] = api_base

    async def analyze_batch(self, texts: List[str]) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Analyze a batch of texts using LLM for potential personal information.
        Returns list of tuples (risk_score, details).
        """
        client = openai.AsyncOpenAI(**self.client_config)
        batch_size = 3
        results = []
        
        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                tasks = []
                
                for text in batch:
                    task = client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a privacy analysis assistant."},
                            {"role": "user", "content": self.DEFAULT_PROMPT.format(text=text)}
                        ],
                        temperature=0.1
                    )
                    logging.debug(f"Using API base: {client.base_url}")
                    logging.debug(f"Using model: {self.model}")
                    tasks.append(task)
                
                batch_responses = await asyncio.gather(*tasks)
                
                for response in batch_responses:
                    try:
                        import json
                        raw_response = response.choices[0].message.content.strip()
                        logging.debug(f"\n🤖 Raw LLM Response:\n{raw_response}\n")
                        try:
                            # First attempt direct parse, sometimes stupid LLM messes up formatting
                            analysis = json.loads(raw_response)
                        except json.JSONDecodeError:
                            # If that fails, try to extract JSON from markdown blocks
                            if "```json" in raw_response:
                                json_content = raw_response.split("```json")[1].split("```")[0].strip()
                                analysis = json.loads(json_content)
                            else:
                                raise
                        
                        # Calculate risk score based on confidence and PII presence
                        confidence = float(analysis.get('confidence', 0.0))
                        has_pii = analysis.get('has_pii', False)
                        
                        logging.debug(f"Parsed confidence: {confidence}")
                        logging.debug(f"Parsed has_pii: {has_pii}")
                        
                        if has_pii:
                            risk_score = confidence
                        else:
                            risk_score = 0.0
                            analysis = {
                                'has_pii': False,
                                'confidence': 0.0,
                                'details': [],
                                'risk_factors': [],
                                'reasoning': "No PII detected"
                            }
                            
                        results.append((risk_score, analysis))
                    except Exception as e:
                        results.append((0.0, {"error": str(e)}))
                
            return results
            
        except Exception as e:
            logging.error("AI analysis failed")
            print(f"Batch LLM analysis failed: {str(e)}")
            return [(0.0, {"error": str(e)})] * len(texts)

    def analyze_text(self, text: str) -> Tuple[float, Dict[str, Any]]:
        """
        Analyze a single text using LLM for potential personal information.
        Returns tuple of (risk_score, details).
        """
        try:
            results = asyncio.run(self.analyze_batch([text]))
            return results[0]
        except Exception as e:
            print(f"LLM analysis failed: {str(e)}")
            return 0.0, {"error": str(e)}
