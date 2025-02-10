import unittest
import json
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

from reddact.llm_detector import LLMDetector

SAMPLE_RESPONSE = {
    "has_pii": True,
    "confidence": 0.85,
    "details": ["Mentions specific location 'Miami Springs'"],
    "reasoning": "Location mention could help identify author's residence",
    "risk_factors": ["geographical specificity", "local slang reference"]
}

class LLMDetectorTestCases(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_client.chat = MagicMock()
        self.mock_completion = MagicMock()
        
        # Mock async response
        self.mock_message = MagicMock()
        self.mock_message.content = json.dumps(SAMPLE_RESPONSE)
        self.mock_choice = MagicMock()
        self.mock_choice.message = self.mock_message
        self.mock_completion.choices = [self.mock_choice]
        
        # Patch the OpenAI client
        self.client_patcher = patch('openai.AsyncOpenAI', return_value=self.mock_client)
        self.mock_openai = self.client_patcher.start()
        self.addCleanup(self.client_patcher.stop)

    def tearDown(self):
        self.client_patcher.stop()

    @patch('openai.AsyncOpenAI')
    def test_analyze_text_success(self, mock_client):
        """Test successful PII analysis with valid response"""
        # Configure mock
        mock_client.return_value.chat.completions.create = AsyncMock(return_value=self.mock_completion)
        
        detector = LLMDetector(api_key="sk-test")
        risk_score, details = detector.analyze_text("RaunchyRaccoon that looks a lot like Miami Springs!")
        
        self.assertEqual(risk_score, 0.85)
        self.assertEqual(details['details'], SAMPLE_RESPONSE['details'])
        self.assertEqual(details['risk_factors'], SAMPLE_RESPONSE['risk_factors'])
        mock_client.assert_called_once_with(
            api_key="sk-test",
            default_headers={}
        )

    @patch('openai.AsyncOpenAI')
    def test_analyze_invalid_key(self, mock_client):
        """Test authentication error handling"""
        mock_client.side_effect = Exception("Invalid API key")
        
        detector = LLMDetector(api_key="invalid-key")
        risk_score, details = detector.analyze_text("Sample text")
        
        self.assertEqual(risk_score, 0.0)
        self.assertIn("error", details)

    @patch('openai.AsyncOpenAI')
    def test_analyze_batch(self, mock_client):
        """Test batch processing of multiple texts"""
        mock_client.return_value.chat.completions.create = AsyncMock(return_value=self.mock_completion)
        
        detector = LLMDetector(api_key="sk-test")
        texts = [
            "First text with location",
            "Second text with phone number",
            "Third clean text"
        ]
        
        results = asyncio.run(detector.analyze_batch(texts))
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0][0], 0.85)
        self.assertEqual(results[1][0], 0.85)  # All mock responses are same now

    @patch('openai.AsyncOpenAI')
    def test_invalid_json_response(self, mock_client):
        """Test handling of malformed LLM response"""
        bad_message = MagicMock()
        bad_message.content = "```json\n" + json.dumps(SAMPLE_RESPONSE) + "\n```"
        bad_choice = MagicMock()
        bad_choice.message = bad_message
        bad_completion = MagicMock()
        bad_completion.choices = [bad_choice]
        mock_client.return_value.chat.completions.create = AsyncMock(return_value=bad_completion)
        
        detector = LLMDetector(api_key="sk-test")
        risk_score, details = detector.analyze_text("Sample text")
        
        self.assertEqual(risk_score, 0.85)
        self.assertEqual(details['details'], SAMPLE_RESPONSE['details'])

if __name__ == '__main__':
    unittest.main()
