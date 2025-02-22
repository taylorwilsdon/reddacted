import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from reddacted.llm_detector import LLMDetector

SAMPLE_RESPONSE = {
    "has_pii": True,
    "confidence": 0.85,
    "details": ["Mentions specific location 'Miami Springs'"],
    "reasoning": "Location mention could help identify author's residence",
    "risk_factors": ["geographical specificity", "local slang reference"]
}

@pytest.fixture
def mock_openai():
    """Fixture to provide mocked OpenAI client"""
    with patch('openai.AsyncOpenAI') as mock:
        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock.return_value = mock_client
        yield mock

@pytest.fixture
def mock_completion():
    """Fixture to provide mocked completion response"""
    completion = MagicMock()
    message = MagicMock()
    message.content = json.dumps(SAMPLE_RESPONSE)
    choice = MagicMock()
    choice.message = message
    completion.choices = [choice]
    return completion

class TestLLMDetector:
    @pytest.mark.asyncio
    async def test_analyze_text_success(self, mock_openai, mock_completion):
        """Test successful PII analysis with valid response"""
        mock_openai.return_value.chat.completions.create = AsyncMock(return_value=mock_completion)
        
        detector = LLMDetector(api_key="sk-test")
        risk_score, details = await detector.analyze_text("RaunchyRaccoon that looks a lot like Miami Springs!")
        
        assert risk_score == 0.85
        assert details['details'] == SAMPLE_RESPONSE['details']
        assert details['risk_factors'] == SAMPLE_RESPONSE['risk_factors']
        mock_openai.assert_called_once_with(
            api_key="sk-test",
            default_headers={}
        )

    @pytest.mark.asyncio
    async def test_analyze_invalid_key(self, mock_openai):
        """Test authentication error handling"""
        mock_openai.side_effect = Exception("Invalid API key")
        
        detector = LLMDetector(api_key="invalid-key")
        risk_score, details = await detector.analyze_text("Sample text")
        
        assert risk_score == 0.0
        assert "error" in details

    @patch('openai.AsyncOpenAI')
    def test_analyze_batch(self, mock_client):
        """Test batch processing of multiple texts"""
        # Configure different mock responses for each text
        responses = [
            {"has_pii": True, "confidence": 0.9, "details": ["Contains location"], "risk_factors": ["location"]},
            {"has_pii": True, "confidence": 0.8, "details": ["Contains phone number"], "risk_factors": ["contact"]},
            {"has_pii": False, "confidence": 0.0, "details": [], "risk_factors": []}
        ]

        async def mock_completion(*args, **kwargs):
            # Get the input text from the API call
            messages = kwargs.get('messages', [])
            text_index = len(mock_completion.call_count)
            mock_completion.call_count.append(1)  # Track number of calls
            
            # Create mock response
            mock_msg = MagicMock()
            mock_msg.content = json.dumps(responses[text_index])
            mock_choice = MagicMock()
            mock_choice.message = mock_msg
            mock_resp = MagicMock()
            mock_resp.choices = [mock_choice]
            return mock_resp

        # Initialize call counter
        mock_completion.call_count = []
        mock_client.return_value.chat.completions.create = AsyncMock(side_effect=mock_completion)

        detector = LLMDetector(api_key="sk-test")
        texts = [
            "123 Main St, New York",
            "Call me at 555-0123",
            "Just a regular text"
        ]

        results = asyncio.run(detector.analyze_batch(texts))
        
        # Verify number of API calls
        self.assertEqual(len(results), 3)
        self.assertEqual(len(mock_completion.call_count), 3)
        
        # Verify individual results
        self.assertEqual(results[0][0], 0.9)  # Location text
        self.assertEqual(results[0][1]['risk_factors'], ["location"])
        
        self.assertEqual(results[1][0], 0.8)  # Phone number text
        self.assertEqual(results[1][1]['risk_factors'], ["contact"])
        
        self.assertEqual(results[2][0], 0.0)  # Clean text
        self.assertEqual(results[2][1]['risk_factors'], [])

        # Verify API was called with correct parameters
        mock_client.assert_called_once_with(
            api_key="sk-test",
            default_headers={}
        )

    @patch('openai.AsyncOpenAI')
    async def test_invalid_json_response(self, mock_client):
        """Test handling of malformed LLM response"""
        bad_message = MagicMock()
        bad_message.content = "```json\n" + json.dumps(SAMPLE_RESPONSE) + "\n```"
        bad_choice = MagicMock()
        bad_choice.message = bad_message
        bad_completion = MagicMock()
        bad_completion.choices = [bad_choice]
        mock_client.return_value.chat.completions.create = AsyncMock(return_value=bad_completion)

        detector = LLMDetector(api_key="sk-test")
        risk_score, details = await detector.analyze_text("Sample text")

        self.assertEqual(risk_score, 0.85)
        self.assertEqual(details['details'], SAMPLE_RESPONSE['details'])

if __name__ == '__main__':
    unittest.main()
