import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from reddacted.llm_detector import LLMDetector

import asyncio
from typing import Dict, Any, List

SAMPLE_RESPONSE = {
    "has_pii": True,
    "confidence": 0.85,
    "details": ["Mentions specific location 'Miami Springs'"],
    "reasoning": "Location mention could help identify author's residence",
    "risk_factors": ["geographical specificity", "local slang reference"]
}

TEST_CASES = [
    {
        "text": "My phone number is 555-0123",
        "response": {
            "has_pii": True,
            "confidence": 0.95,
            "details": ["Contains phone number"],
            "risk_factors": ["contact_info"],
            "reasoning": "Phone number present"
        }
    },
    {
        "text": "I live at 123 Main St, Springfield",
        "response": {
            "has_pii": True,
            "confidence": 0.90,
            "details": ["Contains address"],
            "risk_factors": ["location"],
            "reasoning": "Street address present"
        }
    },
    {
        "text": "Just a regular comment about cats",
        "response": {
            "has_pii": False,
            "confidence": 0.1,
            "details": [],
            "risk_factors": [],
            "reasoning": "No PII detected"
        }
    }
]

@pytest.fixture
def mock_responses() -> List[Dict[str, Any]]:
    """Fixture providing a list of test responses"""
    return [case["response"] for case in TEST_CASES]

@pytest.fixture
def mock_texts() -> List[str]:
    """Fixture providing a list of test texts"""
    return [case["text"] for case in TEST_CASES]

@pytest.fixture
def mock_api_error():
    """Fixture providing a mock API error"""
    return Exception("API Error: Rate limit exceeded")

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
    """Test suite for LLMDetector class"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup method run before each test"""
        self.detector = LLMDetector(api_key="sk-test")
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
        
        risk_score, details = await self.detector.analyze_text("Sample text")
        
        assert risk_score == 0.0
        assert "error" in details
        assert "Invalid API key" in details["error"]

    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, mock_openai, mock_api_error):
        """Test handling of rate limit errors"""
        mock_openai.side_effect = mock_api_error
        
        risk_score, details = await self.detector.analyze_text("Test text")
        
        assert risk_score == 0.0
        assert "error" in details
        assert "Rate limit" in details["error"]

    @pytest.mark.asyncio
    async def test_empty_text_handling(self, mock_openai):
        """Test handling of empty text input"""
        risk_score, details = await self.detector.analyze_text("")
        
        assert risk_score == 0.0
        assert "error" in details
        assert "Empty text provided" in details["error"]
        mock_openai.return_value.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_long_text_handling(self, mock_openai):
        """Test handling of very long text input"""
        # Create text that exceeds OpenAI's typical token limit
        long_text = "test " * 5000
        
        # Create a specific mock response for truncated text
        mock_completion = MagicMock()
        message = MagicMock()
        message.content = json.dumps({
            "has_pii": False,
            "confidence": 0.5,
            "details": [],
            "risk_factors": [],
            "truncated": True,
            "reasoning": "Text was truncated due to length"
        })
        choice = MagicMock()
        choice.message = message
        mock_completion.choices = [choice]
        
        mock_openai.return_value.chat.completions.create = AsyncMock(return_value=mock_completion)
        
        risk_score, details = await self.detector.analyze_text(long_text)
        
        assert isinstance(risk_score, float)
        assert details.get("truncated") is True
        assert "Text was truncated" in details.get("reasoning", "")

    @pytest.mark.asyncio
    async def test_batch_concurrent_processing(self, mock_openai, mock_responses, mock_texts):
        """Test concurrent processing of batch texts"""
        mock_completions = []
        for response in mock_responses:
            completion = MagicMock()
            message = MagicMock()
            message.content = json.dumps(response)
            choice = MagicMock()
            choice.message = message
            completion.choices = [choice]
            mock_completions.append(completion)

        mock_openai.return_value.chat.completions.create = AsyncMock(side_effect=mock_completions)
        
        results = await self.detector.analyze_batch(mock_texts)
        
        assert len(results) == len(mock_texts)
        assert all(isinstance(score, float) for score, _ in results)
        assert all(isinstance(detail, dict) for _, detail in results)

    @pytest.mark.asyncio
    async def test_batch_error_handling(self, mock_openai, mock_texts, mock_api_error):
        """Test error handling in batch processing"""
        mock_openai.return_value.chat.completions.create = AsyncMock(side_effect=mock_api_error)
        
        results = await self.detector.analyze_batch(mock_texts)
        
        assert len(results) == len(mock_texts)
        assert all(score == 0.0 for score, _ in results)
        assert all("error" in detail for _, detail in results)

    @pytest.mark.asyncio
    async def test_analyze_batch(self, mock_openai):
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
        mock_openai.return_value.chat.completions.create = AsyncMock(side_effect=mock_completion)

        detector = LLMDetector(api_key="sk-test")
        texts = [
            "123 Main St, New York",
            "Call me at 555-0123",
            "Just a regular text"
        ]

        results = await detector.analyze_batch(texts)
        
        # Verify number of API calls
        assert len(results) == 3
        assert len(mock_completion.call_count) == 3
        
        # Verify individual results
        assert results[0][0] == 0.9  # Location text
        assert results[0][1]['risk_factors'] == ["location"]
        
        assert results[1][0] == 0.8  # Phone number text
        assert results[1][1]['risk_factors'] == ["contact"]
        
        assert results[2][0] == 0.0  # Clean text
        assert results[2][1]['risk_factors'] == []

        # Verify API was called with correct parameters
        mock_openai.assert_called_once_with(
            api_key="sk-test",
            default_headers={}
        )

    @pytest.mark.asyncio
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

