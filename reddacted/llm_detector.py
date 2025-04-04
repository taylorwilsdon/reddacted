import json
import asyncio
from typing import Tuple, Dict, Any, List, Optional
import openai
from reddacted.utils.logging import get_logger, with_logging
from reddacted.utils.exceptions import handle_exception

logger = get_logger(__name__)


@with_logging(logger)
class LLMDetector:
    """Uses LLM to detect potential PII and personal information in text,
    and can suggest sarcastic replacements."""

    DEFAULT_PROMPT = """
    Analyze the following text for any information that could potentially identify the author or reveal personal details about them.
    Consider both explicit PII (like names, addresses) and implicit personal information (like specific life events, locations, relationships).

   YOU MUST Respond in JSON format with these fields. DO NOT CHANGE FIELD NAMES, THEY ARE VERY IMPORTANT.
    - has_pii: boolean
    - confidence: float (0-1)
    - details: list of findings with type and example from the comment text
    - reasoning: detailed explanation of why this content might identify the author
    - risk_factors: list of specific elements that contribute to the risk score

    Text to analyze: {text}
    """

    REPLACEMENT_PROMPT_TEMPLATE = """
    You are a creative writing assistant specializing in sarcastic and nonsensical rewrites.
    Your task is to rewrite the following text, replacing any identified personal information with humorous, absurd, or sarcastic placeholders. Maintain the original structure and tone as much as possible, but ensure all sensitive details are obscured.

    Original Text:
    "{original_text}"

    Identified Personal Information Details:
    {pii_details}

    Rewrite the text, replacing the identified information with sarcastic/nonsensical content.
    ONLY output the rewritten text. Do not include explanations, apologies, or any text other than the rewritten version.
    """

    def __init__(
        self, api_key: str, api_base: str = None, model: str = "gpt-3.5-turbo", headers: dict = None
    ):
        self.model = model
        self.client_config = {
            "api_key": api_key,
        }
        if headers:
            self.client_config["default_headers"] = headers
        if api_base:
            self.client_config["base_url"] = api_base

    async def analyze_batch(self, texts: List[str]) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Analyze a batch of texts using LLM for potential personal information.
        Returns list of tuples (risk_score, details).
        """
        batch_size = 10
        results = []
        try:
            client = openai.AsyncOpenAI(**self.client_config)
        except openai.AuthenticationError as e:
            error_msg = str(e)
            if "Incorrect API key provided" in error_msg:
                # Extract the redacted key if present
                key_preview = (
                    error_msg.split("provided: ")[1].split(".")[0]
                    if "provided: " in error_msg
                    else "UNKNOWN"
                )
                raise ValueError(f"Invalid API key (provided: {key_preview})") from e
            raise ValueError("Authentication failed - please check your API key") from e
        except openai.APIError as e:
            raise ConnectionError(f"API error: {e.message}") from e

        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                tasks = []
                for text in batch:
                    task = client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a privacy analysis assistant."},
                            {"role": "user", "content": self.DEFAULT_PROMPT.format(text=text)},
                        ],
                        temperature=0.1,
                    )
                    logger.debug_with_context(f"Using API base: {client.base_url}")
                    logger.debug_with_context(f"Using model: {self.model}")
                    tasks.append(task)

                logger.info_with_context(f"Awaiting {len(tasks)} LLM analysis tasks...") # Added log
                batch_responses = await asyncio.gather(*tasks)
                logger.info_with_context("LLM analysis tasks completed.") # Added log

                for response in batch_responses:
                    try:
                        raw_response = response.choices[0].message.content.strip()
                        logger.debug_with_context(f"\n🤖 Raw LLM Response:\n{raw_response}\n")
                        try:
                            # First attempt direct parse, sometimes stupid LLM messes up formatting
                            analysis = json.loads(raw_response)
                        except json.JSONDecodeError:
                            # If that fails, try to extract JSON from markdown blocks
                            if "```json" in raw_response:
                                json_content = (
                                    raw_response.split("```json")[1].split("```")[0].strip()
                                )
                                analysis = json.loads(json_content)
                            else:
                                raise

                        # Calculate risk score based on confidence and PII presence
                        confidence = float(analysis.get("confidence", 0.0))
                        has_pii = analysis.get("has_pii", False)

                        logger.debug_with_context(f"Parsed confidence: {confidence}")
                        logger.debug_with_context(f"Parsed has_pii: {has_pii}")

                        if has_pii:
                            risk_score = confidence
                        else:
                            risk_score = 0.0
                            analysis = {
                                "has_pii": False,
                                "confidence": 0.0,
                                "details": [],
                                "risk_factors": [],
                                "reasoning": "No PII detected",
                            }

                        results.append((risk_score, analysis))
                    except Exception as e:
                        logger.warning_with_context(f"Failed to parse LLM analysis response: {e}")
                        results.append((0.0, {"error": f"LLM response parsing failed: {e}"}))
            return results

        except Exception as e:
            logger.error_with_context("AI analysis failed")
            logger.error_with_context(f"Batch LLM analysis failed: {str(e)}")
            error_msg = str(e)
            if isinstance(e, ValueError) and "Invalid API key" in error_msg:
                # Format a user-friendly error message
                return [
                    (
                        0.0,
                        {
                            "error": "Authentication Failed",
                            "details": error_msg,
                            "help": "Please check your OpenAI API key configuration",
                        },
                    )
                ] * len(texts)
            return [
                (
                    0.0,
                    {
                        "error": "LLM Analysis Failed",
                        "details": error_msg,
                        "help": "Please try again or contact support if the issue persists",
                    },
                )
            ] * len(texts)

    async def analyze_text(self, text: str) -> Tuple[float, Dict[str, Any]]:
        """
        Analyze a single text using LLM for potential personal information.
        Returns tuple of (risk_score, details).
        """
        try:
            results = await self.analyze_batch([text])
            return results[0]
        except Exception as e:
            logger.error_with_context(f"LLM analysis failed: {str(e)}")
            return 0.0, {"error": str(e)}

    async def suggest_replacement(self, text: str, analysis: Dict[str, Any]) -> Optional[str]:
        """
        Suggests a sarcastic/nonsensical replacement for the text, obscuring PII.

        Args:
            text: The original text.
            analysis: The analysis result dictionary from analyze_text/analyze_batch.

        Returns:
            The suggested replacement text, or None if no PII was found or an error occurred.
        """
        if not analysis or not analysis.get("has_pii"):
            logger.info_with_context("No PII found, skipping replacement suggestion.")
            return None

        pii_details_list = analysis.get("details", [])
        if not pii_details_list:
             logger.warning_with_context("has_pii is True, but no details found. Skipping replacement.")
             return None

        # Format PII details for the prompt
        pii_details_str = "\n".join([f"- Type: {item.get('type', 'N/A')}, Example: {item.get('example', 'N/A')}" for item in pii_details_list])

        prompt = self.REPLACEMENT_PROMPT_TEMPLATE.format(
            original_text=text,
            pii_details=pii_details_str
        )

        try:
            # Create a client instance for this specific call
            client = openai.AsyncOpenAI(**self.client_config)
            logger.debug_with_context("Requesting replacement suggestion from LLM.")
            logger.debug_with_context(f"Replacement Prompt:\n{prompt}")

            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a creative writing assistant specializing in sarcastic and nonsensical rewrites."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7, # Slightly higher temp for creativity
            )

            replacement_text = response.choices[0].message.content.strip()
            logger.debug_with_context(f"Raw replacement suggestion:\n{replacement_text}")

            # Basic check to ensure it's not empty or just whitespace
            if not replacement_text:
                 logger.warning_with_context("LLM returned an empty replacement suggestion.")
                 return None

            return replacement_text

        except openai.AuthenticationError as e:
            error_msg = str(e)
            key_preview = "UNKNOWN"
            if "Incorrect API key provided" in error_msg and "provided: " in error_msg:
                 key_preview = error_msg.split("provided: ")[1].split(".")[0]
            logger.error_with_context(f"Authentication failed for replacement suggestion (key: {key_preview}): {e}")
            # Propagate a clear error message or handle as needed downstream
            # For now, returning None as the function signature suggests optional return
            return None
        except openai.APIError as e:
            logger.error_with_context(f"API error during replacement suggestion: {e.message}")
            return None
        except Exception as e:
            logger.error_with_context(f"Unexpected error during replacement suggestion: {str(e)}")
            return None
