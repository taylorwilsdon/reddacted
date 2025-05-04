#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import asyncio
import re
from os import environ
from typing import List, Dict, Any, Optional, Tuple

# Third-party
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.console import Console

# Local
from reddacted.utils.log_handler import get_logger, with_logging

# Initialize rich console
console = Console()
from reddacted.utils.log_handler import handle_exception
from reddacted.api.scraper import Scraper
from reddacted.api.reddit import Reddit
from reddacted.pii_detector import PIIDetector
from reddacted.llm_detector import LLMDetector
from reddacted.results import ResultsFormatter, AnalysisResult

logger = get_logger(__name__)

_COMMENT_ANALYSIS_HEADERS = {"User-agent": "reddacted"}


# Sentiment constants
HAPPY_SENTIMENT = "😁"
SAD_SENTIMENT = "😕"
NEUTRAL_SENTIMENT = "😐"


class Sentiment:
    """Performs the LLM PII & sentiment analysis on a given set of Reddit Objects."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Sentiment Analysis using a configuration dictionary.

        Args:
            config (Dict[str, Any]): Dictionary containing all configuration settings.
        """
        logger.debug_with_context(f"Initializing Sentiment Analyzer with config: {config}")

        # Extract values from config, providing defaults
        self.auth_enabled = config.get("enable_auth", False)
        self.pii_enabled = True # Assuming PII is always enabled for now, adjust if needed
        self.llm_config = None # Will be constructed below if needed
        self.pii_only = config.get("pii_only", False)
        self.sort = config.get("sort", "new")
        # Default limit to 20 if missing, map 0 to None (unlimited)
        limit_val = config.get("limit", 20)
        self.limit = None if limit_val == 0 else limit_val
        self.skip_text = config.get("skip_text")
        self.use_random_string = config.get("use_random_string", False)

        # Download required NLTK data if not already present
        try:
            nltk.data.find("sentiment/vader_lexicon")
        except LookupError:
            logger.debug("Downloading required NLTK data...")
            nltk.download("vader_lexicon", quiet=True)

        # Initialize necessary variables
        self.llm_detector = None  # Initialize llm_detector early
        # Initialize batch processing attributes
        self._llm_batch = []
        self._llm_batch_indices = []
        self._pending_results = []

        try:
            self.api = Scraper() # Default to Scraper
            self.score = 0
            self.sentiment = NEUTRAL_SENTIMENT
            self.headers = _COMMENT_ANALYSIS_HEADERS
            self.pii_detector = PIIDetector() if self.pii_enabled else None
            logger.debug_with_context(
                "Initialized base attributes with configuration: "
                f"auth_enabled={self.auth_enabled}, "
                f"pii_enabled={self.pii_enabled}, "
                f"pii_only={self.pii_only}, "
                f"sort={self.sort}, "
                f"limit={self.limit}"
            )

            logger.debug_with_context("Base sentiment analyzer initialized")
        except Exception as e:
            handle_exception(e, "Failed to initialize base Sentiment analyzer")
            logger.error_with_context("Failed to initialize base Sentiment analyzer")
            raise

        # Construct LLM config dictionary if applicable
        if config.get("model"):
            # If model is specified but no LLM URL, default to local
            if not config.get("local_llm") and not config.get("openai_key"):
                config["local_llm"] = "http://localhost:11434" # Modify config directly or use a local var
                logger.warning_with_context("No LLM URL specified, defaulting to local")

            self.llm_config = {
                "api_key": config.get("openai_key") if config.get("use_openai_api") else "sk-not-needed",
                "api_base": config.get("local_llm") if not config.get("use_openai_api") else "https://api.openai.com/v1",
                "model": config.get("model"),
            }
            # Adjust api_base for local LLM if needed
            if not config.get("use_openai_api") and self.llm_config["api_base"]:
                base_url = self.llm_config["api_base"].rstrip('/')
                if not base_url.endswith('/v1'):
                    self.llm_config["api_base"] = f"{base_url}/v1"
        elif config.get("openai_key") or config.get("local_llm"):
             # Handle case where URL/key is provided but no model selected yet
             self.llm_config = {
                "api_key": config.get("openai_key") if config.get("use_openai_api") else "sk-not-needed",
                "api_base": config.get("local_llm") if not config.get("use_openai_api") else "https://api.openai.com/v1",
                "model": None, # Explicitly set model to None
            }
             if not config.get("use_openai_api") and self.llm_config["api_base"]:
                base_url = self.llm_config["api_base"].rstrip('/')
                if not base_url.endswith('/v1'):
                    self.llm_config["api_base"] = f"{base_url}/v1"
             logger.warning_with_context("LLM URL/Key provided, but no model selected. LLM analysis may be limited.")


        # Initialize LLM detector if config was constructed
        if self.llm_config:
            logger.debug_with_context(f"Constructed LLM Config: {self.llm_config}")
            try:
                api_key = self.llm_config.get("api_key")
                api_base = self.llm_config.get("api_base")
                model = self.llm_config.get("model") # Already extracted

                logger.debug_with_context(f"LLM Config - API Base: {api_base}, Model: {model}")
                # Initialize LLM detector if we have sufficient configuration
                if not model:
                    logger.warning_with_context("No model specified in config - LLM analysis disabled")
                    self.llm_detector = None
                elif not api_base:
                    logger.error_with_context("Missing API base URL - required for both local and OpenAI")
                    self.llm_detector = None
                elif api_base == "https://api.openai.com/v1" and not api_key:
                    logger.error_with_context("Missing API key - required for OpenAI API")
                    self.llm_detector = None
                else:
                    self.llm_detector = LLMDetector(
                        api_key=api_key,
                        api_base=api_base,
                        model=model,
                    )
                    logger.info_with_context("LLM Detector initialized")
            except Exception as e:
                logger.error_with_context(f"Failed to initialize LLM Detector: {str(e)}")
                self.llm_detector = None
        else:
            logger.info_with_context("No LLM config provided or model missing, LLM analysis disabled.")

        # Initialize Reddit API if auth is enabled, passing the config
        if self.auth_enabled:
            logger.debug_with_context("Authentication enabled, initializing Reddit API with config")
            # Pass the full config dictionary and the specific use_random_string preference
            self.api = Reddit(config=config, use_random_string=self.use_random_string)
            logger.debug_with_context("Reddit API initialized")
        else:
            logger.debug_with_context("Authentication not enabled, using default Scraper API")
            # self.api remains the Scraper instance initialized earlier

        self.formatter = ResultsFormatter()
        self.formatter.pii_only = self.pii_only
        self.formatter.use_random_string = self.use_random_string # Use instance variable
        # Pass the entire config dictionary to print_config
        self.formatter.print_config(config)

    @with_logging(logger)
    async def _analyze(self, comments):
        """Analyzes comments for both sentiment and PII content.
        :param comments: comments to perform analysis on.
        :return: tuple of (sentiment_score, list of AnalysisResult objects)
        """
        logger.debug_with_context("Starting _analyze function")
        sentiment_analyzer = SentimentIntensityAnalyzer()
        final_score = 0
        results: List[AnalysisResult] = [] # Final results list
        _llm_batch: List[str] = [] # Batch of comments for LLM
        _llm_result_indices: List[int] = [] # Indices in 'results' corresponding to _llm_batch items

        cleanup_regex = re.compile("<.*?>")
        total_comments = len(comments)
        progress = Progress(
            SpinnerColumn(spinner_name="dots"),
            TextColumn("[bold blue]{task.description}"),
            TimeElapsedColumn(),
            transient=True,
        )
        with progress:
            main_task = progress.add_task(
                f"Received {total_comments} comments, processing...", total=total_comments
            )
            pii_task = progress.add_task("🔍 PII Analysis", visible=False, total=1)
            llm_task = progress.add_task("🤖 LLM Analysis", visible=False, total=1)
            for i, comment_data in enumerate(comments, 1):
                try:
                    clean_comment = re.sub(cleanup_regex, "", str(comment_data["text"]))

                    # Skip already reddacted comments
                    if self.skip_text and self.skip_text in clean_comment:
                        logger.debug_with_context(f"Skipping already reddacted comment {i}")
                        progress.update(main_task, advance=1)
                        continue
                    progress.update(
                        main_task,
                        description=f"[bold blue]💭 Processing comment[/] [cyan]{i}[/]/[cyan]{total_comments}[/]",
                    )
                    # Sentiment analysis
                    all_scores = sentiment_analyzer.polarity_scores(clean_comment)
                    score = all_scores["compound"]
                    final_score += score
                    # PII analysis
                    pii_risk_score, pii_matches = 0.0, []
                    if self.pii_enabled:
                        progress.update(pii_task, visible=True)
                        progress.update(pii_task, description=f"🔍 Scanning comment {i} for PII")
                        pii_risk_score, pii_matches = self.pii_detector.get_pii_risk_score(
                            clean_comment
                        )
                        progress.update(pii_task, visible=False)
                        
                    # Create the initial result object
                    result = AnalysisResult(
                        comment_id=comment_data["id"],
                        sentiment_score=score,
                        sentiment_emoji=self._get_sentiment(score),
                        pii_risk_score=pii_risk_score,
                        pii_matches=pii_matches,
                        text=clean_comment,
                        upvotes=comment_data["upvotes"],
                        downvotes=comment_data["downvotes"],
                        permalink=comment_data["permalink"],
                        llm_risk_score=0.0, # Placeholder
                        llm_findings=None, # Placeholder
                    )
                    results.append(result) # Add initial result to final list

                    # If LLM is enabled, add to batch for later processing
                    if self.llm_detector:
                        _llm_batch.append(clean_comment)
                        _llm_result_indices.append(len(results) - 1) # Store index of the result we just added
                        logger.debug_with_context(f"Added comment {i} to LLM batch (size: {len(_llm_batch)})")

                        # Process batch if full
                        if len(_llm_batch) >= 10:
                            batch_size = len(_llm_batch)
                            try:
                                progress.update(llm_task, visible=True)
                                progress.update(llm_task, description=f"[bold blue]🤖 Processing LLM batch[/] ([cyan]{batch_size}[/] items)")
                                batch_llm_results = await self.llm_detector.analyze_batch(_llm_batch)
                                logger.debug_with_context(f"Successfully processed LLM batch of {batch_size} items")

                                # Update results in place
                                for result_idx, (llm_risk_score, findings) in zip(_llm_result_indices, batch_llm_results):
                                    results[result_idx].llm_risk_score = llm_risk_score
                                    results[result_idx].llm_findings = findings
                                    if findings and findings.get("has_pii"):
                                        results[result_idx].pii_risk_score = max(results[result_idx].pii_risk_score, llm_risk_score)
                                logger.debug_with_context(f"Updated {batch_size} results with LLM data")

                            except Exception as e:
                                logger.error_with_context(f"Failed to process LLM batch: {str(e)}")
                            finally:
                                progress.update(llm_task, description=f"[bold green]✅ LLM batch complete[/] ([cyan]{batch_size}[/] items analyzed)", visible=False)
                                # Clear batch lists for next batch
                                _llm_batch = []
                                _llm_result_indices = []
                    else:
                         logger.warning_with_context(f"Skipping LLM analysis for comment {i} - detector not initialized")

                    progress.update(main_task, advance=1)
                except Exception as e:
                    logger.error_with_context(f"Error processing comment {i}: {e}")
                    # Ensure progress advances even on error
                    progress.update(main_task, advance=1)
                    continue

            # --- Process any remaining items in the LLM batch after the loop ---
            if self.llm_detector and _llm_batch:
                batch_size = len(_llm_batch)
                try:
                    progress.update(llm_task, visible=True)
                    progress.update(llm_task, description=f"[bold blue]🤖 Processing final LLM batch[/] ([cyan]{batch_size}[/] items)")
                    batch_llm_results = await self.llm_detector.analyze_batch(_llm_batch)
                    logger.debug_with_context(f"Successfully processed final LLM batch of {batch_size} items")

                    # Update results in place
                    for result_idx, (llm_risk_score, findings) in zip(_llm_result_indices, batch_llm_results):
                        results[result_idx].llm_risk_score = llm_risk_score
                        results[result_idx].llm_findings = findings
                        if findings and findings.get("has_pii"):
                             results[result_idx].pii_risk_score = max(results[result_idx].pii_risk_score, llm_risk_score)
                    logger.debug_with_context(f"Updated {batch_size} results with final LLM data")

                except Exception as e:
                    logger.error_with_context(f"Failed to process final LLM batch: {str(e)}")
                finally:
                     progress.update(llm_task, description=f"[bold green]✅ Final LLM batch complete[/] ([cyan]{batch_size}[/] items analyzed)", visible=False)
                     # No need to clear batch lists here as they are local to the function call

            # --- Calculate final score and return ---
            try:
                # Use len(results) which accurately reflects processed comments
                num_processed = len(results)
                if num_processed == 0:
                     logger.warning("No comments were successfully processed.")
                     return 0.0, []
                # Calculate score based on processed comments' sentiment scores
                final_score = sum(r.sentiment_score for r in results) # Recalculate final_score based on actual results
                rounded_final = round(final_score / num_processed, 4) # Use num_processed
                logger.debug_with_context(f"Final sentiment score calculated: {rounded_final}")
                return rounded_final, results
            except ZeroDivisionError: # Should be caught by num_processed check, but keep for safety
                logger.error_with_context("Division by zero error during final score calculation.")
                return 0.0, []

    @with_logging(logger)
    def _get_sentiment(self, score):
        """Obtains the sentiment using a sentiment score.
        :param score: the sentiment score.
        :return: sentiment from score.
        """
        logger.debug_with_context(f"Calculating sentiment for score {score}")
        if score == 0:
            return NEUTRAL_SENTIMENT
        elif score > 0:
            return HAPPY_SENTIMENT
        else:
            return SAD_SENTIMENT

    @with_logging(logger)
    def _get_comments(self, source_type: str, identifier: str, **kwargs) -> List[Dict[str, Any]]:
        """Unified comment fetching method"""
        logger.debug_with_context(
            f"Fetching comments for {source_type} '{identifier}' with kwargs: {kwargs}"
        )

        # Get the appropriate fetch method
        fetch_method = {"user": self.api.parse_user, "listing": self.api.parse_listing}[source_type]

        # Handle text search if specified
        if text_match := kwargs.pop("text_match", None):
            if source_type == "user":
                # For users, we pass the text_match to parse_user
                return fetch_method(
                    identifier,
                    headers=self.headers,
                    limit=self.limit,
                    text_match=text_match,
                    **kwargs,
                )
            else:
                # For subreddits, use search_comments
                return self.api.search_comments(
                    query=text_match, subreddit=kwargs.get("subreddit"), limit=self.limit
                )

        # Default comment fetching
        if source_type == "listing":
            # Split subreddit/article for listing type
            subreddit = identifier.split("/")[0]
            article = identifier.split("/")[1]
            return fetch_method(
                subreddit, article, headers=self.headers, limit=self.limit, **kwargs
            )
        else:
            return fetch_method(identifier, headers=self.headers, limit=self.limit, **kwargs)

    @with_logging(logger)
    def _run_analysis_flow(
        self, comments: List[Dict[str, Any]]
    ) -> Tuple[float, List[AnalysisResult]]:
        """Centralized analysis execution"""
        logger.debug_with_context("Starting analysis flow")
        logger.debug_with_context(f"Processing {len(comments)} comments")
        logger.debug_with_context(f"LLM Detector status: {'Initialized' if self.llm_detector else 'Not initialized'}")
        
        try:
            loop = asyncio.get_running_loop()
            logger.debug_with_context("Using existing event loop")
            # If we have a running loop, use it
            future = asyncio.ensure_future(self._analyze(comments), loop=loop)
            result = loop.run_until_complete(future)
            logger.info_with_context("Analysis completed")
            return result
        except RuntimeError:
            # No running event loop, create a new one
            logger.debug_with_context("No running loop found, creating new one")
            return asyncio.run(self._analyze(comments))

    @with_logging(logger)
    def get_sentiment(
        self, source_type: str, identifier: str, output_file: Optional[str] = None, **kwargs
    ) -> None:
        """Unified sentiment analysis entry point"""
        logger.debug_with_context(
            f"get_sentiment called with source_type={source_type}, identifier={identifier}"
        )
        comments = self._get_comments(source_type, identifier, **kwargs)
        self.score, self.results = self._run_analysis_flow(comments)
        self.sentiment = self._get_sentiment(self.score)
        if output_file:
            self.formatter.generate_output_file(
                output_file, comments, identifier, self.results, self.score, self.sentiment
            )
        else:
            self.formatter.print_comments(
                comments, identifier, self.results, self.score, self.sentiment, self.api # Pass self.api
            )
