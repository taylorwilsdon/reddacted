#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import asyncio
import logging
import re
from os import environ
from typing import List, Dict, Any, Optional, Tuple

# Third-party
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from rich.panel import Panel
from rich.columns import Columns
from rich.console import Group
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

# Local
from reddacted.utils.logging import get_logger, with_logging
from reddacted.utils.exceptions import handle_exception
from reddacted.api.scraper import Scraper
from reddacted.api.reddit import Reddit
from reddacted.pii_detector import PIIDetector
from reddacted.llm_detector import LLMDetector
from reddacted.results import ResultsFormatter, AnalysisResult

logger = get_logger(__name__)

_COMMENT_ANALYSIS_HEADERS = {
    'User-agent': "reddacted"
}


# Sentiment constants
HAPPY_SENTIMENT = "😁"
SAD_SENTIMENT = "😕"
NEUTRAL_SENTIMENT = "😐"


class Sentiment():
    """Performs the LLM PII & sentiment analysis on a given set of Reddit Objects."""
    def __init__(self, auth_enabled=False, pii_enabled=True, llm_config=None, pii_only=False, sort='New', limit=100, skip_text=None):
        """Initialize Sentiment Analysis with optional PII detection

        Args:
            auth_enabled (bool): Enable Reddit API authentication
            pii_enabled (bool): Enable PII detection
            llm_config (dict): Configuration for LLM-based analysis
            pii_only (bool): Only show comments with PII detected
            debug (bool): Enable debug logging
            limit (int): Maximum number of comments to analyze
            skip_text (str): Text pattern to skip during analysis
        """
        # Set up logging
        logger.debug_with_context("Initializing Sentiment Analyzer")

        # Download required NLTK data if not already present
        try:
            nltk.data.find('sentiment/vader_lexicon')
        except LookupError:
            logger.info("Downloading required NLTK data...")
            nltk.download('vader_lexicon', quiet=True)

        # Initialize necessary variables
        self.skip_text = skip_text
        self.llm_detector = None  # Initialize llm_detector early
        try:
            self.api = Scraper()
            self.score = 0
            self.sentiment = NEUTRAL_SENTIMENT
            self.headers = _COMMENT_ANALYSIS_HEADERS
            self.auth_enabled = auth_enabled
            self.pii_enabled = pii_enabled
            self.pii_detector = PIIDetector() if pii_enabled else None
            self.pii_only = pii_only
            self.sort = sort
            self.limit = limit
            logger.debug_with_context("Initialized with configuration: "
                                    f"pii_enabled={pii_enabled}, "
                                    f"pii_only={pii_only}, "
                                    f"sort={sort}, "
                                    f"limit={limit}")

            logger.debug_with_context("Sentiment analyzer initialized")
        except Exception as e:
            handle_exception(e, "Failed to initialize Sentiment analyzer")
            logger.error_with_context("Failed to initialize Sentiment analyzer")
            raise
        # Initialize LLM detector if config provided
        if llm_config and pii_enabled:
            logger.debug_with_context("Initializing LLM Detector")
            self.llm_detector = LLMDetector(
                api_key=llm_config.get('api_key'),
                api_base=llm_config.get('api_base'),
                model=llm_config.get('model', 'gpt-4o-mini')
            )
            logger.debug_with_context("LLM Detector initialized")
        else:
            logger.debug_with_context("LLM Detector not initialized (llm_config not provided or PII detection disabled)")

        if auth_enabled:
            logger.debug_with_context("Authentication enabled, initializing Reddit API")
            self.api = Reddit()
            logger.debug_with_context("Reddit API initialized")
        else:
            logger.debug_with_context("Authentication not enabled")
        self.formatter = ResultsFormatter()
        self.formatter.print_config(auth_enabled, pii_enabled, llm_config, 
                                  self.pii_only, self.limit, self.sort)


    @with_logging(logger)
    async def _analyze(self, comments):
        """Analyzes comments for both sentiment and PII content.
        :param comments: comments to perform analysis on.
        :return: tuple of (sentiment_score, list of AnalysisResult objects)
        """
        logger.debug_with_context("Starting _analyze function")
        sentiment_analyzer = SentimentIntensityAnalyzer()
        final_score = 0
        results = []
        cleanup_regex = re.compile('<.*?>')
        total_comments = len(comments)
        progress = Progress(
            SpinnerColumn(spinner_name="dots"),
            TextColumn("[bold blue]{task.description}"),
            TimeElapsedColumn(),
            transient=True
        )
        with progress:
            main_task = progress.add_task(f"Received {total_comments} comments, processing...", total=total_comments)
            pii_task = progress.add_task("🔍 PII Analysis", visible=False, total=1)
            llm_task = progress.add_task("🤖 LLM Analysis", visible=False, total=1)
            for i, comment_data in enumerate(comments, 1):
                try:
                    clean_comment = re.sub(cleanup_regex, '', str(comment_data['text']))
                    
                    # Skip already reddacted comments
                    if self.skip_text and self.skip_text in clean_comment:
                        logger.debug_with_context(f"Skipping already reddacted comment {i}")
                        progress.update(main_task, advance=1)
                        continue
                    progress.update(
                        main_task,
                        description=f"[bold blue]💭 Processing comment[/] [cyan]{i}[/]/[cyan]{total_comments}[/]"
                    )
                    # Sentiment analysis
                    all_scores = sentiment_analyzer.polarity_scores(clean_comment)
                    score = all_scores['compound']
                    final_score += score
                    # PII analysis
                    pii_risk_score, pii_matches = 0.0, []
                    if self.pii_enabled:
                        progress.update(pii_task, visible=True)
                        progress.update(pii_task, description=f"🔍 Scanning comment {i} for PII")
                        pii_risk_score, pii_matches = self.pii_detector.get_pii_risk_score(clean_comment)
                        progress.update(pii_task, visible=False)
                        # Store comment for batch processing
                        if not hasattr(self, '_llm_batch'):
                            self._llm_batch = []
                            self._llm_batch_indices = []
                            self._pending_results = []
                        self._llm_batch.append(clean_comment)
                        self._llm_batch_indices.append(len(self._pending_results))
                        # Create result with combined risk score
                        result = AnalysisResult(
                            comment_id=comment_data['id'],
                            sentiment_score=score,
                            sentiment_emoji=self._get_sentiment(score),
                            pii_risk_score=pii_risk_score,  # Initial PII score
                            pii_matches=pii_matches,
                            text=clean_comment,
                            upvotes=comment_data['upvotes'],
                            downvotes=comment_data['downvotes'],
                            permalink=comment_data['permalink'],
                            llm_risk_score=0.0,
                            llm_findings=None
                        )
                        self._pending_results.append(result)
                        # Process batch when full or at end
                        if len(self._llm_batch) >= 10 or i == total_comments:
                            batch_size = len(self._llm_batch)
                            progress.update(llm_task, visible=True)
                            progress.update(
                                llm_task,
                                description=f"[bold blue]🤖 Processing LLM batch[/] ([cyan]{batch_size}[/] items)"
                            )
                            batch_results = await self.llm_detector.analyze_batch(self._llm_batch)
                            progress.update(
                                llm_task,
                                description=f"[bold green]✅ LLM batch complete[/] ([cyan]{batch_size}[/] items analyzed)"
                            )
                            progress.update(llm_task, visible=False)
                            # Update pending results with batch results
                            for batch_idx, (risk_score, findings) in zip(self._llm_batch_indices, batch_results):
                                result = self._pending_results[batch_idx]
                                # Always set LLM results regardless of PII detection
                                result.llm_risk_score = risk_score
                                result.llm_findings = findings
                                # Update PII risk score if LLM found PII
                                if findings and findings.get('has_pii'):
                                    result.pii_risk_score = max(result.pii_risk_score, risk_score)
                                # Add this result to final results immediately
                                results.append(result)
                                logger.debug_with_context(f"Added result to {i} final results")
                            # Clear batch
                            self._llm_batch = []
                            self._llm_batch_indices = []
                            self._pending_results = []
                    # Only append results directly if not using LLM
                    if not self.llm_detector:
                        results.append(AnalysisResult(
                            comment_id=comment_data['id'],
                            sentiment_score=score,
                            sentiment_emoji=self._get_sentiment(score),
                            pii_risk_score=pii_risk_score,
                            pii_matches=pii_matches,
                            text=clean_comment,
                            upvotes=comment_data['upvotes'],
                            downvotes=comment_data['downvotes'],
                            permalink=comment_data['permalink'],
                            llm_risk_score=0.0,
                            llm_findings=None
                        ))
                    progress.update(main_task, advance=1)
                except Exception as e:
                    logger.error_with_context(f"Error processing comment {i}: {e}")
                    continue
            try:
                rounded_final = round(final_score/len(comments), 4)
                logger.debug_with_context(f"Final sentiment score calculated: {rounded_final}")
                return rounded_final, results
            except ZeroDivisionError:
                logger.error_with_context("No comments found")
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
        fetch_method = {
            'user': self.api.parse_user,
            'listing': self.api.parse_listing
        }[source_type]

        # Handle text search if specified
        if text_match := kwargs.pop('text_match', None):
            if source_type == 'user':
                # For users, we pass the text_match to parse_user
                return fetch_method(
                    identifier,
                    headers=self.headers,
                    limit=self.limit,
                    text_match=text_match,
                    **kwargs
                )
            else:
                # For subreddits, use search_comments
                return self.api.search_comments(
                    query=text_match,
                    subreddit=kwargs.get('subreddit'),
                    limit=self.limit
                )
            
        # Default comment fetching
        if source_type == 'listing':
            # Split subreddit/article for listing type
            subreddit = identifier.split('/')[0]
            article = identifier.split('/')[1]
            return fetch_method(
                subreddit,
                article,
                headers=self.headers,
                limit=self.limit,
                **kwargs
            )
        else:
            return fetch_method(
                identifier,
                headers=self.headers,
                limit=self.limit,
                **kwargs
            )

    @with_logging(logger)
    def _run_analysis_flow(self, comments: List[Dict[str, Any]]) -> Tuple[float, List[AnalysisResult]]:
        """Centralized analysis execution"""
        logger.debug_with_context("Running analysis flow")
        if asyncio.get_event_loop().is_running():
            future = asyncio.ensure_future(self._analyze(comments))
            return asyncio.get_event_loop().run_until_complete(future)
        return asyncio.run(self._analyze(comments))

    @with_logging(logger)
    def get_sentiment(self, source_type: str, identifier: str, output_file: Optional[str] = None, **kwargs) -> None:
        """Unified sentiment analysis entry point"""
        logger.debug_with_context(f"get_sentiment called with source_type={source_type}, identifier={identifier}")
        comments = self._get_comments(source_type, identifier, **kwargs)
        self.score, self.results = self._run_analysis_flow(comments)
        self.sentiment = self._get_sentiment(self.score)
        if output_file:
            self.formatter.generate_output_file(output_file, comments, identifier, 
                                             self.results, self.score, self.sentiment)
        else:
            self.formatter.print_comments(comments, identifier, self.results, 
                                        self.score, self.sentiment)
