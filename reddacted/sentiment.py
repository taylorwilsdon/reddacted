#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import asyncio
import logging
import re
from dataclasses import dataclass
from os import environ
from typing import List, Dict, Any, Optional, Tuple, Union, ContextManager

# Third-party
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

logger = get_logger(__name__)

_COMMENT_ANALYSIS_HEADERS = {
    'User-agent': "reddacted"
}


@dataclass
class AnalysisResult:
    """Holds the results of both sentiment and PII analysis"""
    comment_id: str
    sentiment_score: float
    sentiment_emoji: str
    pii_risk_score: float
    pii_matches: List[Any]
    permalink: str
    text: str
    upvotes: int = 0
    downvotes: int = 0
    llm_risk_score: float = 0.0
    llm_findings: Optional[Dict[str, Any]] = None


# Sentiment constants
HAPPY_SENTIMENT = "😁"
SAD_SENTIMENT = "😕"
NEUTRAL_SENTIMENT = "😐"


class Sentiment():
    """Performs the LLM PII & sentiment analysis on a given set of Reddit Objects."""
    def __init__(self, auth_enabled=False, pii_enabled=True, llm_config=None, pii_only=False, sort='New', limit=100):
        """Initialize Sentiment Analysis with optional PII detection

        Args:
            auth_enabled (bool): Enable Reddit API authentication
            pii_enabled (bool): Enable PII detection
            llm_config (dict): Configuration for LLM-based analysis
            pii_only (bool): Only show comments with PII detected
            debug (bool): Enable debug logging
            limit (int): Maximum number of comments to analyze
        """
        # Set up logging
        logger.debug_with_context("Initializing Sentiment Analyzer")

        # Initialize necessary variables
        self.llm_detector = None  # Initialize llm_detector early
        try:
            self.api = Scraper()
            self.score = 0
            self.sentiment = NEUTRAL_SENTIMENT
            self.headers = _COMMENT_ANALYSIS_HEADERS
            self.authEnable = False
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
        self._print_config(auth_enabled, pii_enabled, llm_config)


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
                    progress.update(main_task, description=f"💭 Processing comment {i}/{total_comments}")
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
                            comment_id=str(i),
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
                            logger.debug_with_context(f"Processing LLM batch of {len(self._llm_batch)} items")
                            progress.update(llm_task, visible=True)
                            progress.update(llm_task, description="🤖 LLM analysis in progress...")
                            batch_results = await self.llm_detector.analyze_batch(self._llm_batch)
                            progress.update(llm_task, description="✅ LLM analysis complete")
                            logger.debug_with_context(f"LLM batch_results: {batch_results}")
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
                            comment_id=str(i),
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
    def _create_progress(self):
        """Unified progress context manager"""
        logger.debug_with_context("Creating progress context")
        return Progress(
            SpinnerColumn(spinner_name="dots"),
            TextColumn("[bold blue]{task.description}"),
            TimeElapsedColumn(),
            transient=True
        )


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
    def _generate_output_file(self, filename, comments, url):
        """Outputs a file containing a detailed sentiment and PII analysis per sentence."""
        logger.debug_with_context(f"Generating output file '{filename}' for URL '{url}'")
        # First get all results at once to show proper progress
        with Progress(
            SpinnerColumn(spinner_name="dots"),
            TextColumn("[bold blue]{task.description}"),
            TimeElapsedColumn(),
            transient=True
        ) as progress:
            progress_task = progress.add_task("📝 Generating analysis report...", total=len(comments))

            with open(filename, 'w+') as target:
                target.write(f"# Analysis Report for '{url}'\n\n")
                target.write(f"- **Overall Sentiment Score**: {self.score:.2f}\n")
                target.write(f"- **Overall Sentiment**: {self.sentiment}\n")
                target.write(f"- **Comments Analyzed**: {len(comments)}\n\n")
                target.write("---\n\n")
                # Initialize summary statistics
                total_pii_comments = 0
                total_llm_pii_comments = 0
                max_risk_score = 0.0
                riskiest_comment = None
                sentiment_scores = []
                def should_show_result(result):
                    if not self.pii_only:
                        return True
                    # Only show results with actual PII detections
                    has_pattern_pii = result.pii_risk_score > 0.0
                    has_llm_pii = (result.llm_findings and
                                  isinstance(result.llm_findings, dict) and
                                  result.llm_findings.get('has_pii', False) and
                                  result.llm_findings.get('confidence', 0.0) > 0.0)
                    return has_pattern_pii or has_llm_pii
                comment_count = 1
                for result in self.results:  # Use pre-computed results
                    progress.update(progress_task, description=f"📝 Writing comment {comment_count}/{len(comments)}")

                    if not should_show_result(result):
                        comment_count += 1
                        progress.update(progress_task, advance=1)
                        continue
                    target.write(f"## Comment {comment_count}\n\n")
                    target.write(f"**Text**: {result.text}\n\n")
                    target.write(f"- Sentiment Score: `{result.sentiment_score:.2f}` {result.sentiment_emoji}\n")
                    target.write(f"- PII Risk Score: `{result.pii_risk_score:.2f}`\n")
                    target.write(f"- Votes: ⬆️ `{result.upvotes}` ⬇️ `{result.downvotes}`\n")
                    # PII Matches Section
                    if result.pii_matches:
                        target.write("### Pattern-based PII Detected\n")
                        for pii in result.pii_matches:
                            target.write(f"- **{pii.type}** (confidence: {pii.confidence:.2f})\n")
                        target.write("\n")
                    # LLM Findings Section
                    if result.llm_findings:
                        target.write("### LLM Privacy Analysis\n")
                        target.write(f"- **Risk Score**: `{result.llm_risk_score:.2f}`\n")
                        if isinstance(result.llm_findings, dict):
                            target.write(f"- **PII Detected**: {'Yes' if result.llm_findings.get('has_pii') else 'No'}\n")
                            if details := result.llm_findings.get('details'):
                                target.write("\n#### Findings\n")
                                for detail in details:
                                    target.write(f"- {detail}\n")
                            if reasoning := result.llm_findings.get('reasoning'):
                                target.write(f"\n#### Reasoning\n{reasoning}\n")
                        target.write("\n")
                    target.write("---\n\n")
                    # Update summary stats
                    sentiment_scores.append(result.sentiment_score)
                    if result.pii_risk_score > 0:
                        total_pii_comments += 1
                    if result.llm_risk_score > 0:
                        total_llm_pii_comments += 1
                    if result.pii_risk_score > max_risk_score:
                        max_risk_score = result.pii_risk_score
                        riskiest_comment = result.text[:100] + "..." if len(result.text) > 100 else result.text
                    comment_count += 1
                    progress.update(progress_task, advance=1)
                # Add summary section to file
                target.write("\n# Summary\n\n")
                target.write(f"- Total Comments Analyzed: {len(comments)}\n")
                target.write(f"- Comments with PII Detected: {total_pii_comments} ({total_pii_comments/len(comments):.1%})\n")
                target.write(f"- Comments with LLM Privacy Risks: {total_llm_pii_comments} ({total_llm_pii_comments/len(comments):.1%})\n")
                target.write(f"- Average Sentiment Score: {sum(sentiment_scores)/len(sentiment_scores):.2f}\n")
                target.write(f"- Highest PII Risk Score: {max_risk_score:.2f}\n")
                if riskiest_comment:
                    target.write(f"- Riskiest Comment Preview: '{riskiest_comment}'\n")
                target.write("\n✅ Analysis complete\n")
            # Add console completion message
            progress.console.print(
                Panel(
                    Text.assemble(
                        ("📄 Report saved to ", "bold blue"),
                        (f"{filename}\n", "bold yellow"),
                        ("🗒️  Total comments: ", "bold blue"),
                        (f"{len(comments)}\n", "bold cyan"),
                        ("🔐 PII detected in: ", "bold blue"),
                        (f"{total_pii_comments} ", "bold red"),
                        (f"({total_pii_comments/len(comments):.1%})\n", "dim"),
                        ("🤖 LLM findings in: ", "bold blue"),
                        (f"{total_llm_pii_comments} ", "bold magenta"),
                        (f"({total_llm_pii_comments/len(comments):.1%})", "dim")
                    ),
                    title="[bold green]Analysis Complete[/]",
                    border_style="green",
                    padding=(1, 4)
                )
            )

    @with_logging(logger)
    def _generate_summary_table(self, filtered_results: List[AnalysisResult]) -> Table:
        """Generate summary table with selection indicators"""
        logger.debug_with_context("Generating summary table")
        table = Table(
            title="[bold]Comments Requiring Action[/]",
            header_style="bold magenta",
            box=None
        )
        # Add columns with style parameters directly
        table.add_column("Risk", justify="right", width=8)
        table.add_column("Sentiment", width=12)
        table.add_column("Comment Preview", width=75)
        table.add_column("Upvotes", width=8)
        table.add_column("ID", width=8)
        for result in filtered_results:
            logger.debug_with_context(f"Adding comment ID {result.comment_id} to summary table")
            # Determine risk level styling
            risk_style = "red" if result.pii_risk_score > 0.5 else "yellow" if result.pii_risk_score > 0.2 else "green"
            risk_text = Text(f"{result.pii_risk_score:.0%}", style=risk_style)
            permalink = f"https://reddit.com{result.permalink}"
            # Trim comment text for preview
            preview = result.text[:67] + "..." if len(result.text) > 70 else result.text
            preview = f"[link={permalink}]{preview}[/link]"
            table.add_row(
                risk_text,
                Text(f"{result.sentiment_emoji} {result.sentiment_score:.2f}"),
                preview,
                Text(f"{result.upvotes} / {result.downvotes}"),
                result.comment_id,
            )
        return table

    @with_logging(logger)
    def _print_config(self, auth_enabled: bool, pii_enabled: bool, llm_config: Optional[Dict[str, Any]]) -> None:
        logger.debug_with_context("Printing active configuration")
        progress = Progress(
            SpinnerColumn(spinner_name="dots"),
            TextColumn("[bold blue]{task.description}"),
            TimeElapsedColumn(),
            transient=True
        )
        with progress:
            task = progress.add_task("", total=1, visible=False)
            progress.console.print("\n[bold cyan]Active Configuration[/]")
            def format_status(enabled, true_text="Enabled", false_text="Disabled"):
                return Text.assemble(
                    (true_text if enabled else false_text, "green" if enabled else "red")
                )
            config_table = [
                ("Authentication", format_status(auth_enabled)),
                ("PII Detection", format_status(pii_enabled)),
                ("LLM Analysis", format_status(llm_config is not None, llm_config['model'] if llm_config else "Disabled")),
                ("PII-Only Filter", format_status(self.pii_only, "Active", "Inactive")),
                ("Comment Limit", Text(f"{self.limit if self.limit else 'Unlimited'}", style="cyan")),
                ("Sort Preference", Text(f"{self.sort if self.sort else 'New'}", style="cyan"))
            ]
            panels = []
            panels.append(
                Panel.fit(
                    Group(*[Text.assemble(f"{k}: ", Text("")) + v for k, v in config_table]),
                    title="[bold]Features[/]",
                    border_style="blue"
                )
            )
            if auth_enabled:
                auth_table = [
                    ("REDDIT_USERNAME", environ.get("REDDIT_USERNAME", "[red]Not Set[/]")),
                    ("REDDIT_CLIENT_ID", environ.get("REDDIT_CLIENT_ID", "[red]Not Set[/]"))
                ]
                panels.append(
                    Panel.fit(
                        Group(*[Text(f"{k}: {v}") for k, v in auth_table]),
                        title="[bold]Auth Environment[/]",
                        border_style="yellow"
                    )
                )
            progress.console.print(Columns(panels))
            progress.update(task, advance=1)

    def _print_comments(self, comments: List[Dict[str, Any]], url: str) -> None:
        """Prints out analysis of user comments.
        :param: comments: the parsed contents to analyze.
        :param: url: the url being parsed.
        """
        logger.debug_with_context(f"Printing comments analysis for URL '{url}'")
        def should_show_result(result):
            if not self.pii_only:
                return True
            # Only show results with actual PII detections
            has_pattern_pii = result.pii_risk_score > 0.0
            has_llm_pii = (result.llm_findings and
                          isinstance(result.llm_findings, dict) and
                          result.llm_findings.get('has_pii', False) and
                          result.llm_findings.get('confidence', 0.0) > 0.0)
            return has_pattern_pii or has_llm_pii
        total_comments = len(comments)
        # Create overall stats panel
        stats_panel = Panel(
            Group(
                Text.assemble(("Analysis for: ", "dim"), (f"{url}", "cyan")),
                Text.assemble(("📊 Comments analyzed: ", "dim"), (f"{total_comments}", "cyan")),
                Text.assemble(
                    ("Overall Sentiment Score: ", "dim"),
                    (f"{self.score:.4f}", "cyan bold"),
                    (" ", ""),
                    (f"{self.sentiment}", "yellow")
                )
            ),
            title="[bold]Sentiment Analysis Summary[/]",
            border_style="blue"
        )
        # Filter results if pii_only is enabled
        filtered_results = [r for r in self.results if should_show_result(r)]
        if hasattr(self, 'pii_only') and self.pii_only and not filtered_results:
            logger.info_with_context("No comments with high PII risk found.")
            print("No comments with high PII risk found.")
            return
        panels = []
        for i, result in enumerate(filtered_results, 1):
            # Basic info panel
            basic_info = Group(
                Text("━" * 50, style="dim"),  # Separator line
                Text.assemble(("📝 ", "yellow"), ("Comment Text:", "bold cyan"), style=""),
                Text(f"{result.text}", style="white"),
                Text("━" * 50, style="dim"),  # Separator line
                Text.assemble(
                    ("🎭 ", "yellow"),
                    ("Sentiment Analysis:", "bold cyan"),
                    ("\n   Score: ", "dim"),
                    (f"{result.sentiment_score:.2f}", "cyan bold"),
                    ("  ", ""),
                    (f"{result.sentiment_emoji}", "bold yellow")
                ),
                Text.assemble(
                    ("🔒 ", "yellow"),
                    ("Privacy Risk:", "bold cyan"),
                    ("\n   Score: ", "dim"),
                    (f"{result.pii_risk_score:.2f}", "red bold" if result.pii_risk_score > 0.5 else "green bold")
                ),
                Text("━" * 50, style="dim"),  # Separator line
                Text.assemble(
                    ("📊 ", "yellow"),
                    ("Upvotes: ", "bold cyan"),
                    # (" ⬆️ ", "green"),
                    (f"{result.upvotes} / ", "green"),
                    # (" ⬇️ ", "red"),
                    (f"{result.downvotes}", "red")
                )
            )
            # PII Matches panel
            pii_content = []
            if result.pii_matches:
                for pii in result.pii_matches:
                    pii_content.append(Text.assemble(
                        ("• ", "yellow"),
                        (f"{pii.type}: ", "bold"),
                        (f"({pii.confidence:.2f})", "dim")
                    ))
            # LLM Findings panel
            llm_content = []
            if isinstance(result.llm_findings, dict) and "error" in result.llm_findings:
                # Handle error case
                error_msg = result.llm_findings["error"]
                llm_content.extend([
                    Text("❌ LLM Analysis Failed", style="bold red"),
                    Text.assemble(("Error: ", "bold red"), (error_msg, "red")),
                    Text("\nPlease check:", style="yellow"),
                    Text("• Your OpenAI API key is valid", style="yellow"),
                    Text("• You have sufficient API credits", style="yellow"),
                    Text("• The API service is available", style="yellow")
                ])
            elif result.llm_findings:
                # Helper function to safely extract detail text
                def format_detail(detail):
                    if isinstance(detail, dict):
                        # Handle different model response formats
                        return (
                            f"{detail.get('type', 'Finding')}: {detail.get('example', 'N/A')}".strip()
                            or f"{detail.get('finding', 'N/A')}: {detail.get('reasoning', '')}".strip()
                            or str(detail)
                        )
                    return str(detail)
                llm_content.extend([
                    Text.assemble(
                        ("Risk Score: ", "dim"),
                        (f"{result.llm_risk_score:.2f}", "red" if result.llm_risk_score > 0.5 else "green")
                    ),
                    Text.assemble(
                        ("PII Detected: ", "dim"),
                        ("Yes" if result.llm_findings.get('has_pii') else "No",
                         "red" if result.llm_findings.get('has_pii') else "green")
                    )
                ])
                if result.llm_findings.get('details'):
                    llm_content.append(Text("Findings:", style="bold"))
                    for detail in result.llm_findings['details']:
                        try:
                            detail_text = format_detail(detail)
                            llm_content.append(Text(f"  • {detail_text}", style="cyan"))
                        except Exception as e:
                            logger.debug_with_context(f"Error formatting LLM detail: {str(e)}")
                            llm_content.append(Text("  • [Malformed finding]", style="red dim"))
                if result.llm_findings.get('risk_factors'):
                    llm_content.append(Text("\nRisk Factors:", style="bold"))
                    for factor in result.llm_findings['risk_factors']:
                        llm_content.append(Text(f"  • {factor}", style="yellow"))
            # Create sub-panels
            sub_panels = [Panel(basic_info, title="[bold]Basic Info[/]", border_style="blue")]
            if pii_content:
                sub_panels.append(Panel(
                    Group(*pii_content),
                    title="[bold]Pattern PII[/]",
                    border_style="yellow"
                ))
            if llm_content:
                sub_panels.append(Panel(
                    Group(*llm_content),
                    title="[bold]LLM Analysis[/]",
                    border_style="magenta"
                ))
            # Add sentiment analysis summary panel
            if i == 1:  # Only add to first comment
                sub_panels.append(stats_panel)
            # Combine sub-panels into a main panel for this comment
            panels.append(Panel(
                Columns(sub_panels),
                title=f"[bold]Comment {i}[/]",
                border_style="cyan"
            ))
        # Add summary table
        summary_table = self._generate_summary_table(filtered_results)
        panels.append(Panel(summary_table,
            title="[bold]Action Summary[/]",
            border_style="green",
            padding=(1, 4)
        ))
        # Add action confirmation prompt with ready-to-use commands
        high_risk_comments = [r for r in filtered_results if r.pii_risk_score > 0.5 or 
                            (r.llm_findings and r.llm_findings.get('has_pii', False))]
        comment_ids = [r.permalink for r in high_risk_comments]
        
        action_text = Group(
            Text("Ready-to-use commands for high-risk comments:", style="bold yellow"),
            Text.assemble(
                ("Delete comments:\n", "bold red"),
                ((comment_ids, 'delete'), "italic red")
            ),
            Text.assemble(
                ("\nUpdate/reddact comments:\n", "bold blue"),
                ((comment_ids, 'update'), "italic blue")
            ) if comment_ids else Text("No high-risk comments found", style="green")
        )
        panels.append(
            Panel.fit(
                action_text,
                border_style="yellow",
                title="[bold]Actions[/]"
            )
        )
        # Print all panels
        with Progress(
            SpinnerColumn(spinner_name="dots"),
            TextColumn("[bold blue]{task.description}"),
            TimeElapsedColumn(),
            transient=True
        ) as progress:
            task = progress.add_task("", total=1, visible=False)
            progress.console.print(Group(*panels))
            progress.update(task, advance=1)

    @with_logging(logger)
    def _get_comments(self, source_type: str, identifier: str, **kwargs) -> List[Dict[str, Any]]:
        """Unified comment fetching method"""
        logger.debug_with_context(f"Fetching comments for {source_type} '{identifier}'")
        fetch_method = {
            'user': self.api.parse_user,
            'listing': self.api.parse_listing
        }[source_type]

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
            self._generate_output_file(output_file, comments, identifier)
        else:
            self._print_comments(comments, identifier)
