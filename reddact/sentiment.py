#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os import environ

import logging
import re
import asyncio
import sys
from dataclasses import dataclass
from typing import List, Dict, Any
from rich.panel import Panel
from rich.columns import Columns
from rich.console import Group
from rich.text import Text

from reddact.api.scraper import Scraper
from reddact.api.reddit import Reddit
from reddact.pii_detector import PIIDetector
from reddact.llm_detector import LLMDetector
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from nltk.sentiment.vader import SentimentIntensityAnalyzer


@dataclass
class AnalysisResult:
    """Holds the results of both sentiment and PII analysis"""
    sentiment_score: float
    sentiment_emoji: str
    pii_risk_score: float
    pii_matches: List[Any]
    text: str
    llm_risk_score: float = 0.0
    llm_findings: Dict[str, Any] = None

happy_sentiment = "😁"
sad_sentiment = "😕"
neutral_sentiment = "😐"


class Sentiment():
    """Performs the sentiment analysis on a given set of Reddit Objects."""

    def __init__(self, auth_enabled=False, pii_enabled=True, llm_config=None, pii_only=False, debug=False, limit=100):
        # Configure logging
        log_level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='[%(levelname)s] %(message)s',
            stream=sys.stdout
        )
        self.api = Scraper()
        self.score = 0
        self.sentiment = neutral_sentiment
        self.headers = {'User-agent': "Reddit Sentiment Analyzer"}
        self.authEnable = False
        self.pii_enabled = pii_enabled
        self.pii_detector = PIIDetector() if pii_enabled else None
        self.pii_only = pii_only
        self.limit = limit
        
        # Initialize LLM detector if config provided
        self.llm_detector = None
        if llm_config and pii_enabled:
            self.llm_detector = LLMDetector(
                api_key=llm_config.get('api_key'),
                api_base=llm_config.get('api_base'),
                model=llm_config.get('model', 'gpt-3.5-turbo')
            )
        
        if auth_enabled:
            self.api = Reddit()
        self._print_config(auth_enabled, pii_enabled, llm_config)

    def get_user_sentiment(self, username, output_file=None):
        """Obtains the sentiment for a user's comments.

        :param username: name of user to search
        :param output_file (optional): file to output relevant data.
        """
        comments = self.api.parse_user(username, headers=self.headers, limit=self.limit)
        self.score, self.results = self._analyze(comments)
        self.sentiment = self._get_sentiment(self.score)

        user_id = f"/user/{username}"

        if output_file:
            self._generate_output_file(output_file, comments, user_id)
        else:
             self._print_comments(comments, user_id)

    def get_listing_sentiment(self, subreddit, article, output_file=None):
        """Obtains the sentiment for a listing's comments.

        :param subreddit: a subreddit
        :param article: an article associated with the subreddit
        :param output_file (optional): file to output relevant data.
        """
        comments = self.api.parse_listing(subreddit,
                                          article,
                                          headers=self.headers,
                                          limit=self.limit)
        self.score, self.results = self._analyze(comments)
        self.sentiment = self._get_sentiment(self.score)

        article_id = f"/r/{subreddit}/comments/{article}"

        if output_file:
            self._generate_output_file(output_file, comments, article_id)
        else:
             self._print_comments(comments, article_id)

    def _analyze(self, comments):
        """Analyzes comments for both sentiment and PII content.

        :param comments: comments to perform analysis on.
        :return: tuple of (sentiment_score, list of AnalysisResult objects)
        """
        sentiment_analyzer = SentimentIntensityAnalyzer()
        final_score = 0
        results = []

        cleanup_regex = re.compile('<.*?>')

        total_comments = len(comments)
        print(f"\n📊 Retrieved {total_comments} comments to analyze")
        
        progress = Progress(
            SpinnerColumn(spinner_name="dots"),
            TextColumn("[bold blue]{task.description}"),
            TimeElapsedColumn(),
            transient=True
        )
        with progress:
            main_task = progress.add_task(f"💭 Processing comments...", total=total_comments)
            pii_task = progress.add_task("🔍 PII Analysis", visible=False, total=1)
            llm_task = progress.add_task("🤖 LLM Analysis", visible=False, total=1)
            
            for i, comment in enumerate(comments, 1):
                clean_comment = re.sub(cleanup_regex, '', str(comment))
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
                        sentiment_score=score,
                        sentiment_emoji=self._get_sentiment(score),
                        pii_risk_score=pii_risk_score,  # Initial PII score
                        pii_matches=pii_matches,
                        text=clean_comment,
                        llm_risk_score=0.0,
                        llm_findings=None
                    )
                    self._pending_results.append(result)
                    
                    # Process batch when full or at end
                    if len(self._llm_batch) >= 3 or i == total_comments:
                        logging.debug(f"\nProcessing LLM batch of {len(self._llm_batch)} items")
                        progress.update(llm_task, visible=True)
                        progress.update(llm_task, description="🤖 Starting LLM analysis")
                        batch_results = asyncio.run(self.llm_detector.analyze_batch(self._llm_batch))
                        progress.update(llm_task, description="✅ LLM analysis complete")
                        logging.debug(f"LLM batch_results: {batch_results}")
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
                            logging.debug("Added result to final results")
                        
                        # Clear batch
                        self._llm_batch = []
                        self._llm_batch_indices = []
                        self._pending_results = []
                
                # Only append results directly if not using LLM
                if not self.llm_detector:
                    results.append(AnalysisResult(
                        sentiment_score=score,
                        sentiment_emoji=self._get_sentiment(score),
                        pii_risk_score=pii_risk_score,
                        pii_matches=pii_matches,
                        text=clean_comment,
                        llm_risk_score=0.0,
                        llm_findings=None
                    ))
                
                progress.update(main_task, advance=1)

        try:
            rounded_final = round(final_score/len(comments), 4)
            return rounded_final, results
        except ZeroDivisionError:
            logging.error("No comments found")
            return 0.0, []

    def _get_sentiment(self, score):
        """Obtains the sentiment using a sentiment score.

        :param score: the sentiment score.
        :return: sentiment from score.
        """
        if score == 0:
            return neutral_sentiment
        elif score > 0:
            return happy_sentiment
        else:
            return sad_sentiment

    def _generate_output_file(self, filename, comments, url):
        """Outputs a file containing a detailed sentiment and PII analysis per
        sentence.

        :param: filename: the name of the file to create and edit
        :param: comments: the parsed contents to analyze.
        :param: url: the url being parsed.
        """
        with open(filename, 'w+') as target:
            target.write(f"Analysis for '{url}'\n")
            target.write(f"Overall Sentiment Score: {self.score}\n")
            target.write(f"Overall Sentiment: {self.sentiment}\n\n")

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
            for comment in comments:
                score, results = self._analyze([comment])
                filtered_results = [r for r in results if should_show_result(r)]
                
                if not filtered_results and hasattr(self, 'pii_only') and self.pii_only:
                    continue
                    
                for result in filtered_results:
                    target.write(f"Comment {comment_count}:\n")
                    target.write(f"Text: {result.text}\n")
                    target.write(f"Sentiment Score: {result.sentiment_score}\n")
                    target.write(f"Sentiment: {result.sentiment_emoji}\n")
                    target.write(f"PII Risk Score: {result.pii_risk_score:.2f}\n")
                    
                    if result.pii_matches:
                        target.write("Pattern-based PII Detected:\n")
                        for pii in result.pii_matches:
                            target.write(f"  - Type: {pii.type}\n")
                            target.write(f"    Confidence: {pii.confidence:.2f}\n")
                    
                    if result.llm_findings:
                        target.write("\nLLM Privacy Analysis:\n")
                        target.write(f"  Risk Score: {result.llm_risk_score:.2f}\n")
                        if isinstance(result.llm_findings, dict):
                            if result.llm_findings.get('has_pii'):
                                target.write("  PII Detected: Yes\n")
                            if result.llm_findings.get('details'):
                                target.write("  Findings:\n")
                                for detail in result.llm_findings['details']:
                                    target.write(f"    - {detail}\n")
                            if result.llm_findings.get('reasoning'):
                                target.write(f"\n  Reasoning:\n    {result.llm_findings['reasoning']}\n")
                            if result.llm_findings.get('risk_factors'):
                                target.write("\n  Risk Factors:\n")
                                for factor in result.llm_findings['risk_factors']:
                                    target.write(f"    - {factor}\n")
                    target.write("\n")
                    
                comment_count += 1


    def _print_comments(self, comments, url):
        """Prints out analysis of user comments.

        :param: comments: the parsed contents to analyze.
        :param: url: the url being parsed.
        """
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
            if result.llm_findings:
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
                        if isinstance(detail, dict):
                            llm_content.append(Text(f"  • {detail['finding']}: {detail['reasoning']}", style="cyan"))
                        else:
                            llm_content.append(Text(f"  • {detail}", style="cyan"))
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
  
    def _print_config(self, auth_enabled, pii_enabled, llm_config):
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
                ("Comment Limit", Text(f"{self.limit if self.limit else 'Unlimited'}", style="cyan"))
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
