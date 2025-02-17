#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import logging
from os import environ
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Third-party
from rich.panel import Panel
from rich.columns import Columns
from rich.console import Group
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

# Local
from reddacted.utils.logging import get_logger, with_logging

logger = get_logger(__name__)

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

class ResultsFormatter:
    """Handles formatting and display of analysis results"""
    
    def __init__(self):
        self.logger = get_logger(__name__)

    @with_logging(logger)
    def create_progress(self) -> Progress:
        """Creates a unified progress context manager"""
        return Progress(
            SpinnerColumn(spinner_name="dots"),
            TextColumn("[bold blue]{task.description}"),
            TimeElapsedColumn(),
            transient=True
        )

    @with_logging(logger)
    def generate_output_file(self, filename: str, comments: List[Dict[str, Any]], 
                           url: str, results: List[AnalysisResult], 
                           overall_score: float, overall_sentiment: str) -> None:
        """Outputs a file containing a detailed sentiment and PII analysis per sentence."""
        with self.create_progress() as progress:
            progress_task = progress.add_task("📝 Generating analysis report...", total=len(comments))

            with open(filename, 'w+') as target:
                target.write(f"# Analysis Report for '{url}'\n\n")
                target.write(f"- **Overall Sentiment Score**: {overall_score:.2f}\n")
                target.write(f"- **Overall Sentiment**: {overall_sentiment}\n")
                target.write(f"- **Comments Analyzed**: {len(comments)}\n\n")
                target.write("---\n\n")

                total_pii_comments = 0
                total_llm_pii_comments = 0
                max_risk_score = 0.0
                riskiest_comment = None
                sentiment_scores = []

                comment_count = 1
                for result in results:
                    progress.update(progress_task, description=f"📝 Writing comment {comment_count}/{len(comments)}")

                    if not self._should_show_result(result):
                        comment_count += 1
                        progress.update(progress_task, advance=1)
                        continue

                    self._write_comment_details(target, result, comment_count)
                    self._update_summary_stats(result, sentiment_scores)
                    
                    if result.pii_risk_score > max_risk_score:
                        max_risk_score = result.pii_risk_score
                        riskiest_comment = result.text[:100] + "..." if len(result.text) > 100 else result.text
                    
                    comment_count += 1
                    progress.update(progress_task, advance=1)

                self._write_summary_section(target, comments, sentiment_scores, 
                                         total_pii_comments, total_llm_pii_comments,
                                         max_risk_score, riskiest_comment)

            self._print_completion_message(filename, comments, results)

    @with_logging(logger)
    def print_config(self, auth_enabled: bool, pii_enabled: bool, 
                    llm_config: Optional[Dict[str, Any]], 
                    pii_only: bool, limit: int, sort: str) -> None:
        """Prints the active configuration"""
        with self.create_progress() as progress:
            task = progress.add_task("", total=1, visible=False)
            progress.console.print("\n[bold cyan]Active Configuration[/]")
            
            config_table = self._create_config_table(auth_enabled, pii_enabled, 
                                                   llm_config, pii_only, limit, sort)
            panels = [self._create_features_panel(config_table)]
            
            if auth_enabled:
                panels.append(self._create_auth_panel())
                
            progress.console.print(Columns(panels))
            progress.update(task, advance=1)

    @with_logging(logger)
    def print_comments(self, comments: List[Dict[str, Any]], url: str, 
                      results: List[AnalysisResult], 
                      overall_score: float, overall_sentiment: str) -> None:
        """Prints out analysis of user comments"""
        filtered_results = [r for r in results if self._should_show_result(r)]
        
        if not filtered_results and getattr(self, 'pii_only', False):
            logger.info("No comments with high PII risk found.")
            print("No comments with high PII risk found.")
            return

        panels = []
        stats_panel = self._create_stats_panel(url, len(comments), overall_score, overall_sentiment)
        
        for i, result in enumerate(filtered_results, 1):
            panels.append(self._create_comment_panel(result, i, stats_panel if i == 1 else None))

        summary_table = self._generate_summary_table(filtered_results)
        panels.append(self._create_summary_panel(summary_table))
        panels.append(self._create_action_panel(filtered_results))

        with self.create_progress() as progress:
            task = progress.add_task("", total=1, visible=False)
            progress.console.print(Group(*panels))
            progress.update(task, advance=1)

    def _should_show_result(self, result: AnalysisResult) -> bool:
        """Determines if a result should be shown based on PII detection settings"""
        if not getattr(self, 'pii_only', False):
            return True
        
        has_pattern_pii = result.pii_risk_score > 0.0
        has_llm_pii = (result.llm_findings and
                      isinstance(result.llm_findings, dict) and
                      result.llm_findings.get('has_pii', False) and
                      result.llm_findings.get('confidence', 0.0) > 0.0)
        return has_pattern_pii or has_llm_pii

    def _generate_summary_table(self, filtered_results: List[AnalysisResult]) -> Table:
        """Generate summary table with selection indicators"""
        table = Table(
            title="[bold]Comments Requiring Action[/]",
            header_style="bold magenta",
            box=None
        )
        
        table.add_column("Risk", justify="right", width=8)
        table.add_column("Sentiment", width=12)
        table.add_column("Comment Preview", width=75)
        table.add_column("Upvotes", width=8)
        table.add_column("ID", width=8)
        
        for result in filtered_results:
            risk_style = "red" if result.pii_risk_score > 0.5 else "yellow" if result.pii_risk_score > 0.2 else "green"
            risk_text = Text(f"{result.pii_risk_score:.0%}", style=risk_style)
            permalink = f"https://reddit.com{result.permalink}"
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

    # Helper methods for panel creation
    def _create_features_panel(self, config_table):
        return Panel.fit(
            Group(*[Text.assemble(f"{k}: ", Text("")) + v for k, v in config_table]),
            title="[bold]Features[/]",
            border_style="blue"
        )

    def _create_auth_panel(self):
        auth_table = [
            ("REDDIT_USERNAME", environ.get("REDDIT_USERNAME", "[red]Not Set[/]")),
            ("REDDIT_CLIENT_ID", environ.get("REDDIT_CLIENT_ID", "[red]Not Set[/]"))
        ]
        return Panel.fit(
            Group(*[Text(f"{k}: {v}") for k, v in auth_table]),
            title="[bold]Auth Environment[/]",
            border_style="yellow"
        )

    def _create_config_table(self, auth_enabled, pii_enabled, llm_config, pii_only, limit, sort):
        def format_status(enabled, true_text="Enabled", false_text="Disabled"):
            return Text.assemble(
                (true_text if enabled else false_text, "green" if enabled else "red")
            )
        
        return [
            ("Authentication", format_status(auth_enabled)),
            ("PII Detection", format_status(pii_enabled)),
            ("LLM Analysis", format_status(llm_config is not None, 
                                         llm_config['model'] if llm_config else "Disabled")),
            ("PII-Only Filter", format_status(pii_only, "Active", "Inactive")),
            ("Comment Limit", Text(f"{limit if limit else 'Unlimited'}", style="cyan")),
            ("Sort Preference", Text(f"{sort if sort else 'New'}", style="cyan"))
        ]

    def _create_stats_panel(self, url, total_comments, score, sentiment):
        return Panel(
            Group(
                Text.assemble(("Analysis for: ", "dim"), (f"{url}", "cyan")),
                Text.assemble(("📊 Comments analyzed: ", "dim"), (f"{total_comments}", "cyan")),
                Text.assemble(
                    ("Overall Sentiment Score: ", "dim"),
                    (f"{score:.4f}", "cyan bold"),
                    (" ", ""),
                    (f"{sentiment}", "yellow")
                )
            ),
            title="[bold]Sentiment Analysis Summary[/]",
            border_style="blue"
        )

    def _create_comment_panel(self, result, index, stats_panel=None):
        sub_panels = [self._create_basic_info_panel(result)]
        
        if result.pii_matches:
            sub_panels.append(self._create_pii_panel(result))
            
        if result.llm_findings:
            sub_panels.append(self._create_llm_panel(result))
            
        if stats_panel:
            sub_panels.append(stats_panel)
            
        return Panel(
            Columns(sub_panels),
            title=f"[bold]Comment {index}[/]",
            border_style="cyan"
        )

    def _create_basic_info_panel(self, result):
        return Panel(
            Group(
                Text("━" * 50, style="dim"),
                Text.assemble(("📝 ", "yellow"), ("Comment Text:", "bold cyan")),
                Text(f"{result.text}", style="white"),
                Text("━" * 50, style="dim"),
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
                    (f"{result.pii_risk_score:.2f}", 
                     "red bold" if result.pii_risk_score > 0.5 else "green bold")
                ),
                Text("━" * 50, style="dim"),
                Text.assemble(
                    ("📊 ", "yellow"),
                    ("Upvotes: ", "bold cyan"),
                    (f"{result.upvotes} / ", "green"),
                    (f"{result.downvotes}", "red")
                )
            ),
            title="[bold]Basic Info[/]",
            border_style="blue"
        )

    def _create_pii_panel(self, result):
        pii_content = [
            Text.assemble(
                ("• ", "yellow"),
                (f"{pii.type}: ", "bold"),
                (f"({pii.confidence:.2f})", "dim")
            ) for pii in result.pii_matches
        ]
        return Panel(
            Group(*pii_content),
            title="[bold]Pattern PII[/]",
            border_style="yellow"
        )

    def _create_llm_panel(self, result):
        llm_content = []
        if isinstance(result.llm_findings, dict) and "error" in result.llm_findings:
            llm_content.extend(self._create_llm_error_content(result.llm_findings["error"]))
        else:
            llm_content.extend(self._create_llm_analysis_content(result))
        return Panel(
            Group(*llm_content),
            title="[bold]LLM Analysis[/]",
            border_style="magenta"
        )

    def _create_llm_error_content(self, error_msg):
        return [
            Text("❌ LLM Analysis Failed", style="bold red"),
            Text.assemble(("Error: ", "bold red"), (error_msg, "red")),
            Text("\nPlease check:", style="yellow"),
            Text("• Your OpenAI API key is valid", style="yellow"),
            Text("• You have sufficient API credits", style="yellow"),
            Text("• The API service is available", style="yellow")
        ]

    def _create_llm_analysis_content(self, result):
        content = [
            Text.assemble(
                ("Risk Score: ", "dim"),
                (f"{result.llm_risk_score:.2f}", 
                 "red" if result.llm_risk_score > 0.5 else "green")
            ),
            Text.assemble(
                ("PII Detected: ", "dim"),
                ("Yes" if result.llm_findings.get('has_pii') else "No",
                 "red" if result.llm_findings.get('has_pii') else "green")
            )
        ]
        
        if result.llm_findings.get('details'):
            content.append(Text("Findings:", style="bold"))
            for detail in result.llm_findings['details']:
                detail_text = self._format_llm_detail(detail)
                content.append(Text(f"  • {detail_text}", style="cyan"))
                
        if result.llm_findings.get('risk_factors'):
            content.append(Text("\nRisk Factors:", style="bold"))
            for factor in result.llm_findings['risk_factors']:
                content.append(Text(f"  • {factor}", style="yellow"))
                
        return content

    def _format_llm_detail(self, detail):
        if isinstance(detail, dict):
            return (
                f"{detail.get('type', 'Finding')}: {detail.get('example', 'N/A')}".strip()
                or f"{detail.get('finding', 'N/A')}: {detail.get('reasoning', '')}".strip()
                or str(detail)
            )
        return str(detail)

    def _create_summary_panel(self, summary_table):
        return Panel(
            summary_table,
            title="[bold]Action Summary[/]",
            border_style="green",
            padding=(1, 4)
        )

    def _create_action_panel(self, filtered_results):
        high_risk_comments = [
            r for r in filtered_results 
            if r.pii_risk_score > 0.5 or (
                r.llm_findings and r.llm_findings.get('has_pii', False)
            )
        ]
        comment_ids = [r.comment_id for r in high_risk_comments]
        
        action_text = Group(
            Text("Ready-to-use commands for high-risk comments:", style="bold yellow"),
            Text.assemble(
                ("Delete comments:\n", "bold red"),
                ("reddacted delete " + " ".join(comment_ids), "italic red")
            ),
            Text.assemble(
                ("\nReddact (edit) comments:\n", "bold blue"),
                ("reddacted update " + " ".join(comment_ids), "italic blue")
            ) if comment_ids else Text("No high-risk comments found", style="green")
        )
        
        return Panel.fit(
            action_text,
            border_style="yellow",
            title="[bold]Actions[/]"
        )
