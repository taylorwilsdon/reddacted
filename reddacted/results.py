#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import logging
import os
from itertools import zip_longest
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
from reddacted.report_generator import (
    generate_analysis_report,
    should_show_result,
    format_llm_detail,
)

logger = get_logger(__name__)

@dataclass
class AnalysisResult:
    """Holds the results of both sentiment and PII analysis."""
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
    """Handles formatting and display of analysis results."""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.total_pii_comments = 0
        self.total_llm_pii_comments = 0

    @with_logging(logger)
    def create_progress(self) -> Progress:
        """Creates a unified progress context manager."""
        if not hasattr(self, '_progress'):
            self._progress = Progress(
                SpinnerColumn(spinner_name="dots"),
                TextColumn("[bold blue]{task.description}"),
                TimeElapsedColumn(),
                transient=True
            )
        return self._progress

    @with_logging(logger)
    def generate_output_file(
        self,
        filename: str,
        comments: List[Dict[str, Any]],
        url: str,
        results: List[AnalysisResult],
        overall_score: float,
        overall_sentiment: str
    ) -> None:
        """Outputs a file containing a detailed sentiment and PII analysis per comment."""
        progress = self.create_progress()
        with progress:
            progress_task = progress.add_task("📝 Generating analysis report...", total=len(comments))
            try:
                stats = generate_analysis_report(
                    filename=filename,
                    comments=comments,
                    url=url,
                    results=results,
                    overall_score=overall_score,
                    overall_sentiment=overall_sentiment,
                    pii_only=getattr(self, 'pii_only', False)
                )
                self.total_pii_comments = stats['total_pii_comments']
                self.total_llm_pii_comments = stats['total_llm_pii_comments']
                self._print_completion_message(filename, comments, results, progress)
            except Exception as e:
                self.logger.exception("Failed to generate output file: %s", e)
                raise

    @with_logging(logger)
    def print_config(
        self,
        auth_enabled: bool,
        pii_enabled: bool,
        llm_config: Optional[Dict[str, Any]],
        pii_only: bool,
        limit: int,
        sort: str
    ) -> None:
        """Prints the active configuration."""
        progress = self.create_progress()
        with progress:
            progress.console.print("\n[bold cyan]Active Configuration[/]")
            features_panel = self._create_features_panel(auth_enabled, pii_enabled, llm_config, pii_only, limit, sort)
            panels = [features_panel]
            if auth_enabled:
                auth_panel = self._create_auth_panel()
                panels.append(auth_panel)
            progress.console.print(Columns(panels))

    @with_logging(logger)
    def print_comments(
        self,
        comments: List[Dict[str, Any]],
        url: str,
        results: List[AnalysisResult],
        overall_score: float,
        overall_sentiment: str
    ) -> None:
        """Prints out analysis of user comments."""
        filtered_results = [r for r in results if should_show_result(r, getattr(self, 'pii_only', False))]
        if not filtered_results and getattr(self, 'pii_only', False):
            self.logger.info("No comments with high PII risk found.")
            print("No comments with high PII risk found.")
            return
        panels = []
        stats_panel = self._create_stats_panel(url, len(comments), overall_score, overall_sentiment)
        for idx, result in enumerate(filtered_results, 1):
            comment_panel = self._create_comment_panel(result, idx)
            panels.append(comment_panel)
        summary_table = self._generate_summary_table(filtered_results)
        summary_panel = self._create_summary_panel(summary_table)
        action_panel = self._create_action_panel(filtered_results)
        progress = self.create_progress()
        with progress:
            progress.console.print(Group(stats_panel, *panels, summary_panel, action_panel))

    def _generate_summary_table(self, filtered_results: List[AnalysisResult]) -> Table:
        """Generates a summary table with selection indicators."""
        table = Table(
            header_style="bold magenta",
            box=None,
            padding=(0,1),
            collapse_padding=True
        )
        table.add_column("Risk", justify="center", style="bold", width=10)
        table.add_column("Sentiment", justify="center", width=15)
        table.add_column("Comment Preview", justify="center", width=75)
        table.add_column("Votes", justify="center", width=10)
        table.add_column("ID", justify="center", width=10)
        for result in filtered_results:
            risk_style = self._get_risk_style(result.pii_risk_score)
            risk_text = Text(f"{result.pii_risk_score:.0%}", style=risk_style)
            permalink = f"https://reddit.com{result.permalink}"
            preview = (result.text[:67] + "...") if len(result.text) > 70 else result.text
            preview = f"[link={permalink}]{preview}[/link]"
            # Format votes based on whether they're positive or negative
            vote_display = (
                f"[green]⬆️ {result.upvotes:>3}[/]" if result.upvotes > result.downvotes else
                f"[red]⬇️ {result.downvotes:>3}[/]" if result.downvotes > result.upvotes else
                f"[dim]0[/]"
            )
            table.add_row(
                risk_text,
                Text(f"{result.sentiment_emoji} {result.sentiment_score:.2f}"),
                preview,
                vote_display,
                result.comment_id
            )
        return table

    # Helper methods for creating panels and formatting
    def _get_risk_style(self, score: float) -> str:
        if score > 0.5:
            return "red"
        elif score > 0.2:
            return "yellow"
        else:
            return "green"

    def _create_features_panel(
        self,
        auth_enabled: bool,
        pii_enabled: bool,
        llm_config: Optional[Dict[str, Any]],
        pii_only: bool,
        limit: int,
        sort: str
    ) -> Panel:
        """Creates a panel displaying the features configuration."""
        # Create a table with two columns
        features_table = Table(
            show_header=False,
            box=None,
            padding=(0, 2),
            collapse_padding=True,
            expand=True
        )
        features_table.add_column("Left", ratio=1, justify="left")
        features_table.add_column("Right", ratio=1, justify="left")

        # Define all config items
        config_items = [
            ("🔐 Authentication", self._format_status(auth_enabled)),
            ("🔍 PII Detection", self._format_status(pii_enabled)),
            ("🤖 LLM Analysis", Text(llm_config['model'], style="green") if llm_config else self._format_status(False)),
            ("🎯 PII-Only Filter", self._format_status(pii_only, "Active", "Inactive")),
            ("📊 Comment Limit", Text(f"{limit}" if limit else "Unlimited", style="cyan")),
            ("📑 Sort Preference", Text(f"{sort}" if sort else "New", style="cyan"))
        ]

        # Split items into two columns
        mid_point = (len(config_items) + 1) // 2
        left_items = config_items[:mid_point]
        right_items = config_items[mid_point:]

        # Create formatted text for each column
        for left, right in zip_longest(left_items, right_items, fillvalue=None):
            left_text = Text.assemble(f"{left[0]}: ", left[1]) if left else Text("")
            right_text = Text.assemble(f"{right[0]}: ", right[1]) if right else Text("")
            features_table.add_row(left_text, right_text)

        return Panel(
            features_table,
            title="[bold]Features[/]",
            border_style="blue",
            padding=(1, 1),
            expand=True
        )

    def _create_auth_panel(self) -> Panel:
        """Creates a panel displaying the authentication environment variables."""
        auth_vars = [
            ("REDDIT_USERNAME", os.environ.get("REDDIT_USERNAME", "[red]Not Set[/]")),
            ("REDDIT_CLIENT_ID", os.environ.get("REDDIT_CLIENT_ID", "[red]Not Set[/]"))
        ]
        auth_texts = [Text(f"{k}: {v}") for k, v in auth_vars]
        return Panel(
            Group(*auth_texts),
            title="[bold]Auth Environment[/]",
            border_style="yellow"
        )

    def _format_status(self, enabled: bool, true_text: str = "Enabled", false_text: str = "Disabled") -> Text:
        """Formats a status text based on a boolean value."""
        return Text(true_text if enabled else false_text, style="green" if enabled else "red")

    def _create_stats_panel(
        self,
        url: str,
        total_comments: int,
        score: float,
        sentiment: str
    ) -> Panel:
        """Creates a panel displaying the sentiment analysis summary."""
        # Create metrics table
        metrics_table = Table(
            show_header=False,
            box=None,
            padding=(0, 2),
            collapse_padding=True
        )
        metrics_table.add_column("Icon", justify="right", style="bold")
        metrics_table.add_column("Label", style="bold")
        metrics_table.add_column("Value", justify="left")

        # Add rows with proper spacing and alignment
        metrics_table.add_row(
            "🔍",
            "Analysis for:",
            f"[link=https://reddit.com/u/{url}]{url}[/]" if url.startswith('u/') else f"[cyan]{url}[/]"
        )
        metrics_table.add_row(
            "📊",
            "Comments analyzed:",
            f"[cyan bold]{total_comments:>4}[/]"
        )
        metrics_table.add_row(
            "🎭",
            "Overall Sentiment:",
            f"[cyan bold]{score:>6.2f}[/] {sentiment}"
        )

        return Panel(
            metrics_table,
            title="[bold]Sentiment Analysis Summary[/]",
            border_style="blue",
            padding=(1, 1)
        )

    def _create_comment_panel(self, result: AnalysisResult, index: int) -> Panel:
        """Creates a panel for a single comment."""
        sub_panels = [self._create_basic_info_panel(result)]
        if result.pii_matches:
            sub_panels.append(self._create_pii_panel(result))
        if result.llm_findings:
            sub_panels.append(self._create_llm_panel(result))
        return Panel(
            Columns(sub_panels),
            title=f"[bold]Comment {index}[/]",
            border_style="cyan"
        )

    def _create_basic_info_panel(self, result: AnalysisResult) -> Panel:
        """Creates a panel displaying basic comment information."""
        # Create metrics table
        metrics_table = Table(
            show_header=False,
            box=None,
            padding=(0, 2),
            collapse_padding=True
        )
        metrics_table.add_column("Icon", justify="right", style="bold")
        metrics_table.add_column("Label", style="bold")
        metrics_table.add_column("Value", justify="left")

        # Risk score styling
        risk_score_style = "red bold" if result.pii_risk_score > 0.5 else "green bold"

        # Add rows with proper spacing and alignment
        metrics_table.add_row(
            "🎭",
            "Sentiment:",
            f"[cyan bold]{result.sentiment_score:>6.2f}[/] {result.sentiment_emoji}"
        )
        metrics_table.add_row(
            "🔒",
            "Privacy Risk:",
            f"[{risk_score_style}]{result.pii_risk_score:>6.2f}[/]"
        )
        # Format votes based on whether they're positive or negative
        vote_display = (
            f"[green]⬆️ {result.upvotes:>3}[/]" if result.upvotes > result.downvotes else
            f"[red]⬇️ {result.downvotes:>3}[/]" if result.downvotes > result.upvotes else
            f"[dim]0[/]"
        )
        # metrics_table.add_row(
        #     "📊",
        #     "Votes:",
        #     vote_display
        # )

        # Combine comment text and metrics
        basic_info = Group(
            Text(result.text, style="white"),
            Text("─" * 50, style="dim"),
            metrics_table
        )

        return Panel(
            basic_info,
            title="[bold]Basic Info[/]",
            border_style="blue",
            padding=(1, 1)
        )

    def _create_pii_panel(self, result: AnalysisResult) -> Panel:
        """Creates a panel displaying pattern-based PII matches."""
        pii_contents = [
            Text(f"• {pii.type} (confidence: {pii.confidence:.2f})", style="cyan")
            for pii in result.pii_matches
        ]
        return Panel(
            Group(*pii_contents),
            title="[bold]Pattern-based PII Detected[/]",
            border_style="yellow"
        )

    def _create_llm_panel(self, result: AnalysisResult) -> Panel:
        """Creates a panel displaying LLM analysis findings."""
        # Create metrics table similar to basic info panel
        metrics_table = Table(
            show_header=False,
            box=None,
            padding=(0, 2),
            collapse_padding=True
        )
        metrics_table.add_column("Icon", justify="right", style="bold")
        metrics_table.add_column("Label", style="bold")
        metrics_table.add_column("Value", justify="left")

        if isinstance(result.llm_findings, dict) and "error" in result.llm_findings:
            error_group = self._create_llm_error_content(result.llm_findings["error"])
            return Panel(error_group, title="[bold]LLM Analysis[/]", border_style="red")

        # Risk score styling
        risk_style = "red bold" if result.llm_risk_score > 0.5 else "green bold"
        pii_style = "red bold" if result.llm_findings.get('has_pii', False) else "green bold"

        # Add main metrics rows
        metrics_table.add_row(
            "🎯",
            "Risk Score:",
            f"[{risk_style}]{result.llm_risk_score:>6.2f}[/]"
        )
        metrics_table.add_row(
            "🔍",
            "PII Detected:",
            f"[{pii_style}]{'Yes' if result.llm_findings.get('has_pii') else 'No':>6}[/]"
        )

        # Create content groups
        content_groups = [metrics_table]

        # Add findings if present
        if details := result.llm_findings.get('details'):
            findings_table = Table(show_header=False, box=None, padding=(0, 2))
            findings_table.add_column(style="cyan")
            content_groups.extend([
                Text("\n📋 Findings:", style="bold"),
                *[Text(f"  • {format_llm_detail(detail)}", style="cyan") for detail in details]
            ])

        # Add risk factors if present
        if risk_factors := result.llm_findings.get('risk_factors'):
            content_groups.extend([
                Text("\n⚠️ Risk Factors:", style="bold"),
                *[Text(f"  • {factor}", style="yellow") for factor in risk_factors]
            ])

        return Panel(
            Group(*content_groups),
            title="[bold]LLM Analysis[/]",
            border_style="magenta",
            padding=(1, 1)
        )

    def _create_llm_error_content(self, error_msg: str) -> Group:
        """Creates content for LLM analysis errors."""
        error_table = Table(show_header=False, box=None, padding=(0, 2))
        error_table.add_column(style="red")
        error_table.add_row("❌ LLM Analysis Failed")
        error_table.add_row(f"Error: {error_msg}")
        error_table.add_row("Please check your OpenAI API key and ensure you have sufficient credits.")
        return Group(error_table)

    def _create_summary_panel(self, summary_table: Table) -> Panel:
        """Creates a panel displaying the action summary."""
        return Panel(
            summary_table,
            title="[bold]Output Review[/]",
            border_style="green",
            padding=(1, 4)
        )

    def _create_action_panel(self, filtered_results: List[AnalysisResult]) -> Panel:
        """Creates a panel displaying actions for high-risk comments."""
        high_risk_comments = [
            r for r in filtered_results
            if r.pii_risk_score > 0.5 or (
                r.llm_findings and r.llm_findings.get('has_pii', False)
            )
        ]
        comment_ids = [r.comment_id for r in high_risk_comments]
        if comment_ids:
            action_text = Group(
                Text("Ready-to-use commands for high-risk comments:", style="bold yellow"),
                Text(f"Delete comments:\nreddacted delete {' '.join(comment_ids)}", style="italic red"),
                Text(f"\nReddact (edit) comments:\nreddacted update {' '.join(comment_ids)}", style="italic blue")
            )
        else:
            action_text = Text("No high-risk comments found.", style="green")
        return Panel(
            action_text,
            border_style="yellow",
            title="[bold]Actions[/]"
        )

    def _print_completion_message(
        self,
        filename: str,
        comments: List[Dict[str, Any]],
        results: List[AnalysisResult],
        progress: Progress
    ) -> None:
        """Prints completion message with file info and action panel."""
        high_risk_comments = [
            r for r in results if r.pii_risk_score > 0.5 or
            (r.llm_findings and r.llm_findings.get('has_pii', False))
        ]
        comment_ids = [r.comment_id for r in high_risk_comments]
        completion_panel = Panel(
            Text.assemble(
                ("📄 Report saved to ", "bold blue"), (f"{filename}\n", "bold yellow"),
                ("🗒️  Total comments: ", "bold blue"), (f"{len(comments)}\n", "bold cyan"),
                ("🔐 PII detected in: ", "bold blue"), (f"{self.total_pii_comments} ", "bold red"),
                (f"({self.total_pii_comments/len(comments):.1%})\n", "dim"),
                ("🤖 LLM findings in: ", "bold blue"), (f"{self.total_llm_pii_comments} ", "bold magenta"),
                (f"({self.total_llm_pii_comments/len(comments):.1%})", "dim")
            ),
            title="[bold green]Analysis Complete[/]",
            border_style="green",
            padding=(1, 4)
        )
        if comment_ids:
            actions_panel = self._create_action_panel(results)
            progress.console.print(Group(completion_panel, actions_panel))
        else:
            progress.console.print(completion_panel)
