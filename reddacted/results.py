#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Dict, Any, Optional

from rich.columns import Columns
from rich.console import Group
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from reddacted.utils.logging import get_logger, with_logging
from reddacted.utils.report import (
    generate_analysis_report,
    should_show_result,
)
from reddacted.utils.tables import TableFormatter
from reddacted.utils.panels import PanelFormatter
from reddacted.utils.analysis import AnalysisResult
from reddacted.textual_ui import show_results


logger = get_logger(__name__)


class ResultsFormatter(TableFormatter, PanelFormatter):
    """Handles formatting and display of analysis results."""

    def __init__(self):
        TableFormatter.__init__(self)
        PanelFormatter.__init__(self)
        self.logger = get_logger(__name__)
        self.total_pii_comments = 0
        self.total_llm_pii_comments = 0

    @with_logging(logger)
    def create_progress(self) -> Progress:
        """Creates a unified progress context manager."""
        if not hasattr(self, "_progress"):
            self._progress = Progress(
                SpinnerColumn(spinner_name="dots"),
                TextColumn("[bold blue]{task.description}"),
                TimeElapsedColumn(),
                transient=True,
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
        overall_sentiment: str,
    ) -> None:
        """Outputs a file containing a detailed sentiment and PII analysis per comment."""
        progress = self.create_progress()
        with progress:
            progress_task = progress.add_task(
                "📝 Generating analysis report...", total=len(comments)
            )
            try:
                stats = generate_analysis_report(
                    filename=filename,
                    comments=comments,
                    url=url,
                    results=results,
                    overall_score=overall_score,
                    overall_sentiment=overall_sentiment,
                    pii_only=getattr(self, "pii_only", False),
                )
                self.total_pii_comments = stats["total_pii_comments"]
                self.total_llm_pii_comments = stats["total_llm_pii_comments"]
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
        sort: str,
    ) -> None:
        """Prints the active configuration."""
        progress = self.create_progress()
        with progress:
            progress.console.print("\n[bold cyan]Active Configuration[/]")
            features_panel = self.create_features_panel(
                auth_enabled, pii_enabled, llm_config, pii_only, limit, sort
            )
            panels = [features_panel]
            if auth_enabled:
                auth_panel = self.create_auth_panel()
                panels.append(auth_panel)
            progress.console.print(Columns(panels))

    @with_logging(logger)
    def print_comments(
        self,
        comments: List[Dict[str, Any]],
        url: str,
        results: List[AnalysisResult],
        overall_score: float,
        overall_sentiment: str,
    ) -> None:
        """Prints out analysis of user comments using Textual UI."""
        filtered_results = [
            r for r in results if should_show_result(r, getattr(self, "pii_only", False))
        ]
        if not filtered_results and getattr(self, "pii_only", False):
            self.logger.info("No comments with high PII risk found.")
            print("No comments with high PII risk found.")
            return

        # Show interactive results view
        show_results(
            url=url,
            comments=comments,
            results=filtered_results,
            overall_score=overall_score,
            overall_sentiment=overall_sentiment,
        )

    def _print_completion_message(
        self,
        filename: str,
        comments: List[Dict[str, Any]],
        results: List[AnalysisResult],
        progress: Progress,
    ) -> None:
        """Prints completion message with file info and action panel."""
        high_risk_comments = [
            r
            for r in results
            if r.pii_risk_score > 0.5 or (r.llm_findings and r.llm_findings.get("has_pii", False))
        ]
        comment_ids = [r.comment_id for r in high_risk_comments]
        completion_panel = self.create_completion_panel(
            filename, len(comments), self.total_pii_comments, self.total_llm_pii_comments
        )
        if comment_ids:
            actions_panel = self.create_action_panel(results)
            progress.console.print(Group(completion_panel, actions_panel))
        else:
            progress.console.print(completion_panel)
