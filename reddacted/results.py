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
        self.use_random_string = False  # Default to False

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
    def print_config(self, config: Dict[str, Any]) -> None:
        """Prints the active configuration using the provided config dictionary."""
        progress = self.create_progress()

        # Extract values needed for panels from the config dict
        auth_enabled = config.get("enable_auth", False)
        pii_enabled = True # Assuming PII is always enabled for now
        pii_only = config.get("pii_only", False)
        limit_val = config.get("limit", 20)
        limit = None if limit_val == 0 else limit_val
        sort = config.get("sort", "new")
        use_random_string = config.get("use_random_string", False) # Get from config

        # Construct llm_config dict for the features panel if applicable
        llm_config = None
        if config.get("model"):
            llm_config = {
                "api_key": config.get("openai_key") if config.get("use_openai_api") else "sk-not-needed",
                "api_base": config.get("local_llm") if not config.get("use_openai_api") else "https://api.openai.com/v1",
                "model": config.get("model"),
            }
            # Adjust api_base for local LLM if needed (redundant with Sentiment.__init__ but safe)
            if not config.get("use_openai_api") and llm_config.get("api_base"):
                base_url = llm_config["api_base"].rstrip('/')
                if not base_url.endswith('/v1'):
                    llm_config["api_base"] = f"{base_url}/v1"
        elif config.get("openai_key") or config.get("local_llm"):
             llm_config = { # Handle case where URL/key provided but no model
                "api_key": config.get("openai_key") if config.get("use_openai_api") else "sk-not-needed",
                "api_base": config.get("local_llm") if not config.get("use_openai_api") else "https://api.openai.com/v1",
                "model": None,
            }
             if not config.get("use_openai_api") and llm_config.get("api_base"):
                base_url = llm_config["api_base"].rstrip('/')
                if not base_url.endswith('/v1'):
                    llm_config["api_base"] = f"{base_url}/v1"


        with progress:
            progress.console.print("\n[bold cyan]Active Configuration[/]")
            features_panel = self.create_features_panel(
                auth_enabled, pii_enabled, llm_config, pii_only, limit, sort,
                use_random_string=use_random_string # Use value from config
            )
            panels = [features_panel]
            # Pass the full config to create_auth_panel
            auth_panel = self.create_auth_panel(config)
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
            use_random_string=getattr(self, "use_random_string", False),
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
            actions_panel = self.create_action_panel(results, use_random_string=getattr(self, "use_random_string", False))
            progress.console.print(Group(completion_panel, actions_panel))
        else:
            progress.console.print(completion_panel)
