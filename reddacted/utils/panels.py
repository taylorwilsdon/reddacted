import os
from typing import List, Dict, Any, Optional, Tuple
from itertools import zip_longest

from rich.panel import Panel
from rich.columns import Columns
from rich.console import Group
from rich.text import Text
from rich.table import Table

from reddacted.utils.analysis import AnalysisResult
from reddacted.utils.base import BaseFormatter
from reddacted.utils.log_handler import get_logger, with_logging
from reddacted.utils.tables import TableFormatter
from reddacted.utils.report import format_llm_detail


class PanelFormatter(BaseFormatter):
    """Handles creation and formatting of Rich panels for the Reddit comment analysis UI."""

    def __init__(self):
        super().__init__()
        # Logger not currently used in this class
        self.table_formatter = TableFormatter()

    def create_features_panel(
        self,
        auth_enabled: bool,
        pii_enabled: bool,
        llm_config: Optional[Dict[str, Any]],
        pii_only: bool,
        limit: int,
        sort: str,
        use_random_string: bool = False,
    ) -> Panel:
        """Creates a panel displaying the features configuration."""
        # Create a table with two columns
        features_table = Table(
            show_header=False, box=None, padding=(0, 2), collapse_padding=True, expand=True
        )
        features_table.add_column("Left", ratio=1, justify="left")
        features_table.add_column("Right", ratio=1, justify="left")

        # Define all config items
        config_items = [
            ("🔐 Authentication", self._format_status(auth_enabled)),
            ("🔍 PII Detection", self._format_status(pii_enabled)),
            (
                "🤖 LLM Analysis",
                (
                    Text(llm_config["model"], style="green") # Display model name if available
                    if llm_config and llm_config.get("model")
                    else Text("Not Selected", style="yellow") # Indicate if URL/Key provided but no model
                    if llm_config
                    else self._format_status(False) # Disabled if no LLM config at all
                ),
            ),
            ("🎯 PII-Only Filter", self._format_status(pii_only, "Active", "Inactive")),
            ("🎲 Random String", self._format_status(use_random_string, "Enabled", "Disabled")),
            ("📊 Comment Limit", Text(f"{limit}" if limit else "Unlimited", style="cyan")),
            ("📑 Sort Preference", Text(f"{sort}" if sort else "New", style="cyan")),
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
            expand=True,
        )

    def create_auth_panel(self, config: Dict[str, Any]) -> Panel:
        """Creates a panel displaying the authentication status and values based on config and environment."""
        auth_enabled = config.get("enable_auth", False)
        auth_texts = []

        # Determine status based on config first, then environment
        username_config = config.get("reddit_username")
        client_id_config = config.get("reddit_client_id")
        username_env = os.environ.get("REDDIT_USERNAME")
        client_id_env = os.environ.get("REDDIT_CLIENT_ID")

        # --- Username Status ---
        username_value = None
        username_style = "red"
        username_source = ""

        if auth_enabled and username_config:
            username_value = username_config
            username_style = "green"
            username_source = " (Config)"
        elif username_env:
            username_value = username_env
            username_style = "blue"
            username_source = " (Env Var)"

        if username_value:
             auth_texts.append(
                 Text.assemble("REDDIT_USERNAME: ", (username_value, username_style), username_source)
             )
        else:
             auth_texts.append(Text("REDDIT_USERNAME: Not Set", style="red"))


        # --- Client ID Status ---
        client_id_value = None
        client_id_style = "red"
        client_id_source = ""

        if auth_enabled and client_id_config:
            client_id_value = client_id_config
            client_id_style = "green"
            client_id_source = " (Config)"
        elif client_id_env:
            client_id_value = client_id_env
            client_id_style = "blue"
            client_id_source = " (Env Var)"

        if client_id_value:
             # Display only first/last few chars of client_id for brevity/security if desired
             # display_client_id = f"{client_id_value[:4]}...{client_id_value[-4:]}" if len(client_id_value) > 8 else client_id_value
             display_client_id = client_id_value # Show full ID for now
             auth_texts.append(
                 Text.assemble("REDDIT_CLIENT_ID: ", (display_client_id, client_id_style), client_id_source)
             )
        else:
             auth_texts.append(Text("REDDIT_CLIENT_ID: Not Set", style="red"))

        # Note: We don't display password or secret for security

        return Panel(Group(*auth_texts), title="[bold]Auth Status[/]", border_style="yellow")

    def create_stats_panel(
        self, url: str, total_comments: int, score: float, sentiment: str
    ) -> Panel:
        """Creates a panel displaying the sentiment analysis summary."""
        # Create metrics table
        metrics_table = Table(show_header=False, box=None, padding=(0, 2), collapse_padding=True)
        metrics_table.add_column("Icon", justify="right", style="bold")
        metrics_table.add_column("Label", style="bold")
        metrics_table.add_column("Value", justify="left")

        # Add rows with proper spacing and alignment
        metrics_table.add_row(
            "🔍",
            "Analysis for:",
            (
                f"[link=https://reddit.com/u/{url}]{url}[/]"
                if url.startswith("u/")
                else f"[cyan]{url}[/]"
            ),
        )
        metrics_table.add_row("📊", "Comments analyzed:", f"[cyan bold]{total_comments:>4}[/]")
        metrics_table.add_row(
            "🎭", "Overall Sentiment:", f"[cyan bold]{score:>6.2f}[/] {sentiment}"
        )

        return Panel(
            metrics_table,
            title="[bold]Sentiment Analysis Summary[/]",
            border_style="blue",
            padding=(1, 1),
        )

    def create_comment_panel(self, result: AnalysisResult, index: int) -> Panel:
        """Creates a panel for a single comment."""
        sub_panels = [self.create_basic_info_panel(result)]
        if result.pii_matches:
            sub_panels.append(self.create_pii_panel(result))
        if result.llm_findings:
            sub_panels.append(self.create_llm_panel(result))
        return Panel(Columns(sub_panels), title=f"[bold]Comment {index}[/]", border_style="cyan")

    def create_basic_info_panel(self, result: AnalysisResult) -> Panel:
        """Creates a panel displaying basic comment information."""
        # Create metrics table
        metrics_table = Table(show_header=False, box=None, padding=(0, 2), collapse_padding=True)
        metrics_table.add_column("Icon", justify="right", style="bold")
        metrics_table.add_column("Label", style="bold")
        metrics_table.add_column("Value", justify="left")

        # Risk score styling
        risk_score_style = "red bold" if result.pii_risk_score > 0.5 else "green bold"

        # Add rows with proper spacing and alignment
        metrics_table.add_row(
            "🎭",
            "Sentiment:",
            f"[cyan bold]{result.sentiment_score:>6.2f}[/] {result.sentiment_emoji}",
        )
        metrics_table.add_row(
            "🔒", "Privacy Risk:", f"[{risk_score_style}]{result.pii_risk_score:>6.2f}[/]"
        )
        # Format votes based on whether they're positive or negative
        vote_display = (
            f"[green]⬆️ {result.upvotes:>3}[/]"
            if result.upvotes > result.downvotes
            else (
                f"[red]⬇️ {result.downvotes:>3}[/]"
                if result.downvotes > result.upvotes
                else f"[dim]0[/]"
            )
        )

        # Combine comment text and metrics
        basic_info = Group(
            Text(result.text, style="white"), Text("─" * 50, style="dim"), metrics_table
        )

        return Panel(basic_info, title="[bold]Basic Info[/]", border_style="blue", padding=(1, 1))

    def create_pii_panel(self, result: AnalysisResult) -> Panel:
        """Creates a panel displaying pattern-based PII matches."""
        pii_contents = [
            Text(f"• {pii.type} (confidence: {pii.confidence:.2f})", style="cyan")
            for pii in result.pii_matches
        ]
        return Panel(
            Group(*pii_contents), title="[bold]Pattern-based PII Detected[/]", border_style="yellow"
        )

    def create_llm_panel(self, result: AnalysisResult) -> Panel:
        """Creates a panel displaying LLM analysis findings."""
        # Create metrics table similar to basic info panel
        metrics_table = Table(show_header=False, box=None, padding=(0, 2), collapse_padding=True)
        metrics_table.add_column("Icon", justify="right", style="bold")
        metrics_table.add_column("Label", style="bold")
        metrics_table.add_column("Value", justify="left")

        if isinstance(result.llm_findings, dict) and "error" in result.llm_findings:
            error_group = self.create_llm_error_content(result.llm_findings["error"])
            return Panel(error_group, title="[bold]LLM Analysis[/]", border_style="red")

        # Risk score styling
        risk_style = "red bold" if result.llm_risk_score > 0.5 else "green bold"
        pii_style = "red bold" if result.llm_findings.get("has_pii", False) else "green bold"

        # Add main metrics rows
        metrics_table.add_row(
            "🎯", "Risk Score:", f"[{risk_style}]{result.llm_risk_score:>6.2f}[/]"
        )
        metrics_table.add_row(
            "🔍",
            "PII Detected:",
            f"[{pii_style}]{'Yes' if result.llm_findings.get('has_pii') else 'No':>6}[/]",
        )

        # Create content groups
        content_groups = [metrics_table]

        # Add findings if present
        if details := result.llm_findings.get("details"):
            content_groups.extend(
                [
                    Text("\n📋 Findings:", style="bold"),
                    *[Text(f"  • {format_llm_detail(detail)}", style="cyan") for detail in details],
                ]
            )

        # Add risk factors if present
        if risk_factors := result.llm_findings.get("risk_factors"):
            content_groups.extend(
                [
                    Text("\n⚠️ Risk Factors:", style="bold"),
                    *[Text(f"  • {factor}", style="yellow") for factor in risk_factors],
                ]
            )

        return Panel(
            Group(*content_groups),
            title="[bold]LLM Analysis[/]",
            border_style="magenta",
            padding=(1, 1),
        )

    def create_llm_error_content(self, error_msg: str) -> Group:
        """Creates content for LLM analysis errors."""
        error_table = Table(show_header=False, box=None, padding=(0, 2))
        error_table.add_column(style="red")
        error_table.add_row("❌ LLM Analysis Failed")
        error_table.add_row(f"Error: {error_msg}")
        error_table.add_row(
            "Please check your OpenAI API key and ensure you have sufficient credits."
        )
        return Group(error_table)

    def create_summary_panel(self, summary_table: Table) -> Panel:
        """Creates a panel displaying the action summary."""
        return Panel(
            summary_table, title="[bold]Output Review[/]", border_style="green", padding=(1, 4)
        )

    def create_action_panel(self, filtered_results: List[AnalysisResult], use_random_string: bool = False) -> Panel:
        """Creates a panel displaying actions for high-risk comments."""
        high_risk_comments = [
            r
            for r in filtered_results
            if r.pii_risk_score > 0.5 or (r.llm_findings and r.llm_findings.get("has_pii", False))
        ]
        comment_ids = [r.comment_id for r in high_risk_comments]
        if comment_ids:
            action_text = Group(
                Text("Ready-to-use commands for high-risk comments:", style="bold yellow"),
                Text(
                    f"Delete comments:\nreddacted delete {' '.join(comment_ids)}",
                    style="italic red",
                ),
                Text(
                    f"\nReddact (edit) comments:" +
                    (f"\nreddacted update {' '.join(comment_ids)} --use-random-string" if use_random_string else
                     f"\nreddacted update {' '.join(comment_ids)}"),
                    style="italic blue",
                ),
            )
        else:
            action_text = Text("No high-risk comments found.", style="green")
        return Panel(action_text, border_style="yellow", title="[bold]Actions[/]")

    def create_completion_panel(
        self,
        filename: str,
        total_comments: int,
        total_pii_comments: int,
        total_llm_pii_comments: int,
    ) -> Panel:
        """Creates a panel for the completion message with file info."""
        return Panel(
            Text.assemble(
                ("📄 Report saved to ", "bold blue"),
                (f"{filename}\n", "bold yellow"),
                ("🗒️  Total comments: ", "bold blue"),
                (f"{total_comments}\n", "bold cyan"),
                ("🔐 PII detected in: ", "bold blue"),
                (f"{total_pii_comments} ", "bold red"),
                (f"({total_pii_comments/total_comments:.1%})\n", "dim"),
                ("🤖 LLM findings in: ", "bold blue"),
                (f"{total_llm_pii_comments} ", "bold magenta"),
                (f"({total_llm_pii_comments/total_comments:.1%})", "dim"),
            ),
            title="[bold green]Analysis Complete[/]",
            border_style="green",
            padding=(1, 4),
        )
