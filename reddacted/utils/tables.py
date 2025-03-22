from typing import Any, Dict, List, Optional
from itertools import zip_longest

from rich.table import Table
from rich.text import Text

from reddacted.utils.base import BaseFormatter
from reddacted.utils.analysis import AnalysisResult


class TableFormatter(BaseFormatter):
    """Handles creation and formatting of tables."""

    def generate_summary_table(self, filtered_results: List[AnalysisResult]) -> Table:
        """Generates a summary table with selection indicators."""
        table = Table(header_style="bold magenta", box=None, padding=(0, 1), collapse_padding=True)
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

            vote_display = (
                f"[green]⬆️ {result.upvotes:>3}[/]"
                if result.upvotes > result.downvotes
                else (
                    f"[red]⬇️ {result.downvotes:>3}[/]"
                    if result.downvotes > result.upvotes
                    else f"[dim]0[/]"
                )
            )

            table.add_row(
                risk_text,
                Text(f"{result.sentiment_emoji} {result.sentiment_score:.2f}"),
                preview,
                vote_display,
                result.comment_id,
            )

        return table

    def create_features_table(
        self,
        auth_enabled: bool,
        pii_enabled: bool,
        llm_config: Optional[Dict[str, Any]],
        pii_only: bool,
        limit: int,
        sort: str,
    ) -> Table:
        """Creates a table displaying the features configuration."""
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
                    Text(llm_config["model"], style="green")
                    if llm_config
                    else self._format_status(False)
                ),
            ),
            ("🎯 PII-Only Filter", self._format_status(pii_only, "Active", "Inactive")),
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

        return features_table
