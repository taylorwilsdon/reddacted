from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import DataTable, Header, Footer, Static
from textual.binding import Binding
from textual import message

from rich.text import Text
from typing import List, Optional

from reddacted.utils.analysis import AnalysisResult
from reddacted.ui.comment_actions import CommentActionScreen
from reddacted.ui.details_screen import DetailsScreen
from reddacted.styles import TEXTUAL_CSS


class ResultsSummary(DataTable):
    """An interactive data table showing analysis results."""

    def __init__(self, results: List[AnalysisResult]):
        super().__init__()
        self.results = results

    def on_mount(self) -> None:
        """Set up the table when mounted."""
        # Add columns
        self.add_columns("Risk", "Sentiment", "Comment Preview", "Votes", "ID")

        # Add rows from results
        for result in self.results:
            # Format risk score with color based on value
            risk_score = f"{result.pii_risk_score:.0%}"
            risk_style = (
                "red"
                if result.pii_risk_score > 0.7
                else "yellow" if result.pii_risk_score > 0.4 else "green"
            )
            risk_cell = Text(risk_score, style=risk_style)

            # Format sentiment with emoji
            sentiment = Text(f"{result.sentiment_emoji} {result.sentiment_score:.2f}")

            # Format comment preview with link
            preview = (result.text[:67] + "...") if len(result.text) > 70 else result.text
            preview_cell = Text(preview, style="link blue")

            # Format votes
            vote_style = (
                "green"
                if result.upvotes > result.downvotes
                else "red" if result.downvotes > result.upvotes else "dim"
            )
            vote_display = Text(
                (
                    f"⬆️ {result.upvotes:>3}"
                    if result.upvotes > result.downvotes
                    else f"⬇️ {result.downvotes:>3}" if result.downvotes > result.upvotes else "0"
                ),
                style=vote_style,
            )

            self.add_row(risk_cell, sentiment, preview_cell, vote_display, result.comment_id)

    def on_data_table_row_selected(self) -> None:
        """Handle row selection by mouse click."""
        # Trigger the view details action in the parent application
        if self.cursor_row is not None:
            self.app.action_view_details()

    def on_data_table_cell_selected(self) -> None:
        """Handle cell selection."""
        if self.cursor_row is not None:
            self.app.action_view_details()


class StatsDisplay(Static):
    """Displays overall statistics."""

    def __init__(self, url: str, comment_count: int, overall_score: float, overall_sentiment: str):
        super().__init__()
        self.url = url
        self.comment_count = comment_count
        self.overall_score = overall_score
        self.overall_sentiment = overall_sentiment

    def compose(self) -> ComposeResult:
        stat1 = Static(f"📊 Analysis Results for: {self.url}")
        stat2 = Static(f"💬 Total Comments: {self.comment_count}")
        stat3 = Static(f"📈 Overall Score: {self.overall_score:.2f}")
        stat4 = Static(f"🎭 Overall Sentiment: {self.overall_sentiment}")

        # Add stats-text class to all stats
        for stat in [stat1, stat2, stat3, stat4]:
            stat.add_class("stats-text")
            yield stat


class TextualResultsView(App):
    """Main Textual app for displaying analysis results."""

    CSS = TEXTUAL_CSS
    title = "reddacted"

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("enter", "view_details", "View Details", show=True),
        Binding("e", "edit_comment", "Edit Comment", show=True),
        Binding("d", "delete_comment", "Delete Comment", show=True),
    ]

    def on_mount(self) -> None:
        self.title = "reddacted"  # This controls the main header title
        self.sub_title = "Analysis Results"  # Optional: Controls the subtitle

    def _get_selected_comment_id(self) -> Optional[str]:
        """Get the comment ID of the currently selected row."""
        table = self.query_one(ResultsSummary)
        if table.cursor_row is not None:
            return table.get_row_at(table.cursor_row)[-1]
        return None

    def action_edit_comment(self) -> None:
        """Handle editing the selected comment."""
        if comment_id := self._get_selected_comment_id():
            self.push_screen(CommentActionScreen(comment_id, "edit"))

    def action_delete_comment(self) -> None:
        """Handle deleting the selected comment."""
        if comment_id := self._get_selected_comment_id():
            self.push_screen(CommentActionScreen(comment_id, "delete"))

    def on_action_completed(self, event: message.Message) -> None:
        """Handle completion of comment actions."""
        table = self.query_one(ResultsSummary)

        # Find the row index and result for the affected comment
        for i, r in enumerate(self.results):
            if r.comment_id == event.comment_id:
                if event.action == "delete":
                    # Remove from table and results
                    table.remove_row(i)
                    self.results.pop(i)
                elif event.action == "edit":
                    # Update the result text
                    r.text = "r/reddacted"
                    # Update cell in table
                    table.update_cell(i, 2, Text("r/reddacted", style="link blue"))
                break

    def __init__(
        self,
        url: str,
        comments: List[dict],
        results: List[AnalysisResult],
        overall_score: float,
        overall_sentiment: str,
    ):
        super().__init__()
        self.url = url
        self.comments = comments
        self.results = results
        self.overall_score = overall_score
        self.overall_sentiment = overall_sentiment

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header(show_clock=True)
        yield Container(
            StatsDisplay(self.url, len(self.comments), self.overall_score, self.overall_sentiment),
            ResultsSummary(self.results),
        )
        yield Footer()

    def action_view_details(self) -> None:
        """Handle viewing details of selected row."""
        if comment_id := self._get_selected_comment_id():
            result = next((r for r in self.results if r.comment_id == comment_id), None)
            if result:
                self.push_screen(DetailsScreen(result))
            else:
                self.notify(f"No result found for comment ID: {comment_id}")
        else:
            self.notify("No comment ID found")


def show_results(
    url: str,
    comments: List[dict],
    results: List[AnalysisResult],
    overall_score: float,
    overall_sentiment: str,
) -> None:
    """Display results using the Textual UI."""
    app = TextualResultsView(
        url=url,
        comments=comments, # Pass original comments list
        results=results,
        overall_score=overall_score,
        overall_sentiment=overall_sentiment,
    )
    app.run()
