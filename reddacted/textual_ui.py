from textual.app import App, ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import DataTable, Header, Footer, Static
from textual.binding import Binding
from textual.reactive import reactive
from textual import events, message

from rich.text import Text
from typing import List, Optional

from reddacted.utils.analysis import AnalysisResult
from reddacted.ui.comment_actions import CommentActionScreen
from reddacted.styles import TEXTUAL_CSS

class ResultsSummary(DataTable):
    """An interactive data table showing analysis results."""
    
    def __init__(self, results: List[AnalysisResult]):
        super().__init__()
        self.results = results
        
    def on_mount(self) -> None:
        """Set up the table when mounted."""
        # Add columns
        self.add_columns(
            "Risk",
            "Sentiment",
            "Comment Preview",
            "Votes",
            "ID"
        )
        
        # Add rows from results
        for result in self.results:
            # Format risk score with color based on value
            risk_score = f"{result.pii_risk_score:.0%}"
            risk_style = (
                "red" if result.pii_risk_score > 0.7
                else "yellow" if result.pii_risk_score > 0.4
                else "green"
            )
            risk_cell = Text(risk_score, style=risk_style)
            
            # Format sentiment with emoji
            sentiment = Text(f"{result.sentiment_emoji} {result.sentiment_score:.2f}")
            
            # Format comment preview with link
            preview = (result.text[:67] + "...") if len(result.text) > 70 else result.text
            preview_cell = Text(preview, style="link blue")
            
            # Format votes
            vote_style = (
                "green" if result.upvotes > result.downvotes
                else "red" if result.downvotes > result.upvotes
                else "dim"
            )
            vote_display = Text(
                f"⬆️ {result.upvotes:>3}" if result.upvotes > result.downvotes
                else f"⬇️ {result.downvotes:>3}" if result.downvotes > result.upvotes
                else "0",
                style=vote_style
            )
            
            self.add_row(
                risk_cell,
                sentiment,
                preview_cell,
                vote_display,
                result.comment_id
            )

class StatsDisplay(Static):
    """Displays overall statistics."""
    
    def __init__(self, url: str, comment_count: int, overall_score: float, overall_sentiment: str):
        super().__init__()
        self.url = url
        self.comment_count = comment_count
        self.overall_score = overall_score
        self.overall_sentiment = overall_sentiment
    
    def compose(self) -> ComposeResult:
        yield Static(f"📊 Analysis Results for: {self.url}")
        yield Static(f"💬 Total Comments: {self.comment_count}")
        yield Static(f"📈 Overall Score: {self.overall_score:.2f}")
        yield Static(f"🎭 Overall Sentiment: {self.overall_sentiment}")

class TextualResultsView(App):
    """Main Textual app for displaying analysis results."""
    
    CSS = TEXTUAL_CSS
    
    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("up", "cursor_up", "Move up", show=False),
        Binding("down", "cursor_down", "Move down", show=False),
        Binding("enter", "view_details", "View Details", show=True),
        Binding("e", "edit_comment", "Edit Comment", show=True),
        Binding("d", "delete_comment", "Delete Comment", show=True),
    ]

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
        """Handle completion of comment actions.
        
        Args:
            event: The action completed message containing comment_id and action
        """
        self.log(f"Received action_completed message: {event.comment_id} - {event.action}")
        table = self.query_one(ResultsSummary)
        
        # Find the row index and result for the affected comment
        row_index = None
        result = None
        for i, r in enumerate(self.results):
            if r.comment_id == event.comment_id:
                row_index = i
                result = r
                break
                
        if row_index is None:
            self.log(f"Comment {event.comment_id} not found in results")
            return
            
        if event.action == "delete":
            self.log(f"Removing comment {event.comment_id} from table and results")
            # Remove from table
            table.remove_row(row_index)
            # Remove from results
            self.results.pop(row_index)
            
        elif event.action == "edit":
            self.log(f"Updating comment {event.comment_id} in table")
            # Update the result text without LLM reprocessing
            result.text = "r/reddacted"
            
            # Update just this row in the table
            table.update_cell(
                row_index, 2,  # Column index for text
                Text("r/reddacted", style="link blue")
            )
            
        self.log("Table update complete")
        # Remove the comment from the results if it was deleted
        if message.action == "delete":
            self.results = [r for r in self.results if r.comment_id != message.comment_id]
            # Refresh the table
            table = self.query_one(ResultsSummary)
            table.clear()
            table.add_rows([
                [Text(f"{r.pii_risk_score:.0%}", style="red" if r.pii_risk_score > 0.7 else "yellow" if r.pii_risk_score > 0.4 else "green"),
                 Text(f"{r.sentiment_emoji} {r.sentiment_score:.2f}"),
                 Text((r.text[:67] + "...") if len(r.text) > 70 else r.text, style="link blue"),
                 Text(f"⬆️ {r.upvotes:>3}" if r.upvotes > r.downvotes else f"⬇️ {r.downvotes:>3}" if r.downvotes > r.upvotes else "0",
                      style="green" if r.upvotes > r.downvotes else "red" if r.downvotes > r.upvotes else "dim"),
                 r.comment_id]
                for r in self.results
            ])
    
    def __init__(
        self,
        url: str,
        comments: List[dict],
        results: List[AnalysisResult],
        overall_score: float,
        overall_sentiment: str
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
            StatsDisplay(
                self.url,
                len(self.comments),
                self.overall_score,
                self.overall_sentiment
            ),
            ResultsSummary(self.results),
        )
        yield Footer()
    
    def action_view_details(self) -> None:
        """Handle viewing details of selected row."""
        table = self.query_one(ResultsSummary)
        if table.cursor_row is not None:
            # Get the comment ID from the selected row
            comment_id = table.get_row_at(table.cursor_row)[-1]
            # Find the corresponding result
            result = next((r for r in self.results if r.comment_id == comment_id), None)
            if result:
                # TODO: Show detailed view of the selected comment
                self.notify(f"Viewing details for comment {comment_id}")

def show_results(
    url: str,
    comments: List[dict],
    results: List[AnalysisResult],
    overall_score: float,
    overall_sentiment: str
) -> None:
    """Display results using the Textual UI."""
    app = TextualResultsView(
        url=url,
        comments=comments,
        results=results,
        overall_score=overall_score,
        overall_sentiment=overall_sentiment
    )
    app.run()