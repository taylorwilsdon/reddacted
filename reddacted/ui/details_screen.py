from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Vertical, ScrollableContainer, Horizontal, Grid
from textual.widgets import Static, Label, Markdown, Button
from textual.binding import Binding
from textual import message

from typing import List, Dict, Any, Optional

from reddacted.utils.report import format_llm_detail
from reddacted.ui.comment_actions import CommentActionScreen


class DetailsScreen(Screen):
    """Screen for displaying detailed PII analysis for a comment."""

    BINDINGS = [
        Binding("escape", "go_back", "Return to Results", show=True),
        Binding("b", "go_back", "Back", show=True),
        Binding("e", "edit_comment", "Edit Comment", show=True),
        Binding("d", "delete_comment", "Delete Comment", show=True),
    ]

    def __init__(self, result):
        """Initialize the details screen.
        
        Args:
            result: The AnalysisResult object containing the comment data
        """
        super().__init__()
        self.result = result

    def compose(self) -> ComposeResult:
        """Create child widgets for the screen."""
        with ScrollableContainer():
            with Vertical(classes="details-container"):
                yield Label("Comment Details", classes="details-title")
                
                # Header grid with key information
                with Grid(id="details-header-grid"):
                    # First row: ID and actions
                    yield Static(f"ID: {self.result.comment_id}", classes="details-id")
                    yield Static(f"Risk Score: {self.result.pii_risk_score:.0%}",
                                classes=f"details-risk-{self._get_risk_class()}")
                    
                    # Second row: Sentiment and votes
                    yield Static(
                        f"Sentiment: {self.result.sentiment_emoji} {self.result.sentiment_score:.2f}",
                        classes="details-sentiment"
                    )
                    yield Static(
                        f"Votes: ⬆️ {self.result.upvotes} ⬇️ {self.result.downvotes}",
                        classes="details-votes"
                    )
                
                # Back and Comment Actions section
                with Horizontal(classes="details-actions"):
                    yield Button("← Back to Results", variant="default", id="back-btn")
                    yield Button("Edit Comment", variant="primary", id="edit-btn")
                    yield Button("Delete Comment", variant="error", id="delete-btn")
                
                # Comment text section
                yield Label("Comment Text", classes="subsection-header")
                yield Markdown(self.result.text, classes="details-text")
                
                # Pattern-based PII section
                if self.result.pii_matches:
                    yield Label("Pattern-based PII Detected", classes="section-header")
                    for pii in self.result.pii_matches:
                        yield Static(
                            f"• {pii.type} (confidence: {pii.confidence:.2f})",
                            classes="details-pii-item"
                        )
                
                # LLM analysis section
                if self.result.llm_findings:
                    yield Label("LLM Privacy Analysis", classes="section-header")
                    yield Static(f"Risk Score: {self.result.llm_risk_score:.2f}", classes="details-llm-risk")
                    
                    findings = self.result.llm_findings
                    if isinstance(findings, dict):
                        has_pii = findings.get('has_pii', False)
                        yield Static(
                            f"PII Detected: {'Yes' if has_pii else 'No'}",
                            classes=f"details-has-pii-{'yes' if has_pii else 'no'}"
                        )
                        
                        if details := findings.get('details'):
                            yield Label("Findings:", classes="subsection-header")
                            for detail in details:
                                yield Static(f"• {format_llm_detail(detail)}", classes="details-llm-item")
                        
                        if reasoning := findings.get('reasoning'):
                            yield Label("Reasoning:", classes="subsection-header")
                            yield Markdown(reasoning, classes="details-reasoning")

    def _get_risk_class(self) -> str:
        """Get risk class based on PII risk score."""
        if self.result.pii_risk_score > 0.7:
            return "high"
        elif self.result.pii_risk_score > 0.4:
            return "medium"
        else:
            return "low"
            
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        button_id = event.button.id
        if button_id == "back-btn":
            self.action_go_back()
        elif button_id == "edit-btn":
            self.action_edit_comment()
        elif button_id == "delete-btn":
            self.action_delete_comment()
    
    def on_comment_action_screen_action_completed(self, event: message.Message) -> None:
        """Handle action_completed events from CommentActionScreen."""
        action_type = "edited" if event.action == "edit" else "deleted"
        self.app.notify(f"Comment {self.result.comment_id} successfully {action_type}")
        # Return to the main screen after an action
        self.app.pop_screen()
    
    def action_edit_comment(self) -> None:
        """Handle editing the current comment."""
        self.app.push_screen(CommentActionScreen(self.result.comment_id, "edit"))
    
    def action_delete_comment(self) -> None:
        """Handle deleting the current comment."""
        self.app.push_screen(CommentActionScreen(self.result.comment_id, "delete"))

    def action_go_back(self) -> None:
        """Return to the results screen."""
        self.app.pop_screen()
    
    class DetailActionComplete(message.Message):
        """Message sent when returning to main screen."""
        def __init__(self, comment_id: str, action: str = None):
            self.comment_id = comment_id
            self.action = action
            super().__init__()