from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Center, Vertical
from textual.widgets import Button, Static, Label
from textual.binding import Binding
from textual import message

from reddacted.api.reddit import Reddit


class CommentActionScreen(Screen):
    """Screen for confirming and executing comment actions."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=True),
    ]

    def __init__(self, comment_id: str, action: str):
        """Initialize the action screen.

        Args:
            comment_id: The ID of the comment to act on
            action: Either 'edit' or 'delete'
        """
        super().__init__()
        self.comment_id = comment_id
        self.action = action
        self.api = Reddit()

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        action_text = "edit" if self.action == "edit" else "delete"
        with Center():
            with Vertical():
                yield Label(f"Are you sure you want to {action_text} comment {self.comment_id}?")
                with Center():
                    yield Button("Confirm", variant="error", id="confirm")
                    yield Button("Cancel", variant="primary", id="cancel")
                yield Static("", id="status")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "cancel":
            self.app.pop_screen()
        elif event.button.id == "confirm":
            self._execute_action()

    def action_cancel(self) -> None:
        """Handle escape key."""
        self.app.pop_screen()

    def _execute_action(self) -> None:
        """Execute the requested action."""
        status = self.query_one("#status", Static)
        try:
            if self.action == "edit":
                result = self.api.update_comments([self.comment_id])
                action_text = "edited"
            else:  # delete
                result = self.api.delete_comments([self.comment_id])
                action_text = "deleted"

            if result["success"] > 0:
                # Notify parent to refresh
                self.app.post_message(self.ActionCompleted(self.comment_id, self.action))

                # Close the screen after a short delay to show success
                def close_screen():
                    self.app.pop_screen()

                self.set_timer(0.5, close_screen)
                status.update(f"✅ Successfully {action_text} comment")
            else:
                status.update(f"❌ Failed to {self.action} comment")
        except Exception as e:
            status.update(f"❌ Error: {str(e)}")

    class ActionCompleted(message.Message):
        """Message sent when action is completed successfully."""

        def __init__(self, comment_id: str, action: str):
            self.comment_id = comment_id
            self.action = action
            super().__init__()

        @property
        def message_type(self) -> str:
            return "action_completed"
