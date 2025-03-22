from typing import List, Dict, Any, Optional
from rich.text import Text

from reddacted.utils.logging import get_logger, with_logging


class BaseFormatter:
    """Base class for formatters with shared utilities."""

    def __init__(self):
        self.logger = get_logger(__name__)

    def _get_risk_style(self, score: float) -> str:
        """Determines text style based on risk score."""
        if score > 0.5:
            return "red"
        elif score > 0.2:
            return "yellow"
        else:
            return "green"

    def _format_status(
        self, enabled: bool, true_text: str = "Enabled", false_text: str = "Disabled"
    ) -> Text:
        """Formats a status text based on a boolean value."""
        return Text(true_text if enabled else false_text, style="green" if enabled else "red")
