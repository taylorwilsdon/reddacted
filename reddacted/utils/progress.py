from typing import Optional

from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from reddacted.utils.logging import get_logger, with_logging
from reddacted.utils.base import BaseFormatter

logger = get_logger(__name__) # Define logger at module level

class ProgressManager(BaseFormatter):
    """Manages progress bars and indicators."""

    def __init__(self):
        super().__init__()
        self._progress: Optional[Progress] = None

    @with_logging(logger) # Use the module-level logger variable
    def create_progress(self) -> Progress:
        """Creates a unified progress context manager."""
        if not hasattr(self, "_progress") or self._progress is None:
            self._progress = Progress(
                SpinnerColumn(spinner_name="dots"),
                TextColumn("[bold blue]{task.description}"),
                TimeElapsedColumn(),
                transient=True,
            )
        return self._progress
