import sys
import inspect
import traceback
import logging
from typing import Optional, Type
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.traceback import Traceback

console = Console()
logger = logging.getLogger(__name__)


def format_exception(exc: Exception, include_trace: bool = False) -> str:
    """Format exception with source location and clean message

    Args:
        exc: The exception to format
        include_trace: Whether to include full traceback

    Returns:
        Formatted error message with location and optional trace

    Raises:
        TypeError: If exc is not an Exception instance
    """
    if not isinstance(exc, Exception):
        raise TypeError("exc must be an Exception instance")

    # Get the exception chain
    exc_chain = []
    current = exc
    while current:
        exc_chain.append(current)
        current = current.__cause__ or current.__context__

    # Get traceback information
    tb = getattr(exc, "__traceback__", None) or sys.exc_info()[2]
    if tb is None:
        # Fallback location info if no traceback available
        module_name = "unknown"
        func_name = "unknown"
        line_no = 0
    else:
        # Find the deepest relevant frame
        while tb and tb.tb_next:
            tb = tb.tb_next

        try:
            frame = tb.tb_frame
            func_name = frame.f_code.co_name
            line_no = tb.tb_lineno
            module = inspect.getmodule(frame)
            module_name = module.__name__ if module else "unknown"
        except (AttributeError, ValueError):
            # Fallback if frame access fails
            module_name = "unknown"
            func_name = "unknown"
            line_no = 0

    # Build the error message
    messages = []
    for i, e in enumerate(reversed(exc_chain)):
        try:
            error_type = e.__class__.__name__
            error_msg = str(e)
        except Exception:
            error_type = "UnknownError"
            error_msg = "Failed to format error message"

        if i == 0:
            messages.append(f"[bold red]{error_type}[/]: {error_msg}")
        else:
            messages.append(f"[dim]Caused by: {error_type}: {error_msg}[/]")

    location = f"[dim]Location: {module_name}.{func_name}(), line {line_no}[/]"

    if include_trace and tb is not None:
        try:
            trace = Traceback.extract(exc_type=type(exc), exc_value=exc, traceback=tb)
            return "\n".join(messages + [location, "", str(trace)])
        except Exception:
            # Fallback if traceback extraction fails
            return "\n".join(messages + [location, "", "Failed to extract traceback"])

    return "\n".join(messages + [location])


def handle_exception(exc: Exception, context: Optional[str] = None, debug: bool = False) -> None:
    """Print a formatted exception with optional context and debugging

    Args:
        exc: The exception to handle
        context: Optional context about what was happening
        debug: Whether to include full traceback
    """
    # Log the full exception for debugging
    logger.error(f"Error in {context or 'unknown context'}", exc_info=exc if debug else False)

    # Format the error message
    error_msg = format_exception(exc, include_trace=debug)
    if context:
        error_msg = f"{context}\n{error_msg}"

    # Print to console
    console.print(
        Panel(Text.from_markup(error_msg), title="[bold red]Error[/]", border_style="red")
    )
