# reddacted/utils/log_handler.py
import logging
import sys
import inspect
import traceback
from functools import wraps
from typing import Callable, Any, Optional, Union, Dict, TypeVar, Type

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.traceback import Traceback

# --- Type Definitions ---
T = TypeVar("T")
LoggerType = logging.Logger
LogLevel = Union[int, str]

# --- Globals ---
console = Console()

# === Logging Setup and Configuration ===

def setup_logging(initial_level: LogLevel = logging.INFO) -> None:
    """Configure root logger with file and console handlers."""
    root_logger = logging.getLogger()
    # Set root to DEBUG to capture everything, handlers control output level
    root_logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers if called multiple times
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s')

    # File Handler (writes to reddacted.log in current directory)
    try:
        file_handler = logging.FileHandler('reddacted.log', mode='a')
        file_handler.setLevel(initial_level) # Set initial level
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        # Fallback or notify if file logging fails
        sys.stderr.write(f"Error setting up file logger: {e}\n")

    # Console Handler (stderr)
    console_handler = logging.StreamHandler(sys.stderr)
    # Console handler is INFO unless global level is DEBUG
    console_level = logging.INFO if initial_level != logging.DEBUG else logging.DEBUG
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Set initial level for httpx (less noisy)
    logging.getLogger("httpx").setLevel(logging.WARNING if initial_level > logging.DEBUG else logging.DEBUG)


def set_global_logging_level(level: LogLevel) -> None:
    """Set the global logging level for root logger and handlers.

    Args:
        level: The logging level to set globally. Can be an integer level or string name.

    Note:
        This affects all existing loggers in the hierarchy.
        Some third-party loggers may be set to specific levels for noise reduction.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level) # Set root level first

    # Adjust handler levels
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.setLevel(level) # File handler matches global level
        elif isinstance(handler, logging.StreamHandler):
            # Set console handler level directly to the requested global level
            handler.setLevel(level)

    # Adjust specific noisy loggers
    httpx_level = logging.WARNING if level > logging.DEBUG else logging.DEBUG
    logging.getLogger("httpx").setLevel(httpx_level)


# === Logger Retrieval and Contextual Logging ===

def get_logger(name: str) -> LoggerType:
    """Get or create a logger with consistent formatting and contextual logging methods.

    Args:
        name: The name of the logger, typically __name__

    Returns:
        A Logger instance with additional contextual logging methods

    Example:
        >>> logger = get_logger(__name__) # Level is now controlled globally
        >>> logger.info_with_context("Starting process")
    """
    logger = logging.getLogger(name)
    # Level is inherited from root logger and its handlers

    # Check if methods already exist to avoid adding them multiple times
    if not hasattr(logger, "debug_with_context"):
        def make_log_method(log_level: int) -> Callable[[str, Optional[Callable]], None]:
            def log_method(msg: str, func: Optional[Callable] = None) -> None:
                log_with_context(logger, log_level, msg, func)

            return log_method

        # Add typed convenience methods
        setattr(logger, "debug_with_context", make_log_method(logging.DEBUG))
        setattr(logger, "info_with_context", make_log_method(logging.INFO))
        setattr(logger, "warning_with_context", make_log_method(logging.WARNING))
        setattr(logger, "error_with_context", make_log_method(logging.ERROR))
        setattr(logger, "critical_with_context", make_log_method(logging.CRITICAL))

    return logger


def get_log_context(func: Callable[..., Any], frame: Optional[Any] = None) -> str:
    """Get standardized logging context with file, function, and line number.

    Args:
        func: The function from which the log was called
        frame: Optional stack frame, will get caller's frame if None

    Returns:
        A formatted string with module, function and line information
    """
    if frame is None:
        frame = sys._getframe(2)  # Get caller's frame
    return f"[{func.__module__}:{func.__name__}:{frame.f_lineno}]"


def log_with_context(
    logger: LoggerType, level: int, msg: str, func: Optional[Callable[..., Any]] = None
) -> None:
    """Log a message with standardized context information.

    Args:
        logger: The logger instance to use
        level: The logging level for this message
        msg: The message to log
        func: Optional function to use for context, defaults to caller

    Note:
        This function automatically adds context information including:
        - Module name
        - Function name
        - Line number
    """
    frame = sys._getframe(1)
    if func is None:
        # Try to get caller's context if func is not provided
        try:
            frame = sys._getframe(2) # Go one level deeper to get the caller of log_with_context
            func_name = frame.f_code.co_name
            module_name = frame.f_globals.get("__name__", "unknown_module")
            context = f"[{module_name}:{func_name}:{frame.f_lineno}]"
        except (ValueError, AttributeError):
             context = "[unknown_context]" # Fallback if frame inspection fails
    else:
        context = get_log_context(func, frame)
    logger.log(level, f"{context} {msg}")


# === Exception Handling and Formatting ===

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
        # Prefer __cause__ for explicit chaining, fallback to __context__
        current = getattr(current, '__cause__', None) or getattr(current, '__context__', None)
        # Avoid infinite loops with self-referential contexts
        if current in exc_chain:
            break


    # Get traceback information
    tb = getattr(exc, "__traceback__", None) or sys.exc_info()[2]
    module_name = "unknown_module"
    func_name = "unknown_function"
    line_no = 0

    if tb:
        # Find the deepest relevant frame in the traceback
        relevant_tb = tb
        while relevant_tb.tb_next:
            relevant_tb = relevant_tb.tb_next

        try:
            frame = relevant_tb.tb_frame
            func_name = frame.f_code.co_name
            line_no = relevant_tb.tb_lineno
            module = inspect.getmodule(frame)
            module_name = module.__name__ if module else "unknown_module"
        except (AttributeError, ValueError):
            # Fallback if frame access fails
            pass # Keep defaults

    # Build the error message
    messages = []
    for i, e in enumerate(reversed(exc_chain)):
        try:
            error_type = e.__class__.__name__
            error_msg = str(e)
        except Exception:
            error_type = "UnknownError"
            error_msg = "Failed to format error message"

        if i == 0: # Original exception
            messages.append(f"[bold red]{error_type}[/]: {error_msg}")
        else: # Caused by / Context
            messages.append(f"[dim]Caused by: {error_type}: {error_msg}[/]")

    location = f"[dim]Location: {module_name}.{func_name}(), line {line_no}[/]"

    if include_trace and tb is not None:
        try:
            # Use rich Traceback for pretty printing
            rich_trace = Traceback.from_exception(
                exc_type=type(exc),
                exc_value=exc,
                traceback=tb,
                show_locals=False # Keep it concise by default
            )
            # Convert rich Traceback to string for return
            trace_str = "\n".join(str(line) for line in console.render_lines(rich_trace))
            return "\n".join(messages + [location, "", trace_str])
        except Exception as format_err:
            # Fallback if rich traceback formatting fails
            fallback_trace = "".join(traceback.format_exception(type(exc), exc, tb))
            return "\n".join(messages + [location, "", f"Failed to format traceback with Rich: {format_err}\n{fallback_trace}"])

    return "\n".join(messages + [location])


def handle_exception(exc: Exception, context: Optional[str] = None, debug: bool = False) -> None:
    """Logs and prints a formatted exception with optional context and debugging.

    Args:
        exc: The exception to handle
        context: Optional context about what was happening
        debug: Whether to include full traceback in logs and output
    """
    # Use get_logger internally to ensure we have a logger instance
    internal_logger = get_logger(__name__)

    # Log the full exception details for debugging purposes
    # exc_info=True automatically includes traceback if available
    log_context = f"Error in {context or 'unknown context'}"
    internal_logger.error(log_context, exc_info=exc if debug else False)

    # Format the error message for console output
    error_msg = format_exception(exc, include_trace=debug)
    if context:
        # Prepend the user-provided context to the formatted message
        error_msg = f"[yellow]Context:[/yellow] {context}\n{error_msg}"

    # Print the formatted error to the console using Rich Panel
    console.print(
        Panel(Text.from_markup(error_msg), title="[bold red]Error[/]", border_style="red", expand=False)
    )


# === Decorator ===

def with_logging(logger: LoggerType) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that adds logging context and exception handling to methods.

    Args:
        logger: The logger instance to use

    Returns:
        A decorator function that wraps the original function with logging

    Example:
        >>> logger = get_logger(__name__)
        >>> @with_logging(logger)
        >>> def process_data(data: dict) -> None:
        >>>     # Function implementation
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            frame = sys._getframe(1) # Get caller's frame (wrapper's caller)
            context_str = get_log_context(func, frame) # Use the original func for context

            try:
                logger.debug(f"{context_str} Starting {func.__name__}")
                result = func(*args, **kwargs)
                logger.debug(f"{context_str} Completed {func.__name__}")
                return result
            except Exception as e:
                error_msg_context = f"Exception in {func.__name__}"
                # Call the local handle_exception function directly
                # Determine debug flag based on logger's effective level
                is_debug = logger.getEffectiveLevel() <= logging.DEBUG
                handle_exception(e, error_msg_context, debug=is_debug)
                raise # Re-raise the exception after handling

        return wrapper

    return decorator