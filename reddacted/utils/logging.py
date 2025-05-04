import logging
import sys
from functools import wraps
from typing import Callable, Any, Optional, Union, Dict, TypeVar
from reddacted.utils.exceptions import handle_exception

T = TypeVar("T")
LoggerType = logging.Logger
LogLevel = Union[int, str]

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
    console_handler.setLevel(logging.INFO) # Console always shows INFO+
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
            # Console handler is INFO unless global level is DEBUG
            handler.setLevel(max(level, logging.INFO))

    # Adjust specific noisy loggers
    httpx_level = logging.WARNING if level > logging.DEBUG else logging.DEBUG
    logging.getLogger("httpx").setLevel(httpx_level)


def get_logger(name: str) -> LoggerType:
    """Get or create a logger with consistent formatting and contextual logging methods.

    Args:
        name: The name of the logger, typically __name__
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


def with_logging(logger: LoggerType) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that adds logging context and exception handling to methods.

    Args:
        logger: The logger instance to use

    Returns:
        A decorator function that wraps the original function with logging

    Example:
        >>> @with_logging(logger)
        >>> def process_data(data: dict) -> None:
        >>>     # Function implementation
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            frame = sys._getframe(1)
            context = get_log_context(func, frame)

            try:
                logger.debug(f"{context} Starting {func.__name__}")
                result = func(*args, **kwargs)
                logger.debug(f"{context} Completed {func.__name__}")
                return result
            except Exception as e:
                error_msg = f"Error in {func.__name__}"
                handle_exception(e, error_msg, logger.getEffectiveLevel() <= logging.DEBUG)
                raise

        return wrapper

    return decorator


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
        func_name = frame.f_code.co_name
        module_name = frame.f_globals["__name__"]
        context = f"[{module_name}:{func_name}:{frame.f_lineno}]"
    else:
        context = get_log_context(func, frame)
    logger.log(level, f"{context} {msg}")
