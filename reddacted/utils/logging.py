import logging
import sys
from functools import wraps
from typing import Callable, Any, Optional, Union, Dict, TypeVar
from reddacted.utils.exceptions import handle_exception

T = TypeVar("T")
LoggerType = logging.Logger
LogLevel = Union[int, str]

def set_global_logging_level(level: LogLevel) -> None:
    """Set the global logging level for all loggers.

    Args:
        level: The logging level to set globally. Can be an integer level or string name.

    Note:
        This affects all existing loggers in the hierarchy.
        Some third-party loggers may be set to specific levels for noise reduction.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Set specific levels for noisy third-party loggers when not in debug mode
    if level > logging.DEBUG:
        logging.getLogger("httpx").setLevel(logging.WARNING)

    # Set level for all other loggers
    for logger_name in root.manager.loggerDict:
        if logger_name != "httpx":  # Skip httpx as we handled it above
            logging.getLogger(logger_name).setLevel(level)



def get_logger(name: str, level: LogLevel = logging.INFO) -> LoggerType:
    """Get or create a logger with consistent formatting and contextual logging methods.

    Args:
        name: The name of the logger, typically __name__
        level: The logging level to set, defaults to INFO

    Returns:
        A Logger instance with additional contextual logging methods

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info_with_context("Starting process")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

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
