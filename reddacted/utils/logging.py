import logging
import sys
from functools import wraps
from typing import Callable, Any
from reddacted.utils.exceptions import handle_exception

def get_logger(name: str) -> logging.Logger:
    print('hi get logger call')
    """Get a logger with consistent formatting"""
    logger = logging.getLogger(name)
    
    # Add convenience methods for contextual logging
    def make_log_method(level: int):
        def log_method(msg: str, func: Callable = None):
            log_with_context(logger, level, msg, func)
        return log_method
    
    logger.debug_with_context = make_log_method(logging.DEBUG)
    logger.info_with_context = make_log_method(logging.INFO)
    logger.warning_with_context = make_log_method(logging.WARNING)
    logger.error_with_context = make_log_method(logging.ERROR)
    logger.critical_with_context = make_log_method(logging.CRITICAL)
    
    return logger

def get_log_context(func: Callable, frame=None) -> str:
    """Get standardized logging context with file, function, and line number"""
    if frame is None:
        frame = sys._getframe(2)  # Get caller's frame
    return f"[{func.__module__}:{func.__name__}:{frame.f_lineno}]"

def with_logging(logger: logging.Logger) -> Callable:
    """Decorator that adds logging context and exception handling to methods"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            frame = sys._getframe(1)  # Get caller's frame
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

def log_with_context(logger: logging.Logger, level: int, msg: str, func: Callable = None) -> None:
    """Log a message with standardized context information"""
    frame = sys._getframe(1)  # Get caller's frame
    if func is None:
        func_name = frame.f_code.co_name
        module_name = frame.f_globals['__name__']
        context = f"[{module_name}:{func_name}:{frame.f_lineno}]"
    else:
        context = get_log_context(func, frame)
    logger.log(level, f"{context} {msg}")
