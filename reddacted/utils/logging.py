import logging
import sys
from functools import wraps
from typing import Callable, Any
from reddacted.utils.exceptions import handle_exception

def get_logger(name: str) -> logging.Logger:
    """Get a logger with consistent formatting"""
    logger = logging.getLogger(name)
    return logger

def with_logging(logger: logging.Logger) -> Callable:
    """Decorator that adds logging context and exception handling to methods"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Get context information
            context = f"[{func.__module__}:{func.__name__}:{sys._getframe().f_lineno}]"
            
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
