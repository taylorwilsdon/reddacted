import sys
import inspect
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

def format_exception(exc: Exception) -> str:
    """Format exception with source location and clean message"""
    # Get the frame where the exception occurred
    tb = sys.exc_info()[2]
    while tb.tb_next:
        tb = tb.tb_next
    frame = tb.tb_frame
    
    # Get function name and line number
    func_name = frame.f_code.co_name
    line_no = tb.tb_lineno
    
    # Get the module name
    module = inspect.getmodule(frame)
    module_name = module.__name__ if module else "unknown"
    
    # Format the error message
    error_type = exc.__class__.__name__
    error_msg = str(exc)
    
    return f"[bold red]{error_type}[/]: {error_msg}\n" + \
           f"[dim]Location: {module_name}.{func_name}(), line {line_no}[/]"

def handle_exception(exc: Exception, context: str = None) -> None:
    """Print a formatted exception with optional context"""
    error_msg = format_exception(exc)
    if context:
        error_msg = f"{context}\n{error_msg}"
    
    console.print(Panel(
        Text.from_markup(error_msg),
        title="[bold red]Error[/]",
        border_style="red"
    ))
