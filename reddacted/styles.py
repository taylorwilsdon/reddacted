"""Centralized styling configuration for the application."""

from textual.color import Color

# Textual CSS 
TEXTUAL_CSS = """
Screen {
    background: #0e333d;  /* A custom dark blend derived from bg_0 (#103c48) */
}

Header {
    dock: top;
    background: #184956;
    color: #cad8d9;
    height: 3;
    content-align: center middle;
    border-bottom: heavy #58a3ff;
}

Footer {
    dock: bottom;
    background: #184956;
    color: #cad8d9;
    height: 1;
}

StatsDisplay {
    height: auto;
    padding: 1;
    background: #103c48;
    border: heavy #4695f7;
    margin: 1;
}

DataTable {
    height: auto;
    margin: 1;
    border: heavy #4695f7;
}

.stats-text {
    text-align: center;
}

CommentActionScreen {
    align: center middle;
}
"""

# Color System
COLORS = {
    "primary": Color.parse("#6366f1"),      # Indigo
    "secondary": Color.parse("#a855f7"),    # Purple
    "success": Color.parse("#22c55e"),      # Green
    "warning": Color.parse("#f59e0b"),      # Amber
    "error": Color.parse("#ef4444"),        # Red
    "surface": Color.parse("#1e293b"),      # Slate
    "background": Color.parse("#0f172a"),   # Dark slate
    "text": Color.parse("#f8fafc"),         # Light slate
    "muted": Color.parse("#64748b")         # Medium slate
}

# Typography
TYPOGRAPHY = {
    "h1": ("24px", "bold"),
    "h2": ("20px", "bold"), 
    "h3": ("16px", "bold"),
    "body": ("14px", "normal"),
    "small": ("12px", "normal")
}

# Spacing Scale
SPACING = {
    "xs": 4,
    "sm": 8,
    "md": 16,
    "lg": 24,
    "xl": 32
}

# Component-specific styles
TABLE_STYLES = {
    "header_style": "bold magenta",
    "border": "rounded",
    "padding": (0, 1),
    "collapse_padding": True
}

PANEL_STYLES = {
    "border_style": "blue",
    "padding": (1, 1)
}

# Risk level styles
def get_risk_style(score: float) -> str:
    """Get appropriate color style based on risk score."""
    if score > 0.5:
        return "error"
    elif score > 0.2:
        return "warning" 
    return "success"

# Status styles
def get_status_style(enabled: bool) -> str:
    """Get appropriate color style based on status."""
    return "success" if enabled else "error"