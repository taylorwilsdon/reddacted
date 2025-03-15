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

DetailsScreen {
    background: #0e333d;
}

.details-container {
    padding: 1 2;
    width: 100%;
}

.details-title {
    text-align: center;
    background: #184956;
    color: #cad8d9;
    padding: 1;
    text-style: bold;
    border-bottom: solid #58a3ff;
    margin-bottom: 1;
}

#details-header-grid {
    grid-size: 2 2;
    grid-columns: 1fr 1fr;
    grid-rows: auto auto;
    align: center middle;
    width: 100%;
    margin-bottom: 1;
    padding: 0 1;
    background: #103c48;
    border: solid #4695f7;
}

.details-id {
    color: #cad8d9;
}

.details-sentiment {
    color: #cad8d9;
}

.details-votes {
    color: #cad8d9;
    text-align: right;
}

.details-actions {
    height: auto;
    width: 100%;
    margin-bottom: 1;
    padding: 1;
    background: #184956;
    border: solid #58a3ff;
    align: center middle;
}

.details-actions Button {
    margin: 0 1;
    min-width: 16;
}

#back-btn {
    background: #184956;
    color: #cad8d9;
    border: solid #58a3ff;
}

.details-text {
    padding: 1;
    background: #103c48;
    border: solid #4695f7;
    margin-bottom: 1;
    min-height: 3;
    max-height: 15;
}

.section-header {
    background: #184956;
    color: #cad8d9;
    padding: 0 1;
    margin-top: 1;
    margin-bottom: 1;
    text-style: bold;
}

.subsection-header {
    color: #cad8d9;
    text-style: italic;
    margin-top: 1;
}

.details-pii-item, .details-llm-item {
    padding-left: 2;
    color: #cad8d9;
}

.details-reasoning {
    padding: 1;
    background: #103c48;
    border: solid #4695f7;
    margin-top: 1;
    margin-bottom: 1;
}

.details-risk-high {
    color: #ef4444;
    text-style: bold;
}

.details-risk-medium {
    color: #f59e0b;
    text-style: bold;
}

.details-risk-low {
    color: #22c55e;
    text-style: bold;
}

.details-llm-risk {
    color: #cad8d9;
}

.details-has-pii-yes {
    color: #ef4444;
    text-style: bold;
}

.details-has-pii-no {
    color: #22c55e;
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