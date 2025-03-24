"""
Report Generator Module

This module provides functions to generate analysis reports.
It extracts the report header and output file generation logic from ResultsScreen,
allowing for a single-call report generation.
"""

from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from dataclasses import dataclass

    @dataclass
    class AnalysisResult:
        """Type hint for analysis result structure"""

        comment_id: str
        sentiment_score: float
        sentiment_emoji: str
        pii_risk_score: float
        pii_matches: List[Any]
        text: str
        upvotes: int
        downvotes: int
        llm_risk_score: float
        llm_findings: Optional[Dict[str, Any]]


def write_report_header(
    target, url: str, overall_score: float, overall_sentiment: str, num_comments: int
) -> None:
    """
    Writes the header section of the analysis report.
    """
    target.write(f"# Analysis Report for '{url}'\n\n")
    target.write(f"- **Overall Sentiment Score**: {overall_score:.2f}\n")
    target.write(f"- **Overall Sentiment**: {overall_sentiment}\n")
    target.write(f"- **Comments Analyzed**: {num_comments}\n\n")
    target.write("---\n\n")


__all__ = [
    "generate_analysis_report",
    "should_show_result",
    "format_llm_detail",
]


def should_show_result(result: "AnalysisResult", pii_only: bool = False) -> bool:
    """
    Determines if a result should be shown based on PII detection settings.
    """
    if not pii_only:
        return True
    has_pattern_pii = result.pii_risk_score > 0.0
    has_llm_pii = (
        result.llm_findings is not None
        and isinstance(result.llm_findings, dict)
        and result.llm_findings.get("has_pii", False)
        and result.llm_findings.get("confidence", 0.0) > 0.0
    )
    return has_pattern_pii or has_llm_pii


def format_llm_detail(detail: Any, app=None) -> str:
    """Formats LLM detail information."""
    if isinstance(detail, dict):
        formatted = (
            f"{detail.get('type', 'Finding')}: {detail.get('example', 'N/A')}"
            or f"{detail.get('finding', 'N/A')}: {detail.get('reasoning', '')}"
        )
        return formatted.replace('\n', ' ')  # Replace newlines with spaces
    return str(detail)


def write_comment_details(target, result: "AnalysisResult", index: int) -> None:
    """
    Writes detailed analysis for a single comment.
    """
    target.write(f"## Comment {index}\n\n")
    target.write(f"**Text**: {result.text}\n\n")
    target.write(f"- Sentiment Score: `{result.sentiment_score:.2f}` {result.sentiment_emoji}\n")
    target.write(f"- PII Risk Score: `{result.pii_risk_score:.2f}`\n")
    target.write(f"- Votes: ⬆️ `{result.upvotes}` ⬇️ `{result.downvotes}`\n")
    target.write(f"- Comment ID: `{result.comment_id}`\n\n")
    if result.pii_matches:
        target.write("### Pattern-based PII Detected\n")
        for pii in result.pii_matches:
            target.write(f"- **{pii.type}** (confidence: {pii.confidence:.2f})\n")
        target.write("\n")
    if result.llm_findings:
        target.write("### LLM Privacy Analysis\n")
        target.write(f"- **Risk Score**: `{result.llm_risk_score:.2f}`\n")
        if isinstance(result.llm_findings, dict):
            target.write(
                f"- **PII Detected**: {'Yes' if result.llm_findings.get('has_pii') else 'No'}\n"
            )
            if details := result.llm_findings.get("details"):
                target.write("\n#### Findings\n")
                for detail in details:
                    target.write(f"- {format_llm_detail(detail)}\n")
            if reasoning := result.llm_findings.get("reasoning"):
                target.write(f"\n#### Reasoning\n{reasoning}\n")
        target.write("\n")
    target.write("---\n\n")


def write_summary_section(
    target,
    total_comments: int,
    sentiment_scores: List[float],
    max_risk_score: float,
    riskiest_comment: str,
    total_pii_comments: int = 0,
    total_llm_pii_comments: int = 0,
) -> None:
    """
    Writes the summary section of the analysis report.
    """
    average_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
    target.write("\n# Summary\n\n")
    target.write(f"- Total Comments Analyzed: {total_comments}\n")
    target.write(
        f"- Comments with PII Detected: {total_pii_comments} ({total_pii_comments/total_comments:.1%})\n"
    )
    target.write(
        f"- Comments with LLM Privacy Risks: {total_llm_pii_comments} ({total_llm_pii_comments/total_comments:.1%})\n"
    )
    target.write(f"- Average Sentiment Score: {average_sentiment:.2f}\n")
    target.write(f"- Highest PII Risk Score: {max_risk_score:.2f}\n")
    if riskiest_comment:
        target.write(f"- Riskiest Comment Preview: '{riskiest_comment}'\n")
    target.write("✅ Analysis complete\n")


def generate_analysis_report(
    filename: str,
    comments: List[Dict[str, Any]],
    url: str,
    results: List["AnalysisResult"],
    overall_score: float,
    overall_sentiment: str,
    pii_only: bool = False,
) -> Dict[str, int]:
    """
    Generates an analysis report by writing the header, comment details, and summary.

    This function encapsulates the report generation logic previously embedded
    in the ResultsScreen, thereby reducing file bloat.

    Returns:
        Dict containing statistics about the analysis (total_pii_comments, total_llm_pii_comments)
    """
    try:
        sentiment_scores: List[float] = []
        max_risk_score = 0.0
        riskiest_comment = ""
        total_pii_comments = 0
        total_llm_pii_comments = 0

        with open(filename, "w") as target:
            write_report_header(target, url, overall_score, overall_sentiment, len(comments))

            for idx, result in enumerate(results, 1):
                if not should_show_result(result, pii_only):
                    continue
                write_comment_details(target, result, idx)

                # Update statistics
                sentiment_scores.append(result.sentiment_score)
                if result.pii_risk_score > 0:
                    total_pii_comments += 1
                if result.llm_risk_score > 0 or (
                    result.llm_findings and result.llm_findings.get("has_pii", False)
                ):
                    total_llm_pii_comments += 1

                if result.pii_risk_score > max_risk_score:
                    max_risk_score = result.pii_risk_score
                    riskiest_comment = (
                        (result.text[:100] + "...") if len(result.text) > 100 else result.text
                    )

            write_summary_section(
                target,
                len(comments),
                sentiment_scores,
                max_risk_score,
                riskiest_comment,
                total_pii_comments,
                total_llm_pii_comments,
            )

        print(f"Report generated successfully at {filename}")
        return {
            "total_pii_comments": total_pii_comments,
            "total_llm_pii_comments": total_llm_pii_comments,
        }
    except Exception as e:
        print(f"Error generating report: {e}")
        raise
