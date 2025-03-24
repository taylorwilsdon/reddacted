"""
Reddit CLI for PII Detection and Sentiment Analysis

This module provides a command-line interface for analyzing Reddit content,
detecting PII, and managing comments using both local and OpenAI LLMs.
"""

import sys
import os
import getpass
import logging
import argparse
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable

import requests
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from reddacted.utils.logging import get_logger, with_logging, set_global_logging_level
from reddacted.utils.exceptions import handle_exception
from reddacted.sentiment import Sentiment
from reddacted.api.reddit import Reddit

# Initialize logging with consistent format
set_global_logging_level(logging.INFO)
logger = get_logger(__name__)
console = Console(highlight=True)

# Environment variables required for Reddit API authentication
REDDIT_AUTH_VARS = [
    "REDDIT_USERNAME",
    "REDDIT_PASSWORD",
    "REDDIT_CLIENT_ID",
    "REDDIT_CLIENT_SECRET",
]

@dataclass
class Command:
    """Base class for all CLI commands"""
    name: str
    description: str
    handler: Callable
    arguments: List[Dict[str, Any]] = field(default_factory=list)

    def setup_parser(self, parser: argparse.ArgumentParser) -> None:
        """Configure command-specific arguments"""
        for arg in self.arguments:
            parser.add_argument(*arg["args"], **arg["kwargs"])

@dataclass
class CommandGroup:
    """Group of related commands"""
    name: str
    commands: List[Command]

class ModifyCommand(Command):
    """Base class for comment modification commands"""
    def __init__(self, name: str, description: str, handler: Callable):
        arguments = [
            {
                "args": ["comment_ids"],
                "kwargs": {
                    "help": "Comma-separated list of comment IDs to process (e.g., abc123,def456)",
                }
            },
            {
                "args": ["--batch-size"],
                "kwargs": {
                    "type": int,
                    "default": 10,
                    "help": "Number of comments to process in each API request (default: 10)",
                }
            }
        ]
        super().__init__(name=name, description=description, handler=handler, arguments=arguments)

    @with_logging(logger)
    def process_comments(self, args: argparse.Namespace, action: str) -> Dict[str, Any]:
        """Process comments with the specified action and progress tracking"""
        api = Reddit()
        comment_ids = [id.strip() for id in args.comment_ids.split(",")]
        total_comments = len(comment_ids)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"[cyan]{action.title()} processing...", total=total_comments)

            if action == "delete":
                result = api.delete_comments(comment_ids, batch_size=args.batch_size)
            elif action == "update":
                result = api.update_comments(comment_ids, batch_size=args.batch_size)
            else:
                raise ValueError(f"Invalid action: {action}")

            progress.update(task, completed=result["success"])
            return result

    def format_results(self, results: Dict[str, Any], action: str) -> str:
        """Format operation results for display"""
        details = []
        details.append(f"[cyan]Processed:[/] {results['processed']}")
        details.append(f"[green]Successful:[/] {results['success']}")
        details.append(f"[red]Failed:[/] {results['failures']}\n")

        if results.get("successful_ids"):
            details.append(f"[green]Successfully {action.title()}d Comments:[/]")
            for comment_id in results["successful_ids"]:
                details.append(f"  • [dim]t1_{comment_id}[/]")

        if results.get("failed_ids"):
            details.append(f"\n[red]Failed to {action.title()} Comments:[/]")
            for comment_id in results["failed_ids"]:
                details.append(f"  • [dim]t1_{comment_id}[/]")

        return "\n".join(details)

class AnalyzeCommand(Command):
    """Base class for Reddit analysis commands"""
    def __init__(self, name: str, description: str, handler: Callable, extra_args: Optional[List[Dict[str, Any]]] = None):
        arguments = [
            {
                "args": ["--output-file"],
                "kwargs": {
                    "dest": "output_file",
                    "help": "Outputs a file with information on each sentence of the post, as well as the final score.",
                }
            },
            {
                "args": ["--enable-auth"],
                "kwargs": {
                    "dest": "enable_auth",
                    "action": "store_true",
                    "help": "Enable reddit api authentication using environment variables",
                }
            },
            {
                "args": ["--disable-pii"],
                "kwargs": {
                    "dest": "disable_pii",
                    "action": "store_true",
                    "help": "Disable PII detection in the analysis",
                }
            },
            {
                "args": ["--openai-key"],
                "kwargs": {
                    "dest": "openai_key",
                    "help": "OpenAI API key for LLM-based analysis",
                }
            },
            {
                "args": ["--local-llm"],
                "kwargs": {
                    "dest": "local_llm",
                    "help": "URL for local LLM endpoint (OpenAI compatible)",
                }
            },
            {
                "args": ["--openai-base"],
                "kwargs": {
                    "dest": "openai_base",
                    "help": "Optional OpenAI API base URL",
                }
            },
            {
                "args": ["--model"],
                "kwargs": {
                    "dest": "model",
                    "help": "OpenAI or local LLM model to use",
                }
            },
            {
                "args": ["--pii-only"],
                "kwargs": {
                    "dest": "pii_only",
                    "action": "store_true",
                    "help": "Only show comments that contain PII (0 < score < 1.0)",
                }
            },
            {
                "args": ["--limit"],
                "kwargs": {
                    "dest": "limit",
                    "type": int,
                    "default": 100,
                    "help": "Maximum number of comments to analyze (default: 100, use 0 for unlimited)",
                }
            },
            {
                "args": ["--sort"],
                "kwargs": {
                    "dest": "sort",
                    "choices": ["hot", "new", "controversial", "top"],
                    "default": "new",
                    "help": "Sort method for comments (default: new)",
                }
            },
            {
                "args": ["--time"],
                "kwargs": {
                    "dest": "time",
                    "choices": ["all", "day", "hour", "month", "week", "year"],
                    "default": "all",
                    "help": "Time filter for comments (default: all)",
                }
            },
            {
                "args": ["--text-match"],
                "kwargs": {
                    "dest": "text_match",
                    "help": "Search for comments containing specific text (requires authentication)",
                }
            },
            {
                "args": ["--skip-text"],
                "kwargs": {
                    "dest": "skip_text",
                    "help": "Skip comments containing this text pattern",
                }
            },
        ]
        if extra_args:
            arguments.extend(extra_args)
        
        super().__init__(name=name, description=description, handler=handler, arguments=arguments)

    def check_auth_env_vars(self) -> bool:
        """Check if all required Reddit API environment variables are set"""
        return all(os.getenv(var) for var in REDDIT_AUTH_VARS)

class CLI:
    """Main CLI application class"""
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="""
            Reddit LLM PII & Sentiment Analysis Tool

            Commands:
            listing     Analyze a Reddit post and its comments
            user        Analyze a user's comment history
            delete      Delete comments by ID
            update      Replace comment content with r/reddacted

            Authentication:
            Set these environment variables for Reddit API access:
                REDDIT_USERNAME
                REDDIT_PASSWORD
                REDDIT_CLIENT_ID
                REDDIT_CLIENT_SECRET

            LLM Configuration:
            --openai-key     OpenAI API key
            --local-llm      Local LLM endpoint URL
            --openai-base    Custom OpenAI API base URL
            --model          Model name to use (default: gpt-4)

            Common Options:
            --output-file    Save detailed analysis to file
            --enable-auth    Use Reddit API authentication
            --disable-pii    Skip PII detection
            --pii-only       Show only comments with PII
            --limit          Max comments to analyze (0=unlimited)
            --batch-size     Comments per batch for delete/update
            --text-match     Search for comments containing specific text
            --skip-text      Skip comments containing this text pattern
            """,
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        self.parser.add_argument('--version', action='version', version='%(prog)s 1.0')
        self.parser.add_argument('--debug', action='store_true', help='Enable debug logging')

        # Set up subcommands
        self.subparsers = self.parser.add_subparsers(dest='command', help='Commands')
        self.setup_commands()

    def setup_commands(self) -> None:
        """Register all available commands"""
        # Analysis commands
        listing_cmd = AnalyzeCommand(
            "listing",
            "Analyze a Reddit post and its comments",
            self.handle_listing,
            extra_args=[
                {"args": ["subreddit"], "kwargs": {"help": "The subreddit"}},
                {"args": ["article"], "kwargs": {"help": "The id of the article"}},
            ]
        )
        user_cmd = AnalyzeCommand(
            "user",
            "Analyze a user's comment history",
            self.handle_user,
            extra_args=[
                {"args": ["username"], "kwargs": {"nargs": "?", "help": "The name of the user"}},
            ]
        )

        # Modification commands
        delete_cmd = ModifyCommand(
            "delete",
            "Delete specified Reddit comments permanently",
            self.handle_delete
        )
        update_cmd = ModifyCommand(
            "update",
            'Replace comment content with "r/reddacted"',
            self.handle_update
        )

        # Register commands
        for cmd in [listing_cmd, user_cmd, delete_cmd, update_cmd]:
            parser = self.subparsers.add_parser(cmd.name, help=cmd.description)
            cmd.setup_parser(parser)
            parser.set_defaults(func=cmd.handler)

    @staticmethod
    def _configure_llm(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
        """Centralized LLM configuration handler"""
        logger.debug("Configuring LLM settings")
        if args.disable_pii:
            return None

        if args.local_llm:
            base_url = args.local_llm.rstrip("/v1")
            console.print(f"[blue]Using local LLM endpoint: {base_url}[/]")

            try:
                # Get available models
                models_url = f"{base_url}/v1/models"
                response = requests.get(models_url)
                if response.status_code != 200:
                    console.print(f"[red]Error: Could not fetch models: {response.status_code}[/]")
                    return None
                models_data = response.json()
                available_models = [m["id"] for m in models_data.get("data", [])]
                args.model = args.model or available_models[0]

                if args.model not in available_models:
                    console.print(f"[red]Error: Model '{args.model}' not available[/]")
                    return None

                return {
                    "api_key": "sk-not-needed",
                    "api_base": f"{base_url}/v1",
                    "model": args.model,
                    "default_headers": {"User-Agent": "Reddit-Sentiment-Analyzer"},
                }

            except Exception as e:
                console.print(f"[red]Connection error: {str(e)}[/]")
                return None

        elif args.openai_key:
            return {
                "api_key": args.openai_key,
                "api_base": args.openai_base or "https://api.openai.com/v1",
                "model": args.model or "gpt-4",
            }

        # Prompt for configuration if none provided
        console.print("[yellow]LLM required for PII detection[/]")
        llm_choice = Prompt.ask("Choose LLM provider", choices=["local", "openai"], default="local")

        if llm_choice == "openai":
            args.openai_key = getpass.getpass("Enter your OpenAI API key: ")
            return {
                "api_key": args.openai_key,
                "api_base": args.openai_base or "https://api.openai.com/v1",
                "model": args.model or "gpt-4",
            }
        else:
            args.local_llm = Prompt.ask(
                "Enter local LLM endpoint URL",
                default="http://localhost:11434"
            )
            return CLI._configure_llm(args)

    @with_logging(logger)
    def handle_listing(self, args: argparse.Namespace) -> None:
        """Handle the listing command"""
        llm_config = self._configure_llm(args)
        limit = None if args.limit == 0 else args.limit

        # Enable auth if flag is set or all env vars are present
        auth_enabled = args.enable_auth or all(os.getenv(var) for var in REDDIT_AUTH_VARS)

        sent = Sentiment(
            auth_enabled=auth_enabled,
            pii_enabled=not args.disable_pii,
            pii_only=args.pii_only,
            llm_config=llm_config,
            sort=args.sort,
            limit=limit,
            skip_text=args.skip_text,
        )
        # Strip 'r/' prefix if present when using API
        subreddit = args.subreddit.replace("r/", "") if auth_enabled else args.subreddit
        sent.get_sentiment("listing", f"{subreddit}/{args.article}", output_file=args.output_file)

    @with_logging(logger)
    def handle_user(self, args: argparse.Namespace) -> None:
        """Handle the user command"""
        try:
            # Prompt for username if not provided
            if not args.username:
                args.username = Prompt.ask("Enter Reddit username to analyze", default="spez")
                console.print(f"[blue]Analyzing user: u/{args.username}[/]")

            llm_config = self._configure_llm(args)
            limit = None if args.limit == 0 else args.limit

            # Enable auth if flag is set or all env vars are present
            auth_enabled = args.enable_auth or all(os.getenv(var) for var in REDDIT_AUTH_VARS)

            sent = Sentiment(
                auth_enabled=auth_enabled,
                pii_enabled=not args.disable_pii,
                llm_config=llm_config,
                pii_only=args.pii_only,
                sort=args.sort,
                limit=limit,
                skip_text=args.skip_text,
            )
            sent.get_sentiment(
                "user",
                args.username,
                output_file=args.output_file,
                sort=args.sort,
                time_filter=args.time,
                text_match=args.text_match,
            )
        except Exception as e:
            handle_exception(
                e,
                f"Failed to analyze user '{args.username}'\n"
                + "Check if the user exists and is not banned/private",
                debug=args.debug
            )
            raise

    @with_logging(logger)
    def handle_delete(self, args: argparse.Namespace) -> None:
        """Handle the delete command"""
        cmd = ModifyCommand("delete", "Delete comments", lambda x: None)
        results = cmd.process_comments(args, "delete")
        formatted_results = cmd.format_results(results, "delete")
        console.print(
            "\n",
            Panel(formatted_results, title="[bold red]Delete Results[/]", expand=False)
        )

    @with_logging(logger)
    def handle_update(self, args: argparse.Namespace) -> None:
        """Handle the update command"""
        cmd = ModifyCommand("update", "Update comments", lambda x: None)
        results = cmd.process_comments(args, "update")
        formatted_results = cmd.format_results(results, "update")
        console.print(
            "\n",
            Panel(formatted_results, title="[bold blue]Update Results[/]", expand=False)
        )

    def run(self, argv: List[str]) -> int:
        """Run the CLI application"""
        try:
            if "--debug" in argv:
                set_global_logging_level(logging.DEBUG)
            else:
                set_global_logging_level(logging.INFO)

            args = self.parser.parse_args(argv)
            if not args.command:
                self.parser.print_help()
                return 0

            args.func(args)
            return 0
        except Exception as e:
            command = argv[0] if argv else "unknown"
            handle_exception(e, f"Failed to execute command '{command}'", debug="--debug" in argv)
            return 1

def main(argv: List[str] = sys.argv[1:]) -> int:
    """Main entry point"""
    cli = CLI()
    return cli.run(argv)

if __name__ == "__main__":
    sys.exit(main())
