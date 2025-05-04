import sys
import os
import logging
import argparse
from typing import Dict, Any, List, Optional, Callable


from rich.console import Console
from reddacted.utils.log_handler import handle_exception
from reddacted.sentiment import Sentiment
from reddacted.api.reddit import Reddit
from .textual_cli import ConfigApp
from .cli_config import ENV_VARS_MAP
from reddacted.utils.log_handler import setup_logging, set_global_logging_level, get_logger, with_logging

# Logging is now set up in CLI.run()
logger = get_logger(__name__) # Logger instance can be created before setup
console = Console(highlight=True)

REDDIT_AUTH_VARS = [
    "REDDIT_USERNAME",
    "REDDIT_PASSWORD",
    "REDDIT_CLIENT_ID",
    "REDDIT_CLIENT_SECRET",
]

@with_logging(logger)
def handle_listing(config: Dict[str, Any]) -> None:
    """Handle the listing command using unified config."""
    logger.debug_with_context(f"Handling listing command with config: {config}")
    # Default to 20 if limit is missing, map 0 to None (unlimited)
    limit_val = config.get("limit", 20) # Default to 20 if key is missing
    limit = None if limit_val == 0 else limit_val

    auth_enabled = config.get("enable_auth", False)

    llm_config = None
    if config.get("openai_key") or config.get("local_llm"):
         llm_config = {
             "api_key": config.get("openai_key") if config.get("use_openai_api") else "sk-not-needed",
             "api_base": config.get("local_llm") if not config.get("use_openai_api") else "https://api.openai.com/v1",
             "model": config.get("model"),
         }
         # Adjust api_base for local LLM if needed
         if not config.get("use_openai_api") and llm_config["api_base"]:
             base_url = llm_config["api_base"].rstrip('/')
             if not base_url.endswith('/v1'):
                 llm_config["api_base"] = f"{base_url}/v1"


    # Pass the entire config dictionary to Sentiment constructor
    sent = Sentiment(config=config)
    subreddit = config["subreddit"].replace("r/", "") if auth_enabled else config["subreddit"]
    target = f"{subreddit}/{config['article']}"
    console.print(f"[cyan]Analyzing listing:[/cyan] {target}")
    sent.get_sentiment("listing", target, output_file=config.get("output_file"))
    console.print("[green]Listing analysis complete.[/green]")


@with_logging(logger)
def handle_user(config: Dict[str, Any]) -> None:
    """Handle the user command using unified config."""
    username = config["username"]
    if not username:
        console.print("[red]Error: Username is required for the user command.[/red]")
        return

    console.print(f"[cyan]Analyzing user:[/cyan] u/{username}")

    # Default to 20 if limit is missing, map 0 to None (unlimited)
    limit_val = config.get("limit", 20) # Default to 20 if key is missing
    limit = None if limit_val == 0 else limit_val
    auth_enabled = config.get("enable_auth", False)

    llm_config = None
    # Set up LLM configuration
    llm_config = None
    if config.get("model"):
        # If model is specified but no LLM URL, default to local
        if not config.get("local_llm") and not config.get("openai_key"):
            config["local_llm"] = "http://localhost:11434"
            console.print("[yellow]Warning:[/yellow] No LLM URL specified, defaulting to local")

        # Construct LLM config with model parameter
        llm_config = {
            "api_key": config.get("openai_key") if config.get("use_openai_api") else "sk-not-needed",
            "api_base": config.get("local_llm") if not config.get("use_openai_api") else "https://api.openai.com/v1",
            "model": config.get("model"),
        }
        # Adjust api_base for local LLM if needed
        if not config.get("use_openai_api") and llm_config["api_base"]:
            base_url = llm_config["api_base"].rstrip('/')
            if not base_url.endswith('/v1'):
                llm_config["api_base"] = f"{base_url}/v1"

    elif config.get("openai_key") or config.get("local_llm"):
        llm_config = {
            "api_key": config.get("openai_key") if config.get("use_openai_api") else "sk-not-needed",
            "api_base": config.get("local_llm") if not config.get("use_openai_api") else "https://api.openai.com/v1",
            "model": config.get("model"),
        }
        if not config.get("use_openai_api") and llm_config["api_base"]:
            base_url = llm_config["api_base"].rstrip('/')
            if not base_url.endswith('/v1'):
                llm_config["api_base"] = f"{base_url}/v1"

    try:
        # Pass the entire config dictionary to Sentiment constructor
        sent = Sentiment(config=config)
        sent.get_sentiment(
            "user",
            username,
            output_file=config.get("output_file"),
            sort=config.get("sort", "new"),
            time_filter=config.get("time", "all"),
            text_match=config.get("text_match"),
        )
        console.print(f"[green]User '{username}' analysis complete.[/green]")
    except Exception as e:
        handle_exception(
            e,
            f"Failed to analyze user '{username}'\n"
            + "Check if the user exists and is not banned/private.",
            debug=config.get("debug", False)
        )
        raise


@with_logging(logger)
def handle_delete(config: Dict[str, Any]) -> None:
    """Handle the delete command using unified config."""
    logger.debug_with_context(f"Handling delete command with config: {config}")
    if not config.get("enable_auth"):
        console.print("[red]Error: Reddit API authentication must be enabled to delete comments.[/red]")
        return

    comment_ids_str = config.get("comment_ids")
    if not comment_ids_str:
         console.print("[red]Error: Comment IDs are required for the delete command.[/red]")
         return

    comment_ids = [id.strip() for id in comment_ids_str.split(",")]
    batch_size = config.get("batch_size", 10)

    console.print(f"[cyan]Attempting to delete {len(comment_ids)} comments...[/cyan]")

    # Pass the entire config dictionary to the Reddit constructor
    reddit_api = Reddit(config=config)

    try:
        result = reddit_api.delete_comments(comment_ids, batch_size=batch_size)

        console.print(f"[green]Delete operation finished.[/green]")
        console.print(f"  Processed: {result.get('processed', 0)}")
        console.print(f"  Successful: {result.get('success', 0)}")
        console.print(f"  Failed: {result.get('failures', 0)}") # Added missing line back
        if result.get("successful_ids"):
            console.print("[bold green]Successfully Deleted:[/bold green]")
            for cid in result["successful_ids"]: console.print(f"  - t1_{cid}")
        if result.get("failed_ids"):
            console.print("[bold red]Failed to Delete:[/bold red]")
            for cid in result["failed_ids"]: console.print(f"  - t1_{cid}")

    except Exception as e:
        handle_exception(e, "Failed during delete operation.", debug=config.get("debug", False))


@with_logging(logger)
def handle_update(config: Dict[str, Any]) -> None:
    """Handle the update command using unified config."""
    logger.debug_with_context(f"Handling update command with config: {config}")
    if not config.get("enable_auth"):
        console.print("[red]Error: Reddit API authentication must be enabled to update comments.[/red]")
        return

    comment_ids_str = config.get("comment_ids")
    if not comment_ids_str:
         console.print("[red]Error: Comment IDs are required for the update command.[/red]")
         return

    comment_ids = [id.strip() for id in comment_ids_str.split(",")]
    batch_size = config.get("batch_size", 10)

    console.print(f"[cyan]Attempting to update {len(comment_ids)} comments...[/cyan]")

    # Get the use_random_string preference from config
    use_random_string = config.get("use_random_string", False)
    
    # Pass the entire config dictionary and use_random_string preference
    reddit_api = Reddit(config=config, use_random_string=use_random_string)

    try:
        # Pass the use_random_string parameter from config
        use_random_string = config.get("use_random_string", False)
        result = reddit_api.update_comments(
            comment_ids,
            batch_size=batch_size,
            use_random_string=use_random_string
        )
        
        # Log which mode was used
        if use_random_string:
            console.print("[cyan]Using random UUID for comment content.[/cyan]")
        else:
            console.print("[cyan]Using standard redaction message.[/cyan]")

        console.print(f"[green]Update operation finished.[/green]")
        console.print(f"  Processed: {result.get('processed', 0)}")
        console.print(f"  Successful: {result.get('success', 0)}")
        console.print(f"  Failed: {result.get('failures', 0)}")
        if result.get("successful_ids"):
            console.print("[bold blue]Successfully Updated:[/bold blue]")
            for cid in result["successful_ids"]: console.print(f"  - t1_{cid}")
        if result.get("failed_ids"):
            console.print("[bold red]Failed to Update:[/bold red]")
            for cid in result["failed_ids"]: console.print(f"  - t1_{cid}")

    except Exception as e:
         handle_exception(e, "Failed during update operation.", debug=config.get("debug", False))


class CLI:
    """Main CLI application class"""
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Reddit LLM PII & Sentiment Analysis Tool (Textual UI)",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        self.parser.add_argument('--version', action='version', version='%(prog)s 2.0 (Textual)')
        self.parser.add_argument('--debug', action='store_true', help='Enable debug logging (passed to UI)')

        # --- Add ALL optional config arguments to the main parser ---
        # These will be passed as initial values to the Textual UI
        self.parser.add_argument("--output-file", dest="output_file", help="Outputs analysis results to a file.")
        self.parser.add_argument("--enable-auth", dest="enable_auth", action=argparse.BooleanOptionalAction, help="Pre-check 'Enable Auth' in UI or force disable.") # Use BooleanOptionalAction
        self.parser.add_argument("--openai-key", dest="openai_key", help="OpenAI API key (passed to UI).")
        self.parser.add_argument("--local-llm", dest="local_llm", help="URL for local LLM endpoint (passed to UI).")
        # self.parser.add_argument("--openai-base", dest="openai_base", help="Optional OpenAI API base URL (handled by UI).") # Let UI handle this based on checkbox
        self.parser.add_argument("--model", dest="model", help="Preferred OpenAI or local LLM model (passed to UI).")
        self.parser.add_argument("--pii-only", dest="pii_only", action='store_true', help="Pre-check 'PII Only' in UI.")
        self.parser.add_argument("--use-random-string", dest="use_random_string", action='store_true', help="Use random UUID instead of standard message when updating comments.")
        self.parser.add_argument("--limit", dest="limit", type=int, help="Maximum comments to analyze (passed to UI).")
        self.parser.add_argument("--sort", dest="sort", choices=["hot", "new", "controversial", "top"], help="Sort method for comments (passed to UI).")
        self.parser.add_argument("--time", dest="time", choices=["all", "day", "hour", "month", "week", "year"], help="Time filter for comments (passed to UI).")
        self.parser.add_argument("--text-match", dest="text_match", help="Search comments containing text (passed to UI).")
        self.parser.add_argument("--skip-text", dest="skip_text", help="Skip comments containing text pattern (passed to UI).")
        self.parser.add_argument("--batch-size", dest="batch_size", type=int, help="Comments per batch for delete/update (passed to UI).")
        self.parser.add_argument("--use-openai-api", dest="use_openai_api", action='store_true', help="Pre-check 'Use OpenAI API' in UI.")

        # --- Subparsers for commands and their REQUIRED arguments ---
        self.subparsers = self.parser.add_subparsers(dest='command', help='Action to perform', required=True)

        # Listing command
        parser_listing = self.subparsers.add_parser('listing', help='Analyze a Reddit post and its comments')
        parser_listing.add_argument("subreddit", help="The subreddit (e.g., news)")
        parser_listing.add_argument("article", help="The ID of the article/post")
        parser_listing.add_argument("--limit", dest="limit", type=int, help="Maximum items to analyze.")
        parser_listing.add_argument("--sort", dest="sort", choices=["hot", "new", "controversial", "top"], help="Sort method for items.")
        parser_listing.add_argument("--time", dest="time", choices=["all", "day", "hour", "month", "week", "year"], help="Time filter for items.")
        parser_listing.add_argument("--text-match", dest="text_match", help="Search items containing text.")
        parser_listing.add_argument("--skip-text", dest="skip_text", help="Skip items containing text pattern.")
        parser_listing.add_argument("--pii-only", dest="pii_only", action='store_true', help="Only analyze for PII.")
        parser_listing.set_defaults(func=handle_listing)

        # User command
        parser_user = self.subparsers.add_parser('user', help="Analyze a user's comment history")
        parser_user.add_argument("username", help="The Reddit username")
        parser_user.add_argument("--limit", dest="limit", type=int, help="Maximum comments to analyze.")
        parser_user.add_argument("--sort", dest="sort", choices=["hot", "new", "controversial", "top"], help="Sort method for comments.")
        parser_user.add_argument("--time", dest="time", choices=["all", "day", "hour", "month", "week", "year"], help="Time filter for comments.")
        parser_user.add_argument("--text-match", dest="text_match", help="Search comments containing text.")
        parser_user.add_argument("--skip-text", dest="skip_text", help="Skip comments containing text pattern.")
        parser_user.add_argument("--pii-only", dest="pii_only", action='store_true', help="Only analyze for PII.")
        parser_user.set_defaults(func=handle_user)

        # Delete command
        parser_delete = self.subparsers.add_parser('delete', help='Delete specified Reddit comments')
        parser_delete.add_argument("comment_ids", help="Comma-separated list of comment IDs (e.g., abc123,def456)")
        parser_delete.set_defaults(func=handle_delete)

        # Update command
        parser_update = self.subparsers.add_parser('update', help='Replace comment content with "r/reddacted"')
        parser_update.add_argument("comment_ids", help="Comma-separated list of comment IDs")
        parser_update.set_defaults(func=handle_update)


    def run(self, argv: List[str]) -> int:
        """Run the CLI application using the Textual UI."""
        try:
            # 1. Initial Logging Setup (File + Console) based on --debug flag
            debug_mode = "--debug" in argv
            initial_log_level = logging.DEBUG if debug_mode else logging.INFO
            setup_logging(initial_log_level)
            logger.info_with_context(f"Initial logging level set to: {logging.getLevelName(initial_log_level)}")
            if debug_mode:
                logger.debug_with_context("Debug logging enabled via --debug flag.")

            # 2. Parse known arguments (needed for command dispatch)
            args = self.parser.parse_args(argv)
            logger.debug_with_context(f"Parsed command args: {args}")

            # 3. Prepare initial config for UI (Env Vars -> CLI Opts)
            initial_config = {}

            # 1. Load Environment Variables
            for env_var, config_key in ENV_VARS_MAP.items():
                value = os.getenv(env_var)
                if value:
                    initial_config[config_key] = value
                    if config_key in ["reddit_username", "reddit_password", "reddit_client_id", "reddit_client_secret"]:
                         if not initial_config.get("enable_auth_explicitly_set", False):
                               initial_config["enable_auth"] = True


            # 2. Load Optional CLI Arguments (override env vars)
            temp_parser = argparse.ArgumentParser(add_help=False)
            temp_parser.add_argument('--debug', action='store_true')
            temp_parser.add_argument("--output-file", dest="output_file")
            temp_parser.add_argument("--enable-auth", dest="enable_auth", action=argparse.BooleanOptionalAction)
            temp_parser.add_argument("--openai-key", dest="openai_key") # Added missing line back
            temp_parser.add_argument("--local-llm", dest="local_llm")
            temp_parser.add_argument("--model", dest="model")
            temp_parser.add_argument("--pii-only", dest="pii_only", action='store_true')
            temp_parser.add_argument("--limit", dest="limit", type=int)
            temp_parser.add_argument("--sort", dest="sort", choices=["hot", "new", "controversial", "top"])
            temp_parser.add_argument("--time", dest="time", choices=["all", "day", "hour", "month", "week", "year"])
            temp_parser.add_argument("--text-match", dest="text_match")
            temp_parser.add_argument("--skip-text", dest="skip_text")
            temp_parser.add_argument("--batch-size", dest="batch_size", type=int)
            temp_parser.add_argument("--use-openai-api", dest="use_openai_api", action='store_true')

            optional_args, _ = temp_parser.parse_known_args(argv)
            logger.debug_with_context(f"Parsed optional CLI args for UI: {vars(optional_args)}")
            # Update initial_config
            for key, value in vars(optional_args).items():
                if value is not None:
                    initial_config[key] = value
                    if key == "enable_auth":
                         initial_config["enable_auth_explicitly_set"] = True


            # Pass the initial debug mode state to the UI config
            initial_config["debug_logging"] = debug_mode
            logger.debug_with_context(f"Initial config for UI: {initial_config}")

            app = ConfigApp(initial_config=initial_config)
            final_config = app.run() # This blocks until the user submits or quits

            if final_config is None:
                console.print("[yellow]Configuration cancelled.[/yellow]")
                return 0

            logger.debug_with_context(f"Final config from UI: {final_config}")

            # 5. Set final logging level based on UI config
            ui_debug_mode = final_config.get("debug_logging", False)
            final_log_level = logging.DEBUG if ui_debug_mode else logging.INFO
            set_global_logging_level(final_log_level)
            logger.info_with_context(f"Logging level set to {logging.getLevelName(final_log_level)} based on final config.")

            # 6. Prepare final run_config by merging UI config and required args
            run_config = final_config.copy() # Start with UI values

            # Add required args from initial parse if not already in UI config
            if 'username' not in run_config and hasattr(args, 'username'):
                 run_config['username'] = args.username
            if 'subreddit' not in run_config and hasattr(args, 'subreddit'):
                 run_config['subreddit'] = args.subreddit
            if 'article' not in run_config and hasattr(args, 'article'):
                 run_config['article'] = args.article
            if 'comment_ids' not in run_config and hasattr(args, 'comment_ids'):
                 run_config['comment_ids'] = args.comment_ids

            # Ensure command and func are correctly set from initial parse
            run_config['command'] = args.command
            run_config['func'] = args.func
            run_config['debug'] = ui_debug_mode # Use the final debug state

            logger.debug_with_context(f"Final run_config before execution: {run_config}") # Add debug log

            # 7. Execute the command
            args.func(run_config) # Pass the correctly merged config

            return 0
        except Exception as e:
            # Determine debug status based on final config if available, else initial flag
            final_debug_status = False
            if 'final_config' in locals() and final_config is not None:
                final_debug_status = final_config.get("debug_logging", False)
            elif 'debug_mode' in locals(): # Fallback if UI was cancelled early
                final_debug_status = debug_mode

            command = argv[0] if argv else "unknown"
            # Use logger here instead of handle_exception directly for consistency?
            # For now, keep handle_exception but use the determined debug status
            # handle_exception below will log the error appropriately based on debug status
            handle_exception(e, f"Failed to execute command '{command}'", debug=final_debug_status)
            return 1

def main(argv: List[str] = sys.argv[1:]) -> int:
    """Main entry point"""
    # Handle --version before setting up logging or parsing fully
    if "--version" in argv:
         try:
             from reddacted.version import __version__
             print(f"reddacted {__version__}")
         except ImportError:
              print("reddacted version unknown (Textual UI)")
         return 0

    cli = CLI()
    return cli.run(argv)

if __name__ == "__main__":
    sys.exit(main())
