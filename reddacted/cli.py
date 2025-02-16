import sys
import getpass
import logging
from typing import Optional, Dict, Any

from cliff.app import App
from cliff.commandmanager import CommandManager
from cliff.command import Command
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.columns import Columns
from reddacted.utils.logging import get_logger, with_logging, set_global_logging_level
from reddacted.utils.exceptions import handle_exception
from reddacted.sentiment import Sentiment
from reddacted.api.reddit import Reddit
import requests

# Configure logging format
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO  # Set default level to INFO
)

logger = get_logger(__name__)
console = Console()


class ModifyComments(Command):
    """Base class for comment modification commands"""

    def get_description(self):
        return self.__doc__ or ''

    def get_parser(self, prog_name):
        parser = super(ModifyComments, self).get_parser(prog_name)
        parser.add_argument(
            'comment_ids',
            help='Comma-separated list of comment IDs to process'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=10,
            help='Number of comments to process per batch'
        )
        return parser

    def process_comments(self, parsed_args, action):
        api = Reddit()
        comment_ids = parsed_args.comment_ids.split(',')
        if action == 'delete':
            return api.delete_comments(comment_ids, batch_size=parsed_args.batch_size)
        elif action == 'update':
            return api.update_comments(comment_ids, batch_size=parsed_args.batch_size)
        
class DeleteComments(ModifyComments):
    """Delete specified comments"""

    def get_description(self):
        return 'Delete specified Reddit comments permanently using their IDs'

    def take_action(self, parsed_args):
        results = self.process_comments(parsed_args, 'delete')
        
        # Create detailed results panel
        details = []
        details.append(f"[cyan]Processed:[/] {results['processed']}")
        details.append(f"[green]Successful:[/] {results['success']}")
        details.append(f"[red]Failed:[/] {results['failures']}\n")
        
        if results.get('successful_ids'):
            details.append("[green]Successfully Deleted Comments:[/]")
            for comment_id in results['successful_ids']:
                details.append(f"  • [dim]t1_{comment_id}[/]")
        
        if results.get('failed_ids'):
            details.append("\n[red]Failed to Delete Comments:[/]")
            for comment_id in results['failed_ids']:
                details.append(f"  • [dim]t1_{comment_id}[/]")
        
        console.print(Panel(
            "\n".join(details),
            title="[bold red]Delete Results[/]",
            expand=False
        ))

class UpdateComments(ModifyComments):
    """Update specified comments to r/reddacted"""

    def get_description(self):
        return 'Replace comment content with "r/reddacted" using their IDs'

    def take_action(self, parsed_args):
        results = self.process_comments(parsed_args, 'update')
        
        # Create detailed results panel
        details = []
        details.append(f"[cyan]Processed:[/] {results['processed']}")
        details.append(f"[green]Successful:[/] {results['success']}")
        details.append(f"[red]Failed:[/] {results['failures']}\n")

        if results.get('successful_ids'):
            details.append("[green]Successfully Updated Comments:[/]")
            for comment_id in results['successful_ids']:
                details.append(f"  • [dim]t1_{comment_id}[/]")
        
        if results.get('failed_ids'):
            details.append("\n[red]Failed to Update Comments:[/]")
            for comment_id in results['failed_ids']:
                details.append(f"  • [dim]t1_{comment_id}[/]")
        
        console.print(Panel(
            "\n".join(details),
            title="[bold blue]Update Results[/]",
            expand=False
        ))


class BaseAnalyzeCommand(Command):
    """Base class for Reddit analysis commands with common arguments"""
    
    def get_parser(self, prog_name):
        parser = super(BaseAnalyzeCommand, self).get_parser(prog_name)
        # Common arguments for both Listing and User commands
        parser.add_argument('--output-file', '-o',
                            help='Outputs a file with information on each '
                                 'sentence of the post, as well as the final '
                                 'score.')
        parser.add_argument('--enable-auth', '-a', action='store_true',
                            help='Enable reddit api authentication by '
                            'using the environment variables '
                            'REDDIT_USERNAME '
                            'REDDIT_PASSWORD '
                            'REDDIT_CLIENT_ID '
                            'REDDIT_CLIENT_SECRET')
        parser.add_argument('--disable-pii', '-p', action='store_true',
                            help='Disable PII detection in the analysis')
        parser.add_argument('--openai-key', type=str,
                            help='OpenAI API key for LLM-based analysis')
        parser.add_argument('--local-llm', type=str,
                            help='URL for local LLM endpoint (OpenAI compatible)')
        parser.add_argument('--openai-base', type=str,
                            help='Optional OpenAI API base URL')
        parser.add_argument('--model', type=str,
                            help='OpenAI or local LLM model to use')
        parser.add_argument('--pii-only', action='store_true',
                            help='Only show comments that contain PII (0 < score < 1.0)')
        parser.add_argument('--limit', type=int, default=100,
                            help='Maximum number of comments to analyze (default: 100, use 0 for unlimited)')
        parser.add_argument('--sort', type=str, choices=['hot', 'top', 'new'], default='new',
                            help='Sort method for comments (default: new)')
        parser.add_argument('--time', type=str, 
                           choices=['all', 'day', 'hour', 'month', 'week', 'year'],
                           default='all',
                           help='Time filter for comments (default: all)')
        return parser


class Listing(BaseAnalyzeCommand):
    def get_description(self):
        return 'Analyze sentiment and detect PII in a Reddit post and its comments'

    def get_parser(self, prog_name):
        parser = super(Listing, self).get_parser(prog_name)
        parser.add_argument('subreddit', help='The subreddit.')
        parser.add_argument('article', help='The id of the article.')
        return parser

    def take_action(self, args):
        llm_config = CLI()._configure_llm(args, console)
        limit = None if args.limit == 0 else args.limit

        sent = Sentiment(
            auth_enabled=args.enable_auth,
            pii_enabled=not args.disable_pii,
            pii_only=args.pii_only,
            llm_config=llm_config,
            sort=args.sort,
            limit=limit
        )
        sent.get_sentiment('listing', f"{args.subreddit}/{args.article}", output_file=args.output_file)


class User(BaseAnalyzeCommand):
    def get_description(self):
        return 'Analyze sentiment and detect PII in a Reddit user\'s comment history'

    def get_parser(self, prog_name):
        parser = super(User, self).get_parser(prog_name)
        parser.add_argument('username', help='The name of the user.')
        return parser

    @with_logging(logger)
    def take_action(self, parsed_args):
        """Execute the user analysis command
        
        Args:
            parsed_args: Command line arguments
            
        Returns:
            None
            
        Raises:
            AttributeError: If required arguments are missing
            Exception: For other unexpected errors
        """
        logger.debug("Executing user analysis command")
        try:
            llm_config = CLI()._configure_llm(parsed_args, console)
            limit = None if parsed_args.limit == 0 else parsed_args.limit

            sent = Sentiment(
                auth_enabled=parsed_args.enable_auth,
                pii_enabled=not parsed_args.disable_pii,
                llm_config=llm_config,
                pii_only=parsed_args.pii_only,
                sort=parsed_args.sort,
                limit=limit
            )
            sent.get_sentiment(
                'user',
                parsed_args.username, 
                output_file=parsed_args.output_file,
                sort=parsed_args.sort,
                time_filter=parsed_args.time
            )
        except AttributeError as e:
            handle_exception(
                e,
                f"Missing or invalid arguments for user '{parsed_args.username}'\n" +
                "Required: username\nOptional: --output-file, --enable-auth, --disable-pii, --limit",
                debug="--debug" in sys.argv
            )
            raise
        except Exception as e:
            handle_exception(
                e,
                f"Failed to analyze user '{parsed_args.username}'\n" +
                "Check if the user exists and is not banned/private",
                debug="--debug" in sys.argv
            )
            raise

class CLI(App):
    def __init__(self):
        # Set debug logging if flag is present
        if '--debug' in sys.argv:
            set_global_logging_level(logging.DEBUG)
        else:
            set_global_logging_level(logging.INFO)
        
        command_manager = CommandManager('reddacted.commands')
        # Analysis commands
        command_manager.add_command('listing', Listing)
        command_manager.add_command('user', User)
        # Modification commands  
        command_manager.add_command('delete', DeleteComments)
        command_manager.add_command('update', UpdateComments)
        
        super(CLI, self).__init__(
            version=1.0,
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
                """,
            command_manager=command_manager,
            deferred_help=True,)

    @with_logging(logger)
    def _configure_llm(self, args, console):
        """Centralized LLM configuration handler"""
        logger.debug("Configuring LLM settings")
        if args.disable_pii:
            return None

        if args.local_llm:
            base_url = args.local_llm.rstrip('/v1')
            console.print(f"[blue]Using local LLM endpoint: {base_url}[/]")

            try:
                # Verify Ollama connection
                response = requests.get(base_url)
                if response.status_code != 200:
                    console.print(f"[red]Error: Could not connect to Ollama at {base_url}[/]")
                    return None

                # Get available models
                models_url = f"{base_url}/api/tags"
                response = requests.get(models_url)
                if response.status_code != 200:
                    console.print(f"[red]Error: Could not fetch models: {response.status_code}[/]")
                    return None

                models_data = response.json()
                available_models = [m['name'] for m in models_data.get('models', [])]
                args.model = args.model or available_models[0]

                if args.model not in available_models:
                    console.print(f"[red]Error: Model '{args.model}' not available[/]")
                    return None

                return {
                    'api_key': 'sk-not-needed',
                    'api_base': f"{base_url}/v1",
                    'model': args.model,
                    'default_headers': {'User-Agent': 'Reddit-Sentiment-Analyzer'}
                }

            except Exception as e:
                console.print(f"[red]Connection error: {str(e)}[/]")
                return None

        elif args.openai_key:
            return {
                'api_key': args.openai_key,
                'api_base': args.openai_base or "https://api.openai.com/v1",
                'model': args.model or "gpt-4"
            }

        # Prompt for configuration if none provided
        console.print("[yellow]LLM required for PII detection[/]")
        llm_choice = Prompt.ask(
            "Choose LLM provider",
            choices=["local", "openai"],
            default="local"
        )

        if llm_choice == "openai":
            args.openai_key = getpass.getpass("Enter your OpenAI API key: ")
            return {
                'api_key': args.openai_key,
                'api_base': args.openai_base or "https://api.openai.com/v1",
                'model': args.model or "gpt-4"
            }
        else:
            args.local_llm = Prompt.ask(
                "Enter local LLM endpoint URL",
                default="http://localhost:11434"
            )
            return self._configure_llm(args, console)


def suggest_command(input_command):
    """Suggests the closest matching command with a fun message"""
    commands = {
        'listing': 'Analyze a Reddit post and its comments',
        'user': 'Analyze a Reddit user\'s comment history',
        'delete': 'Delete comments by ID',
        'update': 'Replace comment content with r/reddacted'
    }
    
    # Map common variations to actual commands
    command_map = {
        'post': 'listing',
        'thread': 'listing',
        'article': 'listing',
        'comments': 'listing',
        'redditor': 'user',
        'profile': 'user',
        'history': 'user',
        'remove': 'delete',
        'del': 'delete',
        'rm': 'delete',
        'edit': 'update',
        'redact': 'update',
        'modify': 'update',
        'change': 'update'
    }
    
    input_command = input_command.lower()
    
    # Direct command match
    if input_command in commands:
        return None
        
    # Check mapped variations
    if input_command in command_map:
        actual_command = command_map[input_command]
        return (f"🤔 Ah, you probably meant '{actual_command}'! That's what we call it around here.\n"
                f"💡 This command will: {commands[actual_command]}")
                
    # Find closest match
    import difflib
    all_commands = list(commands.keys()) + list(command_map.keys())
    matches = difflib.get_close_matches(input_command, all_commands, n=1, cutoff=0.6)
    
    if matches:
        matched = matches[0]
        actual = matched if matched in commands else command_map[matched]
        return (f"🎯 Close! Did you mean '{actual}'?\n"
                f"💡 This command will: {commands[actual]}")
    
    # No close match found
    return (f"🤖 Hmm, I don't recognize '{input_command}'.\n"
            "Here's what I can help you with:\n"
            "📊 'listing' - Analyze a Reddit post\n"
            "👤 'user' - Analyze a user's history\n"
            "🗑️ 'delete' - Remove comments\n"
            "✏️ 'update' - Redact comments\n"
            "\nTry one of these!")

def main(argv=sys.argv[1:]):
    try:
        app = CLI()
        
        return app.run(argv)
    except Exception as e:
        from reddacted.utils.exceptions import handle_exception
        command = argv[0] if argv else "unknown"
        handle_exception(
            e,
            f"Failed to execute command '{command}'",
            debug="--debug" in argv
        )
        return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
