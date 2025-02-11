import sys
import getpass

from cliff.app import App
from cliff.commandmanager import CommandManager
from cliff.command import Command
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.columns import Columns

from reddacted.sentiment import Sentiment

import requests


console = Console()


class Listing(Command):

    def get_description(self):
        return 'get the sentiment score of a post.'

    def get_parser(self, prog_name):
        parser = super(Listing, self).get_parser(prog_name)
        parser.add_argument('subreddit', help='The subreddit.')
        parser.add_argument('article', help='The id of the article.')
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
        return parser

    def take_action(self, args):
        llm_config = CLI()._configure_llm(args, console)
        limit = None if args.limit == 0 else args.limit

        sent = Sentiment(
            auth_enabled=args.enable_auth,
            pii_enabled=not args.disable_pii,
            llm_config=llm_config,
            pii_only=args.pii_only,
            limit=limit
        )
        sent.get_listing_sentiment(args.subreddit, args.article, args.output_file)


class User(Command):

    def get_description(self):
        return 'get the sentiment score of a user.'

    def get_parser(self, prog_name):
        parser = super(User, self).get_parser(prog_name)
        parser.add_argument('username', help='The name of the user.')
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
        return parser

    def take_action(self, args):
        llm_config = CLI()._configure_llm(args, console)
        limit = None if args.limit == 0 else args.limit

        sent = Sentiment(
            auth_enabled=args.enable_auth,
            pii_enabled=not args.disable_pii,
            llm_config=llm_config,
            pii_only=args.pii_only,
            limit=limit
        )
        sent.get_user_sentiment(args.username, args.output_file)


class CLI(App):
    def __init__(self):
        super(CLI, self).__init__(
            version=1.0,
            description="Obtains Sentiment Score of various reddit objects.",
            command_manager=CommandManager('reddit.sentiment'),
            deferred_help=True,)

    def _configure_llm(self, args, console):
        """Centralized LLM configuration handler"""
        llm_config = None
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
            choices=["openai", "local"],
            default="openai"
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


def main(argv=sys.argv[1:]):
    app = CLI()
    return app.run(argv)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
