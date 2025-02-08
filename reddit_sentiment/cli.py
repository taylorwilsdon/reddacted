import sys
import getpass

from cliff.app import App
from cliff.commandmanager import CommandManager
from cliff.command import Command
from rich.console import Console
from rich.prompt import Prompt

from reddit_sentiment.sentiment import Sentiment

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
        parser.add_argument('--openai-model', type=str,
                            default='gpt-3.5-turbo',
                            help='OpenAI model to use (default: gpt-3.5-turbo)')
        parser.add_argument('--pii-only', action='store_true',
                            help='Only show comments that contain PII (0 < score < 1.0)')
        return parser

    def take_action(self, args):
        llm_config = None
        
        # Handle LLM configuration
        if not args.disable_pii:
            if args.local_llm:
                # Remove trailing /v1 if present
                base_url = args.local_llm.rstrip('/v1')
                console.print(f"[blue]Using local LLM endpoint: {base_url}[/]")
                
                # Check model availability
                import requests
                try:
                    models_url = f"{base_url}/v1/models"
                    response = requests.get(models_url)
                    if response.status_code == 200:
                        models = response.json()
                        available_models = [m['id'] for m in models.get('data', [])]
                        if args.openai_model not in available_models:
                            console.print(f"[yellow]Warning: Model '{args.openai_model}' not found in available models:[/]")
                            console.print("[cyan]Available models:[/]")
                            for model in available_models:
                                console.print(f"  • {model}")
                    else:
                        console.print(f"[yellow]Warning: Could not fetch available models: {response.status_code}[/]")
                except Exception as e:
                    console.print(f"[yellow]Warning: Error checking model availability: {str(e)}[/]")
                
                llm_config = {
                    'api_key': 'not-needed',
                    'api_base': base_url,
                    'model': args.openai_model
                }
            elif not args.openai_key:
                console.print("[yellow]No OpenAI API key provided.[/]")
                if Prompt.ask("Would you like to enable LLM-based PII analysis?", choices=["y", "n"], default="y") == "y":
                    args.openai_key = getpass.getpass("Enter your OpenAI API key: ")
        
        if args.openai_key and not args.local_llm:
            llm_config = {
                'api_key': args.openai_key,
                'api_base': args.openai_base,
                'model': args.openai_model
            }
        
        sent = Sentiment(
            auth_enabled=args.enable_auth,
            pii_enabled=not args.disable_pii,
            llm_config=llm_config,
            pii_only=args.pii_only
        )
        sent.get_listing_sentiment(args.subreddit,
                                   args.article,
                                   args.output_file)


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
        parser.add_argument('--openai-model', type=str,
                            default='gpt-3.5-turbo',
                            help='OpenAI model to use (default: gpt-3.5-turbo)')
        parser.add_argument('--pii-only', action='store_true',
                            help='Only show comments that contain PII (0 < score < 1.0)')
        return parser

    def take_action(self, args):
        llm_config = None
        
        # Handle LLM configuration
        if not args.disable_pii:
            if args.local_llm:
                # Remove trailing /v1 if present
                base_url = args.local_llm.rstrip('/v1')
                console.print(f"[blue]Using local LLM endpoint: {base_url}[/]")
                
                # Check model availability
                import requests
                try:
                    models_url = f"{base_url}/v1/models"
                    response = requests.get(models_url)
                    if response.status_code == 200:
                        models = response.json()
                        available_models = [m['id'] for m in models.get('data', [])]
                        if args.openai_model not in available_models:
                            console.print(f"[yellow]Warning: Model '{args.openai_model}' not found in available models:[/]")
                            console.print("[cyan]Available models:[/]")
                            for model in available_models:
                                console.print(f"  • {model}")
                    else:
                        console.print(f"[yellow]Warning: Could not fetch available models: {response.status_code}[/]")
                except Exception as e:
                    console.print(f"[yellow]Warning: Error checking model availability: {str(e)}[/]")
                
                llm_config = {
                    'api_key': 'not-needed',
                    'api_base': base_url,
                    'model': args.openai_model
                }
            elif not args.openai_key:
                console.print("[yellow]No OpenAI API key provided.[/]")
                if Prompt.ask("Would you like to enable LLM-based PII analysis?", choices=["y", "n"], default="y") == "y":
                    args.openai_key = getpass.getpass("Enter your OpenAI API key: ")
        
        if args.openai_key and not args.local_llm:
            llm_config = {
                'api_key': args.openai_key,
                'api_base': args.openai_base,
                'model': args.openai_model
            }
        
        sent = Sentiment(
            auth_enabled=args.enable_auth,
            pii_enabled=not args.disable_pii,
            llm_config=llm_config,
            pii_only=args.pii_only
        )
        sent.get_user_sentiment(args.username, args.output_file)


class CLI(App):
    def __init__(self):
        super(CLI, self).__init__(
            version=1.0,
            description="Obtains Sentiment Score of various reddit objects.",
            command_manager=CommandManager('reddit.sentiment'),
            deferred_help=True,)


def main(argv=sys.argv[1:]):
    app = CLI()
    return app.run(argv)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
