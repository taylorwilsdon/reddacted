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
                    # First check if Ollama is running
                    response = requests.get(base_url)
                    if response.status_code != 200:
                        console.print(f"[red]Error: Could not connect to Ollama at {base_url}[/]")
                        return 1

                    # Get available models
                    models_url = f"{base_url}/api/tags"  # Ollama's actual model list endpoint
                    response = requests.get(models_url)
                    if response.status_code == 200:
                        models_data = response.json()
                        available_models = [m['name'] for m in models_data.get('models', [])]
                        
                        console.print("\n[cyan]Available models:[/]")
                        for model in available_models:
                            console.print(f"  • {model}")
                        
                        # For local LLMs, ensure model name is properly formatted
                        model_name = args.openai_model
                        if model_name in available_models:
                            console.print(f"\n[green]Using model: {model_name}[/]")
                        else:
                            # Try without any prefix/suffix
                            base_model = model_name.split(':')[0].split('/')[-1]
                            if base_model in available_models:
                                model_name = base_model
                                console.print(f"\n[green]Using model: {model_name}[/]")
                            else:
                                console.print(f"\n[red]Error: Model '{args.openai_model}' not found in available models.[/]")
                                return 1
                    else:
                        console.print(f"[red]Error: Could not fetch available models: {response.status_code}[/]")
                        return 1
                except Exception as e:
                    console.print(f"[red]Error checking model availability: {str(e)}[/]")
                    return 1

                # For local LLMs, we need to set both api_key and api_key_path to bypass OpenAI's validation
                llm_config = {
                    'api_key': 'sk-not-needed',
                    'api_key_path': None,  # This helps bypass the key validation
                    'api_base': f"{base_url}/v1",  # Ollama expects /v1 prefix for OpenAI compatibility
                    'model': model_name,
                    'default_headers': {'User-Agent': 'Reddit-Sentiment-Analyzer'}
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
                    # First check if Ollama is running
                    response = requests.get(base_url)
                    if response.status_code != 200:
                        console.print(f"[red]Error: Could not connect to Ollama at {base_url}[/]")
                        return 1

                    # Get available models
                    models_url = f"{base_url}/api/tags"  # Ollama's actual model list endpoint
                    response = requests.get(models_url)
                    if response.status_code == 200:
                        models_data = response.json()
                        available_models = [m['name'] for m in models_data.get('models', [])]
                        
                        console.print("\n[cyan]Available models:[/]")
                        for model in available_models:
                            console.print(f"  • {model}")
                        
                        # For local LLMs, ensure model name is properly formatted
                        model_name = args.openai_model
                        if model_name in available_models:
                            console.print(f"\n[green]Using model: {model_name}[/]")
                        else:
                            # Try without any prefix/suffix
                            base_model = model_name.split(':')[0].split('/')[-1]
                            if base_model in available_models:
                                model_name = base_model
                                console.print(f"\n[green]Using model: {model_name}[/]")
                            else:
                                console.print(f"\n[red]Error: Model '{args.openai_model}' not found in available models.[/]")
                                return 1
                    else:
                        console.print(f"[red]Error: Could not fetch available models: {response.status_code}[/]")
                        return 1
                except Exception as e:
                    console.print(f"[red]Error checking model availability: {str(e)}[/]")
                    return 1

                # For local LLMs, we need to set both api_key and api_key_path to bypass OpenAI's validation
                llm_config = {
                    'api_key': 'sk-not-needed',
                    'api_key_path': None,  # This helps bypass the key validation
                    'api_base': f"{base_url}/v1",  # Ollama expects /v1 prefix for OpenAI compatibility
                    'model': model_name,
                    'default_headers': {'User-Agent': 'Reddit-Sentiment-Analyzer'}
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
