import sys
import getpass

from cliff.app import App
from cliff.commandmanager import CommandManager
from cliff.command import Command
from rich.console import Console
from rich.prompt import Prompt

from reddact.sentiment import Sentiment

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
                            help='OpenAI or local LLM model to use')
        parser.add_argument('--pii-only', action='store_true',
                            help='Only show comments that contain PII (0 < score < 1.0)')
        parser.add_argument('--limit', type=int, default=100,
                            help='Maximum number of comments to analyze (default: 100, use 0 for unlimited)')
        parser.add_argument('--limit', type=int, default=100,
                            help='Maximum number of comments to analyze (default: 100, use 0 for unlimited)')
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
                        
                        # Create a panel with available models
                        from rich.panel import Panel
                        from rich.columns import Columns
                        
                        # For local LLMs, ensure model name is properly formatted
                        model_name = args.openai_model if args.openai_model else available_models[0]
                        if model_name in available_models:
                            # Create a subtle panel for available models
                            model_list = "\n".join(
                                f"• [green]{model}[/] [dim](active)[/]" if model == model_name else f"• {model}"
                                for model in available_models
                            )
                            models_panel = Panel(
                                model_list,
                                title="[dim]Available models[/]",
                                border_style="dim",
                                padding=(0, 1)
                            )
                            console.print(models_panel)
                        else:
                            # Try without any prefix/suffix
                            base_model = model_name.split(':')[0].split('/')[-1]
                            if base_model in available_models:
                                model_name = base_model
                                console.print("\n[dim]Available models:[/]")
                                for model in available_models:
                                    if model == model_name:
                                        console.print(f"  • [green]{model}[/] [dim](active)[/]")
                                    else:
                                        console.print(f"  • {model}")
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
            elif not args.openai_key and not args.local_llm:
                console.print("[yellow]No LLM configuration provided.[/]")
                llm_choice = Prompt.ask(
                    "Choose LLM provider",
                    choices=["openai", "local"],
                    default="openai"
                )
                if llm_choice == "openai":
                    args.openai_key = getpass.getpass("Enter your OpenAI API key: ")
                else:
                    args.local_llm = Prompt.ask(
                        "Enter local LLM endpoint URL",
                        default="http://localhost:11434"
                    )
                    
                    # Check connection and get available models
                    base_url = args.local_llm.rstrip('/v1')
                    try:
                        response = requests.get(base_url)
                        if response.status_code != 200:
                            console.print(f"[red]Error: Could not connect to Ollama at {base_url}[/]")
                            return 1

                        models_url = f"{base_url}/api/tags"
                        response = requests.get(models_url)
                        if response.status_code == 200:
                            models_data = response.json()
                            available_models = [m['name'] for m in models_data.get('models', [])]
                            
                            if not available_models:
                                console.print("[red]Error: No models found on the local LLM server[/]")
                                return 1
                            
                            # Show available models in a panel
                            model_list = "\n".join(f"• {model}" for model in available_models)
                            console.print(Panel(
                                model_list,
                                title="[cyan]Available Models[/]",
                                border_style="dim",
                                padding=(0, 1)
                            ))
                            
                            # Prompt for model selection
                            args.openai_model = Prompt.ask(
                                "\nSelect model",
                                choices=available_models,
                                default=available_models[0]
                            )
                        else:
                            console.print(f"[red]Error: Could not fetch available models: {response.status_code}[/]")
                            return 1
                    except Exception as e:
                        console.print(f"[red]Error checking model availability: {str(e)}[/]")
                        return 1
                    
                    # Recursively call the LLM setup logic for local LLM
                    return self.take_action(args)
        
        if args.openai_key:
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
        # Convert limit of 0 to None for unlimited
        limit = None if args.limit == 0 else args.limit
        sent.get_listing_sentiment(args.subreddit,
                                   args.article,
                                   args.output_file,
                                   limit=limit)


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
                            help='OpenAI or local LLM model to use')
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
                        
                        # Create a panel with available models
                        from rich.panel import Panel
                        from rich.columns import Columns
                        
                        # For local LLMs, ensure model name is properly formatted
                        model_name = args.openai_model if args.openai_model else available_models[0]
                        if model_name in available_models:
                            # Create a subtle panel for available models
                            model_list = "\n".join(
                                f"• [green]{model}[/] [dim](active)[/]" if model == model_name else f"• {model}"
                                for model in available_models
                            )
                            models_panel = Panel(
                                model_list,
                                title="[dim]Available models[/]",
                                border_style="dim",
                                padding=(0, 1)
                            )
                            console.print(models_panel)
                        else:
                            # Try without any prefix/suffix
                            base_model = model_name.split(':')[0].split('/')[-1]
                            if base_model in available_models:
                                model_name = base_model
                                console.print("\n[dim]Available models:[/]")
                                for model in available_models:
                                    if model == model_name:
                                        console.print(f"  • [green]{model}[/] [dim](active)[/]")
                                    else:
                                        console.print(f"  • {model}")
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
            elif not args.openai_key and not args.local_llm:
                console.print("[yellow]No LLM configuration provided.[/]")
                llm_choice = Prompt.ask(
                    "Choose LLM provider",
                    choices=["openai", "local"],
                    default="openai"
                )
                if llm_choice == "openai":
                    args.openai_key = getpass.getpass("Enter your OpenAI API key: ")
                else:
                    args.local_llm = Prompt.ask(
                        "Enter local LLM endpoint URL",
                        default="http://localhost:11434"
                    )
                    
                    # Check connection and get available models
                    base_url = args.local_llm.rstrip('/v1')
                    try:
                        response = requests.get(base_url)
                        if response.status_code != 200:
                            console.print(f"[red]Error: Could not connect to Ollama at {base_url}[/]")
                            return 1

                        models_url = f"{base_url}/api/tags"
                        response = requests.get(models_url)
                        if response.status_code == 200:
                            models_data = response.json()
                            available_models = [m['name'] for m in models_data.get('models', [])]
                            
                            if not available_models:
                                console.print("[red]Error: No models found on the local LLM server[/]")
                                return 1
                            
                            # Show available models in a panel
                            model_list = "\n".join(f"• {model}" for model in available_models)
                            console.print(Panel(
                                model_list,
                                title="[cyan]Available Models[/]",
                                border_style="dim",
                                padding=(0, 1)
                            ))
                            
                            # Prompt for model selection
                            args.openai_model = Prompt.ask(
                                "\nSelect model",
                                choices=available_models,
                                default=available_models[0]
                            )
                        else:
                            console.print(f"[red]Error: Could not fetch available models: {response.status_code}[/]")
                            return 1
                    except Exception as e:
                        console.print(f"[red]Error checking model availability: {str(e)}[/]")
                        return 1
                    
                    # Recursively call the LLM setup logic for local LLM
                    return self.take_action(args)
        
        if args.openai_key:
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
        # Convert limit of 0 to None for unlimited
        limit = None if args.limit == 0 else args.limit
        sent.get_user_sentiment(args.username, args.output_file, limit=limit)


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
