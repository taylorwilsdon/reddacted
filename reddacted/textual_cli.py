import json
import os
import os.path
from typing import Optional, Dict, Any # Added Dict, Any
from textual import on
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll, Horizontal, Container
from textual.validation import Number, Regex
from textual.widgets import Input, Label, Pretty, Checkbox, Select, Button
from textual import work

from reddacted.utils.logging import get_logger
from reddacted.styles import TEXTUAL_CSS
from reddacted.api.list_models import fetch_available_models, ModelFetchError

VALID_SORT_OPTIONS = ["hot", "new", "controversial", "top"]
VALID_TIME_OPTIONS = ["all", "day", "hour", "month", "week", "year"]

URL_REGEX = r"^(http|https)://[^\s/$.?#].[^\s]*$"

# Environment Variable Keys
ENV_VARS_MAP = {
    "REDDIT_USERNAME": "reddit_username",
    "REDDIT_PASSWORD": "reddit_password",
    "REDDIT_CLIENT_ID": "reddit_client_id",
    "REDDIT_CLIENT_SECRET": "reddit_client_secret",
    "OPENAI_API_KEY": "openai_key",
    # Add other relevant env vars if needed
}

logger = get_logger(__name__) # Initialize logger globally

class ConfigApp(App):
    # Combine shared styles with screen-specific styles
    CSS = TEXTUAL_CSS + """

    VerticalScroll {
        height: 100%; /* Limit height */
        border: round $accent; /* Keep border */
        padding: 1 2; /* Keep padding */
        margin: 1 2; /* Use margin for spacing from screen edges */
    }
    Input.-valid {
        border: tall $success 60%;
    }
    Input.-valid:focus {
        border: tall $success;
    }
    Input.-invalid {
        border: tall $error 60%;
    }
    Input.-invalid:focus {
        border: tall $error;
    }
    Input {
        margin-bottom: 1; /* Spacing below inputs */
    }
    Label {
        margin: 1 0 0 0; /* Spacing above labels */
    }
    Pretty#validation-summary { /* Target validation summary specifically */
        margin-top: 1;
        border: thick $error; /* Use error color for border */
        width: 100%;
        height: auto;
        max-height: 10; /* Limit error display height */
        display: none; /* Hide by default */
    }
    #boolean-options {
        height: auto;
        margin-bottom: 1;
        align: center top; /* Center checkboxes horizontally, align top vertically */
    }
    #boolean-options > Checkbox {
        margin-right: 2;
        width: auto;
    }
    #output_file_container {
        display: none; /* Hide by default */
        height: auto; /* Allow height to adjust to content */
        margin-top: 1; /* Add some space when visible */
    }
    #reddit_auth_container {
        display: none; /* Hide by default */
        height: auto; /* Allow height to adjust to content */
        margin-top: 1; /* Add some space when visible */
        border: round $accent; /* Optional: Add border for visual grouping */
        padding: 0 1; /* Optional: Add padding */
        margin-bottom: 1; /* Add space below */
    }
    #llm-url-row, #select-row, #key-model-row { /* Added #key-model-row */
        height: auto;
        margin-bottom: 1;
    }
    .url-input-group, .select-group, .key-model-group { /* Added .key-model-group */
        width: 1fr; /* Distribute space equally */
        margin-right: 2; /* Add space between groups */
        height: auto; /* Allow height to adjust to content */
    }
    .url-input-group:last-of-type, .select-group:last-of-type, .key-model-group:last-of-type { /* Added .key-model-group */
        margin-right: 0; /* No margin on the last item */
    }
    /* Keep url-input-group specific rule as it might differ later */
    .url-input-group {
        width: 1fr;
        height: auto;
    }
    Select {
        width: 100%; /* Make select widgets fill their container */
    }
    #limit-batch-row { /* New ID for the row */
        height: auto;
        margin-bottom: 1; /* Consistent spacing */
    }
    .number-input-group { /* New class for the groups */
        width: 1fr; /* Distribute space equally */
        margin-right: 2; /* Add space between groups */
        height: auto; /* Allow height to adjust to content */
    }
    .number-input-group:last-of-type { /* No margin on the last item */
        margin-right: 0;
    }
    /* Removed CSS for #openai-api-options as it's no longer used */
    #submit-button {
        width: 100%;
        margin-top: 1; /* Add space above the button */
    }
    #button-row {
        height: auto;
        align: center top; /* Center buttons horizontally, align top vertically */
        margin-top: 1;
    }
    #button-row > Button {
        width: auto; /* Allow buttons to size based on content */
        margin-left: 1; /* Add space between buttons */
    }
    """ # Keep existing CSS

    # Add binding for quitting
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, initial_config: Optional[Dict[str, Any]] = None, *args, **kwargs):
        """
        Initialize the ConfigApp.

        Args:
            initial_config: Optional dictionary with initial values from CLI/env vars.
        """
        logger.debug_with_context(f"Initial config: {initial_config}") # Use global logger with context
        
        super().__init__(*args, **kwargs)
        self.initial_config = initial_config or {}
        # Pre-process initial_config: Convert potential boolean strings/numbers to actual booleans
        for key, value in self.initial_config.items():
             if isinstance(value, str):
                 if value.lower() in ('true', '1', 'yes'):
                     self.initial_config[key] = True
                 elif value.lower() in ('false', '0', 'no'):
                     self.initial_config[key] = False
             elif isinstance(value, int) and key in ["enable_auth", "pii_only", "use_openai_api", "write_to_file"]: # Add boolean keys here
                 self.initial_config[key] = bool(value)
        logger.debug_with_context(f"Processed config: {self.initial_config}") # Use global logger with context


    def compose(self) -> ComposeResult:
        with VerticalScroll():
            with Horizontal(id="boolean-options"):
                yield Checkbox("Enable Auth", id="enable_auth")
                yield Checkbox("PII Only", id="pii_only")
                yield Checkbox("Use OpenAI API", id="openai_api_checkbox")
                yield Checkbox("Write to File", id="write_to_file_checkbox")
            with Container(id="output_file_container"):
                yield Label("Output File Path:")
                yield Input(placeholder="e.g., /path/to/results.txt", id="output_file")

            # Reddit Auth Inputs (Conditional)
            with Container(id="reddit_auth_container"):
                yield Label("Reddit Username:")
                yield Input(placeholder="Your Reddit username", id="reddit_username")
                yield Label("Reddit Password:")
                yield Input(placeholder="Your Reddit password", password=True, id="reddit_password")
                yield Label("Reddit Client ID:")
                yield Input(placeholder="Your Reddit app client ID", id="reddit_client_id")
                yield Label("Reddit Client Secret:")
                yield Input(placeholder="Your Reddit app client secret", password=True, id="reddit_client_secret")

            # OpenAI Key / Model Name Row
            with Horizontal(id="key-model-row"):
                with Container(classes="key-model-group"):
                    yield Label("OpenAI API Key:")
                    yield Input(placeholder="Enter your key (optional)", password=True, id="openai_key")
                with Container(classes="key-model-group"):
                    yield Label("LLM Model:")
                    yield Select(
                        [],
                        prompt="Enter URL/Key first...",
                        id="model_select",
                        allow_blank=True,
                        disabled=True
                    )

            with Container(classes="url-input-group"):
                yield Label("LLM URL:")
                yield Input(
                    value="http://localhost:11434",
                    validators=[Regex(URL_REGEX, failure_description="Must be a valid URL (http/https)")],
                    id="local_llm",
                )
            with Horizontal(id="limit-batch-row"):
                with Container(classes="number-input-group"):
                    yield Label("Comment Limit (0 for unlimited):")
                    yield Input(
                        placeholder="Default: 100",
                        validators=[Number(minimum=0)],
                        id="limit",
                        type="integer",
                    )
                with Container(classes="number-input-group"):
                    yield Label("Batch Size (for delete/update):")
                    yield Input(
                        placeholder="Default: 10",
                        validators=[Number(minimum=1)],
                        id="batch_size",
                        type="integer",
                    )

            # Sort/Time Select Row
            with Horizontal(id="select-row"):
                 with Container(classes="select-group"):
                    yield Label("Sort Order:")
                    yield Select(
                        [(option.capitalize(), option) for option in VALID_SORT_OPTIONS],
                        prompt="Select Sort...",
                        value="new",
                        id="sort",
                        allow_blank=False,
                    )
                 with Container(classes="select-group"):
                    yield Label("Time Filter:")
                    yield Select(
                        [(option.capitalize(), option) for option in VALID_TIME_OPTIONS],
                        prompt="Select Time...",
                        value="all",
                        id="time",
                        allow_blank=False,
                    )

            yield Label("Text Match (requires auth):")
            yield Input(placeholder="Search comment text (optional)", id="text_match")

            yield Label("Skip Text Pattern:")
            yield Input(placeholder="Regex to skip comments (optional)", id="skip_text")


            yield Pretty([], id="validation-summary")

            with Horizontal(id="button-row"):
                yield Button("Save Configuration", id="save-button", variant="success")
                yield Button("Submit Configuration", id="submit-button", variant="primary")
                yield Button("Quit", id="quit-button", variant="error")


    @on(Checkbox.Changed, "#write_to_file_checkbox")
    def toggle_output_file_path(self, event: Checkbox.Changed) -> None:
        """Show or hide the output file path input based on the checkbox."""
        output_container = self.query_one("#output_file_container")
        output_container.display = event.value

    @on(Checkbox.Changed, "#enable_auth")
    def toggle_reddit_auth_inputs(self, event: Checkbox.Changed) -> None:
        """Show or hide the Reddit auth input fields based on the checkbox."""
        auth_container = self.query_one("#reddit_auth_container")
        auth_container.display = event.value

    @on(Input.Changed)
    def show_invalid_reasons(self, event: Input.Changed) -> None:
        """Update the validation summary with the focused input's errors."""
        summary = self.query_one("#validation-summary", Pretty)
        if event.validation_result is not None:
            is_valid = event.validation_result.is_valid
            if not is_valid:
                summary.update(event.validation_result.failure_descriptions)
                summary.display = True
            else:
                if self.focused == event.input:
                    summary.update([])
                    summary.display = False
        else:
             if self.focused == event.input:
                summary.update([])
                summary.display = False

    @on(Input.Submitted)
    def input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission - could trigger full submit or just move focus."""
        self.bell()
        if event.validation_result and not event.validation_result.is_valid:
            self.query_one(Pretty).update(event.validation_result.failure_descriptions)
        else:
            pass

    CONFIG_FILE = "config.json"

    def on_mount(self) -> None:
        """Called when the app is mounted. Loads config and fetches initial models."""
        logger.info_with_context("App mounting - starting configuration load") # Use global logger with context
        
        try:
            self.load_configuration()
            logger.info_with_context("Configuration loaded successfully") # Use global logger with context
            
            llm_url_input = self.query_one("#local_llm", Input)
            openai_key_input = self.query_one("#openai_key", Input)
            openai_checkbox = self.query_one("#openai_api_checkbox", Checkbox)
            
            logger.debug_with_context(f"LLM URL: {llm_url_input.value}, OpenAI Checkbox: {openai_checkbox.value}") # Use global logger with context
            
            if openai_checkbox.value and openai_key_input.value:
                logger.info_with_context("Fetching OpenAI models") # Use global logger with context
                self.fetch_models_worker(llm_url_input.value, openai_key_input.value)
            elif not openai_checkbox.value and llm_url_input.value and llm_url_input.is_valid:
                logger.info_with_context("Fetching local LLM models") # Use global logger with context
                self.fetch_models_worker(llm_url_input.value)
            else:
                logger.info_with_context("Skipping model fetch - conditions not met") # Use global logger with context
        except Exception as e:
            logger.error_with_context(f"Error during app mount: {str(e)}", exc_info=True) # Use global logger with context
            raise

    def load_configuration(self) -> None:
        """Loads configuration from file and merges initial config from CLI/env."""
        config_values = {}
        # Load from file if exists
        if os.path.exists(self.CONFIG_FILE):
            try:
                with open(self.CONFIG_FILE, "r") as f:
                    config_values = json.load(f)
                self.app.notify("Configuration loaded from file.", title="Config Load")
            except json.JSONDecodeError:
                self.app.notify(f"Error decoding '{self.CONFIG_FILE}'. Using defaults.", severity="error", title="Config Load Error")
                config_values = {}
            except Exception as e:
                self.app.notify(f"Error loading config file: {e}", severity="error", title="Config Load Error")
                config_values = {}
        else:
             self.app.notify("No configuration file found. Using defaults.", title="Config Load")

        # Merge initial_config (CLI/env vars) - these take precedence
        processed_initial_config = {}
        for key, value in self.initial_config.items():
            if isinstance(value, str):
                if value.lower() in ('true', '1', 'yes'):
                    processed_initial_config[key] = True
                elif value.lower() in ('false', '0', 'no'):
                    processed_initial_config[key] = False
                else:
                    processed_initial_config[key] = value
            elif isinstance(value, int) and key in ["enable_auth", "pii_only", "use_openai_api", "write_to_file"]:
                 processed_initial_config[key] = bool(value)
            else:
                 processed_initial_config[key] = value

        config_values.update(processed_initial_config)

        # Populate UI elements using the final merged config_values
        try:
            # Populate Checkboxes
            write_cb = self.query_one("#write_to_file_checkbox", Checkbox)
            write_cb.value = config_values.get("write_to_file", False)
            self.call_later(self.toggle_output_file_path, Checkbox.Changed(write_cb, write_cb.value))

            auth_cb = self.query_one("#enable_auth", Checkbox)
            auth_cb.value = config_values.get("enable_auth", False)
            self.call_later(self.toggle_reddit_auth_inputs, Checkbox.Changed(auth_cb, auth_cb.value))

            self.query_one("#pii_only", Checkbox).value = config_values.get("pii_only", False)

            openai_cb = self.query_one("#openai_api_checkbox", Checkbox)
            openai_cb.value = config_values.get("use_openai_api", False)
            self.call_later(self.update_llm_url_for_openai, Checkbox.Changed(openai_cb, openai_cb.value))


            # Populate Inputs
            if write_cb.value:
                self.query_one("#output_file", Input).value = str(config_values.get("output_file", ""))

            if auth_cb.value:
                self.query_one("#reddit_username", Input).value = str(config_values.get("reddit_username", ""))
                self.query_one("#reddit_password", Input).value = str(config_values.get("reddit_password", ""))
                self.query_one("#reddit_client_id", Input).value = str(config_values.get("reddit_client_id", ""))
                self.query_one("#reddit_client_secret", Input).value = str(config_values.get("reddit_client_secret", ""))

            self.query_one("#openai_key", Input).value = str(config_values.get("openai_key", ""))
            llm_input = self.query_one("#local_llm", Input)
            if openai_cb.value:
                 llm_input.value = str(config_values.get("local_llm", "https://api.openai.com/v1"))
                 llm_input.disabled = True
            else:
                 llm_input.value = str(config_values.get("local_llm", "http://localhost:11434"))
                 llm_input.disabled = False


            self.query_one("#limit", Input).value = str(config_values.get("limit", ""))
            self.query_one("#batch_size", Input).value = str(config_values.get("batch_size", ""))
            self.query_one("#text_match", Input).value = str(config_values.get("text_match", ""))
            self.query_one("#skip_text", Input).value = str(config_values.get("skip_text", ""))

            # Populate Selects
            self.query_one("#sort", Select).value = config_values.get("sort", "new")
            self.query_one("#time", Select).value = config_values.get("time", "all")

        except Exception as e:
             self.app.notify(f"Error applying loaded/initial config to UI: {e}", severity="error", title="UI Populate Error")


    @work(exclusive=True, thread=True)
    def fetch_models_worker(self, base_url: str, api_key: Optional[str] = None) -> None:
        """Worker to fetch models in the background."""
        logger.info_with_context(f"Starting model fetch from {base_url}") # Use global logger with context
        
        model_select = self.query_one("#model_select", Select)
        intended_model = self.initial_config.get("model", None)
        logger.debug_with_context(f"Intended model from config: {intended_model}") # Use global logger with context
        
        if intended_model is None and os.path.exists(self.CONFIG_FILE):
             try:
                 logger.debug_with_context(f"Loading model from config file: {self.CONFIG_FILE}") # Use global logger with context
                 with open(self.CONFIG_FILE, "r") as f:
                     saved_config = json.load(f)
                     intended_model = saved_config.get("model", None)
                 logger.debug_with_context(f"Loaded model from file: {intended_model}") # Use global logger with context
             except Exception as e:
                 logger.error_with_context(f"Error loading config file: {str(e)}") # Use global logger with context
                 pass

        logger.info_with_context("Updating model select UI before fetch") # Use global logger with context
        model_select.disabled = True
        model_select.set_options([])
        model_select.prompt = "Fetching models..."
        model_select.clear()

        try:
            logger.info_with_context("Fetching available models") # Use global logger with context
            available_models = fetch_available_models(base_url, api_key)
            logger.debug_with_context(f"Fetched models: {available_models}") # Use global logger with context
            
            options = [(model, model) for model in available_models]
            model_select.set_options(options)
            if options:
                if intended_model and intended_model in available_models:
                    logger.info_with_context(f"Setting to intended model: {intended_model}") # Use global logger with context
                    model_select.value = intended_model
                else:
                    logger.info_with_context(f"Setting to first available model: {options[0][1]}") # Use global logger with context
                    model_select.value = options[0][1]
                model_select.prompt = "Select Model..."
            else:
                logger.warning_with_context("No models found") # Use global logger with context
                model_select.prompt = "No models found"
            model_select.disabled = False
            logger.info_with_context("Model fetch completed successfully") # Use global logger with context
        except ModelFetchError as e:
            self.app.notify(f"Error fetching models: {e}", severity="error", timeout=6)
            logger.error_with_context(f"ModelFetchError: {e}") # Log error with context
            model_select.prompt = "Error fetching models"
            model_select.disabled = True
        except Exception as e:
            self.app.notify(f"Unexpected error fetching models: {e}", severity="error", timeout=6)
            logger.error_with_context(f"Unexpected error fetching models: {e}", exc_info=True) # Log error with context
            model_select.prompt = "Error"
            model_select.disabled = True
    @on(Input.Changed)
    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle changes in LLM URL or OpenAI Key inputs."""
        self.show_invalid_reasons(event)

        openai_checkbox = self.query_one("#openai_api_checkbox", Checkbox)
        llm_url_input = self.query_one("#local_llm", Input)
        openai_key_input = self.query_one("#openai_key", Input)
        model_select = self.query_one("#model_select", Select)

        # Trigger fetch if OpenAI checkbox is checked AND API key is changed/provided
        if openai_checkbox.value and event.input.id == "openai_key":
            api_key = event.value.strip()
            if api_key:
                self.fetch_models_worker(llm_url_input.value, api_key)
            else:
                model_select.set_options([])
                model_select.clear()
                model_select.prompt = "Enter API Key..."
                model_select.disabled = True
        elif not openai_checkbox.value and event.input.id == "local_llm":
            url = event.value.strip()
            if url and event.validation_result and event.validation_result.is_valid:
                 self.fetch_models_worker(url)
            elif not url:
                 model_select.set_options([])
                 model_select.clear()
                 model_select.prompt = "Enter URL..."
                 model_select.disabled = True
            else:
                 model_select.set_options([])
                 model_select.clear()
                 model_select.prompt = "Invalid URL"
                 model_select.disabled = True

    @on(Checkbox.Changed, "#openai_api_checkbox")
    def update_llm_url_for_openai(self, event: Checkbox.Changed) -> None:
        """Update the LLM URL input and trigger model fetch based on the OpenAI API checkbox."""
        llm_url_input = self.query_one("#local_llm", Input)
        openai_key_input = self.query_one("#openai_key", Input)
        model_select = self.query_one("#model_select", Select)

        if event.value:
            # If checkbox is checked, set the URL to OpenAI's default and disable editing
            llm_url_input.value = "https://api.openai.com/v1"
            llm_url_input.disabled = True
            api_key = openai_key_input.value.strip()
            if api_key:
                self.fetch_models_worker(llm_url_input.value, api_key)
            else:
                model_select.set_options([])
                model_select.value = None
                model_select.prompt = "Enter API Key..."
                model_select.disabled = True
        else:
            # If checkbox is unchecked, enable editing, reset to default local URL, and fetch models
            llm_url_input.disabled = False
            if "local_llm" not in self.initial_config:
                 llm_url_input.value = "http://localhost:11434"
            else:
                 llm_url_input.value = str(self.initial_config.get("local_llm", "http://localhost:11434"))

            if llm_url_input.value and llm_url_input.is_valid:
                 self.fetch_models_worker(llm_url_input.value)
            else:
                 model_select.set_options([])
                 model_select.value = None
                 model_select.prompt = "Enter valid URL..." if llm_url_input.value else "Enter URL..."
                 model_select.disabled = True

            llm_url_input.focus()
    
        # _validate_all_inputs moved to be a class method
        # Call the validation function and get the results
        is_valid, summary_messages = self._validate_all_inputs() # Call as instance method

        summary_widget = self.query_one("#validation-summary", Pretty)
        if not is_valid:
            summary_widget.update("\n".join(summary_messages)) # Join messages for display
            summary_widget.display = True
            self.app.notify("Please fix the validation errors.", severity="error", title="Validation Failed")
        else:
            summary_widget.update([])
            summary_widget.display = False

        return is_valid

    def _validate_all_inputs(self) -> tuple[bool, list[str]]: # Corrected signature
        """Validate all visible and required inputs."""
        is_valid = True
        summary_messages = []
        for input_widget in self.query(Input):
            if input_widget.display and not input_widget.disabled:
                 if input_widget.validators:
                     validation_result = input_widget.validate(input_widget.value)
                     if validation_result is not None and not validation_result.is_valid:
                         is_valid = False
                         # Find label via DOM traversal instead of CSS selector
                         label_text = input_widget.id # Default to ID
                         try:
                             container = input_widget.parent
                             if container:
                                 label_widget = container.query(Label).first()
                                 if label_widget:
                                     label_text = str(label_widget.renderable) # Use renderable text
                         except Exception:
                             pass # Keep default ID if traversal fails
                         summary_messages.extend([f"{label_text}: {desc}" for desc in validation_result.failure_descriptions])
                         input_widget.add_class("-invalid")
                     else:
                         input_widget.remove_class("-invalid")
                 else:
                      input_widget.remove_class("-invalid")

        # Specific check for output_file if write_to_file is checked
        output_input = self.query_one("#output_file", Input) # Define output_input locally
        write_cb = self.query_one("#write_to_file_checkbox", Checkbox)
        if write_cb.value and not output_input.value.strip():
            is_valid = False
            summary_messages.append("Output File Path: Cannot be empty when 'Write to File' is checked.")
            output_input.add_class("-invalid")
        elif write_cb.value:
             output_input.remove_class("-invalid")


        auth_cb = self.query_one("#enable_auth", Checkbox)
        if auth_cb.value:
            auth_fields = ["reddit_username", "reddit_password", "reddit_client_id", "reddit_client_secret"]
            for field_id in auth_fields:
                auth_input = self.query_one(f"#{field_id}", Input)
                if not auth_input.value.strip():
                    is_valid = False
                    # Find label via DOM traversal instead of CSS selector
                    label_text = field_id # Default to ID
                    try:
                        container = auth_input.parent
                        if container:
                            label_widget = container.query(Label).first()
                            if label_widget:
                                label_text = str(label_widget.renderable) # Use renderable text
                    except Exception:
                        pass # Keep default ID if traversal fails
                    summary_messages.append(f"{label_text}: Cannot be empty when 'Enable Auth' is checked.")
                    auth_input.add_class("-invalid")
                else:
                    auth_input.remove_class("-invalid")
            
            return is_valid, summary_messages
        return is_valid, summary_messages # Ensure return happens even if auth_cb is false


    @on(Button.Pressed, "#submit-button")
    def handle_submit(self, event: Button.Pressed) -> None:
        """Collect all configuration values, validate, and exit returning the config."""
        is_valid, _ = self._validate_all_inputs() # Unpack tuple return
        if not is_valid:
             self.bell()
             return

        config_values = {}
        # Collect Checkbox values
        config_values["write_to_file"] = self.query_one("#write_to_file_checkbox", Checkbox).value
        config_values["enable_auth"] = self.query_one("#enable_auth", Checkbox).value
        config_values["pii_only"] = self.query_one("#pii_only", Checkbox).value
        config_values["use_openai_api"] = self.query_one("#openai_api_checkbox", Checkbox).value

        # Collect Input values
        config_values["output_file"] = self.query_one("#output_file", Input).value if config_values["write_to_file"] else None

        if config_values["enable_auth"]:
            config_values["reddit_username"] = self.query_one("#reddit_username", Input).value
            config_values["reddit_password"] = self.query_one("#reddit_password", Input).value
            config_values["reddit_client_id"] = self.query_one("#reddit_client_id", Input).value
            config_values["reddit_client_secret"] = self.query_one("#reddit_client_secret", Input).value
        else:
            config_values["reddit_username"] = None
            config_values["reddit_password"] = None
            config_values["reddit_client_id"] = None
            config_values["reddit_client_secret"] = None

        config_values["openai_key"] = self.query_one("#openai_key", Input).value
        config_values["local_llm"] = self.query_one("#local_llm", Input).value
        model_select = self.query_one("#model_select", Select)
        config_values["model"] = model_select.value if not model_select.disabled and model_select.value is not Select.BLANK else None

        # Convert limit and batch_size to int, handle empty strings
        limit_str = self.query_one("#limit", Input).value.strip()
        config_values["limit"] = int(limit_str) if limit_str else None # Use None if empty, let downstream handle default
        batch_size_str = self.query_one("#batch_size", Input).value.strip()
        config_values["batch_size"] = int(batch_size_str) if batch_size_str else None # Use None if empty

        config_values["text_match"] = self.query_one("#text_match", Input).value or None # Use None if empty
        config_values["skip_text"] = self.query_one("#skip_text", Input).value or None # Use None if empty

        # Collect Select values
        config_values["sort"] = self.query_one("#sort", Select).value
        config_values["time"] = self.query_one("#time", Select).value

        # Exit the app, returning the collected configuration
        self.exit(result=config_values)


    @on(Button.Pressed, "#save-button")
    def handle_save(self, event: Button.Pressed) -> None:
        """Collects current configuration and saves it to JSON file."""
        # No validation needed for save, just capture current state
        config_values = {}
        # Collect Checkbox values
        config_values["write_to_file"] = self.query_one("#write_to_file_checkbox", Checkbox).value
        config_values["enable_auth"] = self.query_one("#enable_auth", Checkbox).value
        config_values["pii_only"] = self.query_one("#pii_only", Checkbox).value
        config_values["use_openai_api"] = self.query_one("#openai_api_checkbox", Checkbox).value

        # Collect Input values
        config_values["output_file"] = self.query_one("#output_file", Input).value
        config_values["reddit_username"] = self.query_one("#reddit_username", Input).value
        config_values["reddit_password"] = self.query_one("#reddit_password", Input).value
        config_values["reddit_client_id"] = self.query_one("#reddit_client_id", Input).value
        config_values["reddit_client_secret"] = self.query_one("#reddit_client_secret", Input).value
        config_values["openai_key"] = self.query_one("#openai_key", Input).value
        config_values["local_llm"] = self.query_one("#local_llm", Input).value
        model_select = self.query_one("#model_select", Select)
        config_values["model"] = model_select.value if not model_select.disabled and model_select.value is not Select.BLANK else None
        config_values["limit"] = self.query_one("#limit", Input).value
        config_values["batch_size"] = self.query_one("#batch_size", Input).value
        config_values["text_match"] = self.query_one("#text_match", Input).value
        config_values["skip_text"] = self.query_one("#skip_text", Input).value

        # Collect Select values
        config_values["sort"] = self.query_one("#sort", Select).value
        config_values["time"] = self.query_one("#time", Select).value

        # --- Save to JSON file ---
        try:
            with open(self.CONFIG_FILE, "w") as f:
                json.dump(config_values, f, indent=4) # Write with indentation
            self.app.notify(f"Configuration saved successfully to '{self.CONFIG_FILE}'.", title="Save Success")
        except IOError as e:
            self.app.notify(f"Error saving configuration: {e}", severity="error", title="Save Error")
        except Exception as e: # Catch other potential errors
             self.app.notify(f"An unexpected error occurred during save: {e}", severity="error", title="Save Error")

    @on(Button.Pressed, "#quit-button")
    def handle_quit(self, event: Button.Pressed) -> None:
        """Quit the application without returning results."""
        self.exit(result=None)
