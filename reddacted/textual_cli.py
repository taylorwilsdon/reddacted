import os
import time
from typing import Optional, Dict, Any, TYPE_CHECKING
from textual import on
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll, Horizontal, Container
from textual.validation import Number, Regex
from textual.widgets import Input, Label, Pretty, Checkbox, Select, Button
from textual import work
from reddacted.utils.logging import get_logger
from reddacted.styles import TEXTUAL_CSS
from reddacted.api.list_models import fetch_available_models, ModelFetchError
import reddacted.cli_config as cli_config
from reddacted.cli_config import URL_REGEX, VALID_SORT_OPTIONS, VALID_TIME_OPTIONS

if TYPE_CHECKING:
    from reddacted.cli_config import ConfigApp
from reddacted.api.list_models import fetch_available_models, ModelFetchError

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
        logger.debug_with_context(f"Initial config: {initial_config}")
        
        super().__init__(*args, **kwargs)
        self.initial_config = initial_config or {}
        self.model_fetch_worker = None # Initialize worker reference
        logger.debug_with_context(f"Processed config: {self.initial_config}")


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
                    validators=[Regex(cli_config.URL_REGEX, failure_description="Must be a valid URL (http/https)")],
                    id="local_llm",
                )
            with Horizontal(id="limit-batch-row"):
                with Container(classes="number-input-group"):
                    yield Label("Comment Limit (0 for unlimited):")
                    yield Input(
                        placeholder="Default: 20", # Updated placeholder
                        validators=[Number(minimum=0)],
                        id="limit",
                        type="integer",
                        value="20", # Set initial value
                    )
                with Container(classes="number-input-group"):
                    yield Label("Batch Size (for delete/update):")
                    yield Input(
                        placeholder="Default: 10",
                        validators=[Number(minimum=1)], # Removed allow_blank
                        id="batch_size",
                        type="integer",
                    )

            # Sort/Time Select Row
            with Horizontal(id="select-row"):
                 with Container(classes="select-group"):
                    yield Label("Sort Order:")
                    yield Select(
                        [(option.capitalize(), option) for option in cli_config.VALID_SORT_OPTIONS],
                        prompt="Select Sort...",
                        value="new",
                        id="sort",
                        allow_blank=False,
                    )
                 with Container(classes="select-group"):
                    yield Label("Time Filter:")
                    yield Select(
                        [(option.capitalize(), option) for option in cli_config.VALID_TIME_OPTIONS],
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
                yield Button("Save Config", id="save-button", variant="success")
                yield Button("Submit", id="submit-button", variant="primary")
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

    # CONFIG_FILE constant moved to cli_config.py

    def on_mount(self) -> None:
        """Called when the app is mounted. Loads config and fetches initial models."""
        logger.info_with_context("App mounting - starting configuration load")
        
        try:
            self.load_configuration()
            logger.info_with_context("Configuration loaded successfully")
            
            llm_url_input = self.query_one("#local_llm", Input)
            openai_key_input = self.query_one("#openai_key", Input)
            openai_checkbox = self.query_one("#openai_api_checkbox", Checkbox)
            
            logger.debug_with_context(f"LLM URL: {llm_url_input.value}, OpenAI Checkbox: {openai_checkbox.value}")
            
            if openai_checkbox.value and openai_key_input.value:
                time.sleep(1)
                logger.info_with_context("Fetching OpenAI models")
                self.model_fetch_worker = self.fetch_models_worker(llm_url_input.value, openai_key_input.value)
            elif not openai_checkbox.value and llm_url_input.value and llm_url_input.is_valid:
                logger.info_with_context("Fetching local LLM models")
                self.model_fetch_worker = self.fetch_models_worker(llm_url_input.value)
            else:
                logger.info_with_context("Skipping model fetch - conditions not met")
        except Exception as e:
            logger.error_with_context(f"Error during app mount: {str(e)}", exc_info=True)
            raise

    def load_configuration(self) -> None:
        """Loads configuration using cli_config and populates UI."""
        # Load from file using cli_config
        file_config, load_notification = cli_config.load_config_from_file(cli_config.CONFIG_FILE)
        if load_notification:
            severity = "error" if "Error" in load_notification else "information"
            title = "Config Load Error" if "Error" in load_notification else "Config Load"
            # Revert timeout change
            self.app.notify(load_notification, severity=severity, title=title)

        # Merge file config with initial config (CLI/env) using cli_config
        # self.initial_config is set in __init__
        config_values = cli_config.merge_configs(file_config, self.initial_config)

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
                 llm_input.value = str(config_values.get("local_llm", "https://api.openai.com"))
                 llm_input.disabled = True
            else:
                 llm_input.value = str(config_values.get("local_llm", "http://localhost:11434"))
                 llm_input.disabled = False


            # Set default value for limit to 20 if not present in config
            self.query_one("#limit", Input).value = str(config_values.get("limit", "20"))
            # Batch size can be empty, so load "" if not present
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
        logger.info_with_context(f"Starting model fetch from {base_url}")
        
        model_select = self.query_one("#model_select", Select)
        intended_model = self.initial_config.get("model", None)
        logger.debug_with_context(f"Intended model from config: {intended_model}")
        
        if intended_model is None and os.path.exists(cli_config.CONFIG_FILE):
             try:
                 logger.debug_with_context(f"Loading model from config file: {cli_config.CONFIG_FILE}")
                 with open(cli_config.CONFIG_FILE, "r") as f:
                     saved_config = json.load(f)
                     intended_model = saved_config.get("model", None)
                 logger.debug_with_context(f"Loaded model from file: {intended_model}")
             except Exception as e:
                 logger.error_with_context(f"Error loading config file: {str(e)}")
                 pass

        logger.info_with_context("Updating model select UI before fetch")
        model_select.disabled = True
        model_select.set_options([])
        model_select.prompt = "Fetching models..."
        model_select.clear()

        try:
            logger.info_with_context("Fetching available models")
            available_models = fetch_available_models(base_url, api_key)
            logger.debug_with_context(f"Fetched models: {available_models}")
            
            options = [(model, model) for model in available_models]
            model_select.set_options(options)
            if options:
                if intended_model and intended_model in available_models:
                    logger.info_with_context(f"Setting to intended model: {intended_model}")
                    model_select.value = intended_model
                else:
                    logger.info_with_context(f"Setting to first available model: {options[0][1]}")
                    model_select.value = options[0][1]
                model_select.prompt = "Select Model..."
            else:
                logger.warning_with_context("No models found")
                model_select.prompt = "No models found"
            model_select.disabled = False
            logger.info_with_context("Model fetch completed successfully")
        except ModelFetchError as e:
            self.app.notify(f"Error fetching models: {e}", severity="error", timeout=6)
            logger.error_with_context(f"ModelFetchError: {e}")
            model_select.prompt = "Error fetching models"
            model_select.disabled = True
        except Exception as e:
            self.app.notify(f"Unexpected error fetching models: {e}", severity="error", timeout=6)
            logger.error_with_context(f"Unexpected error fetching models: {e}", exc_info=True)
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
            llm_url_input.value = "https://api.openai.com"
            llm_url_input.disabled = True
            api_key = openai_key_input.value.strip()
            if api_key:
                self.fetch_models_worker(llm_url_input.value, api_key)
            else:
                model_select.set_options([])
                model_select.clear()
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

    def _validate_all_inputs(self) -> tuple[bool, list[str]]:
        """Validate all inputs using the utility function."""
        # Delegate validation to the cli_config module
        return cli_config.validate_inputs(self)


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

        # Cancel the worker if it's running before exiting
        if self.model_fetch_worker is not None and self.model_fetch_worker.is_running:
            logger.info_with_context("Cancelling active model fetch worker before exit.")
            try:
                self.model_fetch_worker.cancel()
            except Exception as e:
                logger.error_with_context(f"Error cancelling worker: {e}")

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

        # --- Save to JSON file using cli_config ---
        save_notification = cli_config.save_config_to_file(cli_config.CONFIG_FILE, config_values)
        if save_notification:
            severity = "error" if "Error" in save_notification else "success"
            title = "Save Error" if "Error" in save_notification else "Save Success"
            self.app.notify(save_notification, severity=severity, title=title)

    @on(Button.Pressed, "#quit-button")
    def handle_quit(self, event: Button.Pressed) -> None:
        """Quit the application without returning results."""
        self.exit(result=None)
