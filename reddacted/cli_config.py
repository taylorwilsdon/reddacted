import json
import os
import os.path
from typing import Optional, Dict, Any, TYPE_CHECKING, Tuple, List

from textual.widgets import Input, Label, Checkbox # Needed for validate_inputs

# Forward reference for type hinting ConfigApp to avoid circular import
if TYPE_CHECKING:
    from reddacted.textual_cli import ConfigApp

# --- Constants ---
VALID_SORT_OPTIONS = ["hot", "new", "controversial", "top"]
VALID_TIME_OPTIONS = ["all", "day", "hour", "month", "week", "year"]
URL_REGEX = r"^(http|https)://[^\s/$.?#].[^\s]*$"
CONFIG_FILE = "config.json"

# Environment Variable Keys (Example - adjust if needed)
ENV_VARS_MAP = {
    "REDDIT_USERNAME": "reddit_username",
    "REDDIT_PASSWORD": "reddit_password",
    "REDDIT_CLIENT_ID": "reddit_client_id",
    "REDDIT_CLIENT_SECRET": "reddit_client_secret",
    "OPENAI_API_KEY": "openai_key",
}

# --- Configuration File I/O ---

def load_config_from_file(filepath: str) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Loads configuration from a JSON file.

    Args:
        filepath: The path to the configuration file.

    Returns:
        A tuple containing:
            - A dictionary with the loaded configuration values (empty if file not found or error).
            - An optional notification message (string) for success or error.
    """
    config_values = {}
    notification = None
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                config_values = json.load(f)
            notification = f"Configuration loaded from '{filepath}'."
        except json.JSONDecodeError:
            notification = f"Error decoding '{filepath}'. Using defaults."
            config_values = {}
        except Exception as e:
            notification = f"Error loading config file '{filepath}': {e}"
            config_values = {}
    else:
        notification = f"No configuration file found at '{filepath}'. Using defaults."
    return config_values, notification

def save_config_to_file(filepath: str, config_data: Dict[str, Any]) -> Optional[str]:
    """
    Saves the configuration dictionary to a JSON file.

    Args:
        filepath: The path to the configuration file.
        config_data: The dictionary containing configuration values to save.

    Returns:
        An optional notification message (string) for success or error.
    """
    notification = None
    try:
        with open(filepath, "w") as f:
            json.dump(config_data, f, indent=4) # Write with indentation
        notification = f"Configuration saved successfully to '{filepath}'."
    except IOError as e:
        notification = f"Error saving configuration to '{filepath}': {e}"
    except Exception as e: # Catch other potential errors
        notification = f"An unexpected error occurred during save to '{filepath}': {e}"
    return notification

# --- Configuration Merging & Processing ---

def merge_configs(file_config: Dict[str, Any], initial_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merges configuration from file and initial values (CLI/env).
    Initial values take precedence. Handles boolean type conversions.

    Args:
        file_config: Configuration loaded from the file.
        initial_config: Configuration provided via CLI arguments or environment variables.

    Returns:
        The final merged configuration dictionary.
    """
    merged_config = file_config.copy() # Start with file config

    # Process and merge initial_config, giving it precedence
    processed_initial_config = {}
    boolean_keys = ["enable_auth", "pii_only", "use_openai_api", "write_to_file"] # Keys expected to be boolean
    for key, value in initial_config.items():
        if isinstance(value, str):
            if value.lower() in ('true', '1', 'yes'):
                processed_initial_config[key] = True
            elif value.lower() in ('false', '0', 'no'):
                processed_initial_config[key] = False
            else:
                processed_initial_config[key] = value # Keep as string if not boolean-like
        elif isinstance(value, int) and key in boolean_keys:
            processed_initial_config[key] = bool(value) # Convert int to bool for specific keys
        else:
            processed_initial_config[key] = value # Keep other types as is

    merged_config.update(processed_initial_config) # Update with processed initial values
    return merged_config


# --- Input Validation ---

def validate_inputs(app: 'ConfigApp') -> Tuple[bool, List[str]]:
    """
    Validate all visible and required inputs in the ConfigApp.

    Args:
        app: The instance of the ConfigApp.

    Returns:
        A tuple containing:
            - A boolean indicating if all validations passed.
            - A list of validation failure messages.
    """
    is_valid = True
    summary_messages = []

    # Validate standard Input widgets with validators
    for input_widget in app.query(Input):
        if input_widget.display and not input_widget.disabled:
            # Special case: Skip validation for batch_size if it's empty
            if input_widget.id == "batch_size" and not input_widget.value.strip():
                input_widget.remove_class("-invalid") # Ensure it's not marked invalid if empty
                input_widget.add_class("-valid")
                continue # Skip the rest of the validation for this input

            # Clear previous invalid state
            input_widget.remove_class("-invalid")
            input_widget.add_class("-valid") # Assume valid initially

            if input_widget.validators:
                validation_result = input_widget.validate(input_widget.value)
                if validation_result is not None and not validation_result.is_valid:
                    is_valid = False
                    # Find label via DOM traversal
                    label_text = input_widget.id # Default to ID
                    try:
                        container = input_widget.parent
                        if container:
                            label_widget = container.query(Label).first()
                            if label_widget:
                                label_text = str(label_widget.renderable).strip().rstrip(':') # Use renderable text, clean up
                    except Exception:
                        pass # Keep default ID if traversal fails
                    summary_messages.extend([f"{label_text}: {desc}" for desc in validation_result.failure_descriptions])
                    input_widget.remove_class("-valid")
                    input_widget.add_class("-invalid")

    # Specific check for output_file if write_to_file is checked
    write_cb = app.query_one("#write_to_file_checkbox", Checkbox)
    output_input = app.query_one("#output_file", Input)
    if write_cb.value and not output_input.value.strip():
        is_valid = False
        summary_messages.append("Output File Path: Cannot be empty when 'Write to File' is checked.")
        output_input.remove_class("-valid")
        output_input.add_class("-invalid")
    elif write_cb.value: # If checked and not empty, ensure it's marked valid (if not already invalid by validator)
        if "-invalid" not in output_input.classes:
             output_input.remove_class("-invalid")
             output_input.add_class("-valid")


    # Specific check for Reddit auth fields if enable_auth is checked
    auth_cb = app.query_one("#enable_auth", Checkbox)
    if auth_cb.value:
        auth_fields = ["reddit_username", "reddit_password", "reddit_client_id", "reddit_client_secret"]
        for field_id in auth_fields:
            auth_input = app.query_one(f"#{field_id}", Input)
            if not auth_input.value.strip():
                is_valid = False
                # Find label via DOM traversal
                label_text = field_id # Default to ID
                try:
                    container = auth_input.parent
                    if container:
                        label_widget = container.query(Label).first()
                        if label_widget:
                            label_text = str(label_widget.renderable).strip().rstrip(':') # Use renderable text, clean up
                except Exception:
                    pass # Keep default ID if traversal fails
                summary_messages.append(f"{label_text}: Cannot be empty when 'Enable Auth' is checked.")
                auth_input.remove_class("-valid")
                auth_input.add_class("-invalid")
            else: # If not empty, ensure it's marked valid (if not already invalid by validator)
                 if "-invalid" not in auth_input.classes:
                    auth_input.remove_class("-invalid")
                    auth_input.add_class("-valid")

    return is_valid, summary_messages