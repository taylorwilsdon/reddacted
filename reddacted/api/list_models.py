import requests
from typing import List, Optional

class ModelFetchError(Exception):
    """Custom exception for errors during model fetching."""
    pass

def fetch_available_models(base_url: str, api_key: Optional[str] = None) -> List[str]:
    """
    Fetches available LLM models from a given base URL (OpenAI compatible API).

    Args:
        base_url: The base URL of the LLM API (e.g., http://localhost:11434 or https://api.openai.com).
        api_key: Optional API key required for authentication (e.g., OpenAI).

    Returns:
        A list of available model IDs.

    Raises:
        ModelFetchError: If there's an error fetching or parsing the models.
        requests.exceptions.RequestException: If there's a connection error.
    """
    models_url = f"{base_url.rstrip('/')}/v1/models" # Ensure correct endpoint path
    headers = {"User-Agent": "Reddacted-Config-UI"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = requests.get(models_url, headers=headers, timeout=10) # Added timeout
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

        models_data = response.json()
        available_models = [m.get("id") for m in models_data.get("data", []) if m.get("id")]

        if not available_models:
            raise ModelFetchError("No available models found in the API response.")

        return available_models

    except requests.exceptions.HTTPError as e:
        raise ModelFetchError(f"HTTP Error fetching models: {e.response.status_code} - {e.response.text}") from e
    except requests.exceptions.ConnectionError as e:
        raise ModelFetchError(f"Connection error fetching models from {models_url}: {e}") from e
    except requests.exceptions.Timeout as e:
        raise ModelFetchError(f"Timeout fetching models from {models_url}: {e}") from e
    except requests.exceptions.RequestException as e:
        raise ModelFetchError(f"Error fetching models from {models_url}: {e}") from e
    except (ValueError, KeyError) as e: # Handle potential JSON parsing or key errors
         raise ModelFetchError(f"Error parsing model response from {models_url}: {e}") from e