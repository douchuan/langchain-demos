"""
Configuration module for loading and validating environment settings.

Provides a Settings dataclass with the API key, model name, and base URL
required for initializing chat models.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


# Immutable settings container for API configuration.
# frozen=True makes instances hashable and prevents accidental modification.
@dataclass(frozen=True)
class Settings:
    api_key: str
    model: str
    model2: str
    base_url: str


def load_settings() -> Settings:
    """
    Load environment variables from a .env file and validate that all required
    variables are present.

    Returns:
        Settings: An immutable settings object with api_key, model, and base_url.

    Raises:
        ValueError: If any required environment variable is not configured.
    """
    # Load environment variables from the .env file in the project root
    load_dotenv()

    # Read required environment variables (None if not set)
    api_key = os.getenv("API_KEY")
    model = os.getenv("LLM_MODEL")
    model2 = os.getenv("LLM_MODEL2")
    base_url = os.getenv("BASE_URL")

    # Check which required variables are missing
    missing = [
        name
        for name, value in [
            ("API_KEY", api_key),
            ("LLM_MODEL", model),
            ("LLM_MODEL2", model2),
            ("BASE_URL", base_url),
        ]
        if not value
    ]
    # Raise an error listing any missing variables
    if missing:
        raise ValueError(
            f"Please configure the following in .env: {', '.join(missing)}"
        )

    # Return an immutable Settings object with all validated values
    return Settings(
        api_key=api_key,
        model=model,
        model2=model2,
        base_url=base_url,
    )
