"""
Single model invocation call

Demonstrates basic Q&A and tests model connectivity.
This is the simplest example: send a single prompt and print the response.
"""

from langchain.chat_models import init_chat_model

from utils.config import load_settings

# Load environment variables from .env file and application settings
settings = load_settings()

# Initialize the chat model with OpenAI-compatible API configuration.
# - model: the specific model name provided by the service (e.g. "gpt-4o")
# - model_provider="openai": tells LangChain to use the OpenAI-compatible protocol
# - base_url: the API endpoint URL (can point to a third-party proxy)
# - api_key: authentication key for the API
# - temperature=0.1: low randomness for more deterministic outputs
model = init_chat_model(
    model=settings.model,  # Fill in the actual model name provided by the service
    model_provider="openai",  # Tell LangChain to use the OpenAI-compatible protocol
    base_url=settings.base_url,
    api_key=settings.api_key,
    temperature=0.1,
)

# Invoke the model with a simple question and print the results
response = model.invoke("Hello, what is Python?")
# Print the full response object (includes role, content, and metadata)
print(response)
# Print only the text content of the response
print(response.content)
