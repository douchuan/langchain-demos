"""
Single model invocation call

Demonstrates basic Q&A and tests model connectivity.

Demo: streaming chunk output.
Shows how to receive and print the model's response token by token as it is generated,
instead of waiting for the full response to complete.
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

# Use model.stream() to receive the response in chunks (tokens) as they are generated.
# This provides a real-time typing effect instead of waiting for the full response.
# - chunk.content: the text fragment in this chunk
# - end="": prevents print from adding a newline after each chunk
# - flush=True: forces immediate output to the terminal without buffering
for chunk in model.stream("Hello, what is Python?"):
    print(chunk.content, end="", flush=True)
