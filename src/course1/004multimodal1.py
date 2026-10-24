"""
Multimodal model invocation

Demonstrates how to call a model with both text and image input.
This example sends a text prompt along with an image URL for the model to analyze.
The model will describe the contents of the image referenced by the URL.
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

# Build a multimodal message with both text and image content.
# The "content" field is a list of content blocks, each with a type.
# - "text": a plain text prompt for the model
# - "image": an image reference (can be a URL or base64-encoded data)
#   When using a URL, the model will fetch the image from the given link.
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the contents of this image"},
        {
            "type": "image",
            "url": "https://gips0.baidu.com/it/u=618883841,3286727987&fm=3074&app=3074&f=JPEG?w=1080&h=1060&type=normal&func=",
        },
    ],
}

# Invoke the model with the multimodal message and print the response
response = model.invoke([message])
print(response.content)
