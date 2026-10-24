"""
Multimodal model invocation

Demonstrates how to call a model with both text and image input.
This example sends a text prompt along with a base64-encoded local image file
for the model to analyze. The model will describe the contents of the image.
"""

from base64 import b64encode
from pathlib import Path

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

# Resolve the base directory of this script file (e.g. .../src/step1)
BASE_DIR = Path(__file__).resolve().parent
# Build the path to the image file relative to the project (e.g. .../src/res/logo.jpeg)
IMAGE_PATH = BASE_DIR.parent / "res" / "logo.jpeg"
# Read the image file, encode it as base64, and decode to a UTF-8 string
base64_image = b64encode(IMAGE_PATH.read_bytes()).decode("utf-8")

# Build a multimodal message with both text and image content.
# The "content" field is a list of content blocks, each with a type.
# - "text": a plain text prompt for the model
# - "image": an image reference with base64-encoded data and MIME type
#   Using base64 allows sending local images without hosting them on a URL.
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the contents of this image"},
        {
            "type": "image",
            "base64": base64_image,  # Base64-encoded image data
            "mime_type": "image/jpeg",  # MIME type of the image
        },
    ],
}

# Invoke the model with the multimodal message and print the response
response = model.invoke([message])
print(response.content)
