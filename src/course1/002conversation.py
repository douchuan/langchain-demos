"""
Single model invocation call

Demonstrates basic Q&A and tests model connectivity.

Demo: multi-turn conversation with message history.
Shows how to pass a list of messages (system, human, AI) to simulate a conversation.
"""

from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage, SystemMessage

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

# Build a conversation history with multiple message types:
# - SystemMessage: sets the assistant's behavior/persona
# - HumanMessage: a message from the user
# - AIMessage: a previous response from the assistant (simulated history)
# The last message is a new user question, which the model will answer.
conversation = [
    SystemMessage("Your are a helpful assistant for questions regarding programming"),
    HumanMessage("Hello, what is Python?"),
    AIMessage("Python is a high-level, interpreted programming language."),
    HumanMessage("When was it released?"),
]

# Invoke the model with the full conversation history.
# The model uses the context from previous messages to answer the latest question.
response = model.invoke(conversation)
# Print the full response object (includes role, content, and metadata)
print(response)
# Print only the text content of the response
print(response.content)
