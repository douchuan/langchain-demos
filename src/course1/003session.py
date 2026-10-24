"""
Agent execution loop: the model autonomously decides to call tools for weather queries.

Demonstrates custom context passing, structured responses, multi-tool chaining,
and conversational memory using a checkpointer.

This is an advanced agent example that:
- Uses multiple tools (get_weather + locate_user) that the model can chain together
- Passes user context to tools via a typed context schema
- Returns structured data (pydantic/dataclass-based response format)
- Maintains conversation memory across invocations using a checkpointer
"""

from dataclasses import dataclass

import requests
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver

from utils.config import load_settings


# Dataclass to hold user context information.
# This is passed to tools that need access to user-specific data.
@dataclass
class Context:
    user_id: str


# Dataclass defining the structured response format the agent should return.
# The agent will populate these fields from the weather data.
@dataclass
class ResponseFormat:
    summary: str
    temperature_celsius: float
    temperature_fahrenheit: float
    humidity: float


# Load environment variables from .env file and application settings
settings = load_settings()


# Tool: fetch weather information for a given city.
# The model can call this tool when the user asks about weather in a specific city.
@tool(
    "get_weather",
    return_direct=False,
)
def get_weather(city: str) -> dict:
    """Return weather information for a given city. Input should be an English city name like Beijing."""
    try:
        # Call the wttr.in weather API with a timeout to avoid hanging requests
        resp = requests.get(f"https://wttr.in/{city}?format=j1", timeout=10)
        resp.raise_for_status()  # Raise an exception for HTTP error responses
        return resp.json()
    except requests.RequestException as e:
        return {"error": f"Failed to fetch weather for {city}: {e}"}
    except ValueError as e:
        return {"error": f"Invalid JSON response for {city}: {e}"}


# Tool: look up a user's city based on their user ID from the context.
# This tool receives a runtime object that provides access to the agent's context.
@tool("locate_user")
def locate_user(runtime: ToolRuntime[Context]):
    """Look up a user's city based on the context."""
    # If no context is provided, return Unknown
    if runtime.context is None:
        return "Unknown"
    # Map user IDs to their corresponding cities using pattern matching
    match runtime.context.user_id:
        case "ABC123":
            return "Vienna"
        case "XYZ456":
            return "London"
        case "HJKL111":
            return "Paris"
        case _:
            return "Unknown"


# Initialize the chat model with OpenAI-compatible API configuration
chat_model = init_chat_model(
    model=settings.model,
    model_provider="openai",
    base_url=settings.base_url,
    api_key=settings.api_key,
    temperature=0.1,
)

# Create an in-memory checkpointer for conversation persistence.
# This allows the agent to remember previous interactions across invocations
# when using the same thread_id.
checkpointer = InMemorySaver()

# Create an agent with multiple tools, structured response format, and memory.
# The agent can:
# - Call locate_user to find the user's city when no city is specified
# - Call get_weather to fetch weather data for that city
# - Return a structured response with temperature and humidity fields
agent = create_agent(
    model=chat_model,
    tools=[get_weather, locate_user],
    system_prompt=(
        "You are a helpful weather assistant, who always cracks jokes and is "
        "humorous while remaining helpful. "
        # "When the user asks about weather without specifying a city, first call "
        # "`locate_user` to find their city, then call `get_weather` with that city. "
        # "When building the structured response, read `temp_C`, `temp_F`, and "
        # "`humidity` directly from the weather data; do not recalculate them."
    ),
    context_schema=Context,  # Schema for the context passed to tools
    response_format=ResponseFormat,  # Schema for the structured response
    checkpointer=checkpointer,  # Enable conversation memory
)

# Configuration for the checkpointer. thread_id identifies a conversation thread.
# Messages with the same thread_id share the same conversation history.
config = {"configurable": {"thread_id": "1"}}

# Invoke the agent without specifying a city.
# The agent should call locate_user first (to find the user's city), then get_weather.
# Try different user_id values (e.g. "XYZ456") to see different cities returned.
response = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather like?"}]},
    config=config,
    context=Context(user_id="ABC123"),
)

# Print the structured response object and its individual fields
print(response["structured_response"])
print(response["structured_response"].summary)
print(response["structured_response"].temperature_celsius)


# Demonstrate conversation memory: follow-up question with the same thread_id.
# The agent should remember the previous weather information from the conversation history.
response = agent.invoke(
    {"messages": [{"role": "user", "content": "And is this usual"}]},
    config=config,
    context=Context(user_id="ABC123"),
)
print(response["structured_response"])


# Demonstrate isolated memory: use a different thread_id.
# The agent will NOT remember the previous conversation, so it needs to re-fetch weather.
config = {"configurable": {"thread_id": "2"}}
response = agent.invoke(
    {"messages": [{"role": "user", "content": "And is this usual"}]},
    config=config,
    context=Context(user_id="ABC123"),
)
print(response["structured_response"])
