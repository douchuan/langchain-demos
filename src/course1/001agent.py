"""
Agent execution loop

Demonstrates how a model can autonomously decide to call tools to complete a task.
This example creates a weather assistant agent that can call a weather-fetching tool.
"""

import requests
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

from utils.config import load_settings

# Load environment variables from .env file and application settings
settings = load_settings()


# Define a tool that the agent can call to fetch weather data.
# The @tool decorator registers this function as a callable tool for the agent.
# - "get_weather": the tool name exposed to the model
# - description: tells the model what this tool does and when to use it
# - return_direct=False: the result goes back to the model for further reasoning, not directly to the user
@tool(
    "get_weather",
    description="Return weather information for a given city",
    return_direct=False,
)
def get_weather(city: str) -> str:
    """Fetch current weather for a specified city. Input should be the city name in English, e.g. 'Beijing'."""
    # Call the wttr.in weather API and return the parsed JSON response
    resp = requests.get(f"https://wttr.in/{city}?format=j1")
    return resp.json()


# Initialize the chat model using OpenAI-compatible API settings
client = ChatOpenAI(
    api_key=settings.api_key,
    model=settings.model,
    base_url=settings.base_url,
)

# Create an agent with the model, tools, and a system prompt.
# The agent will decide when and how to call the provided tools based on the user's query.
agent = create_agent(
    model=client,
    tools=[get_weather],
    system_prompt="You are a helpful weather assistant, who always cracks jokes and is humorous while remaining helpful",
)

# Invoke the agent with a user message asking about weather in Beijing.
# The agent may call get_weather internally before producing a final response.
response = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in Beijing?"}]}
)

# Print the full response object (includes all message history and tool calls)
print(response)
# Print only the final assistant message content
print(response["messages"][-1].content)
