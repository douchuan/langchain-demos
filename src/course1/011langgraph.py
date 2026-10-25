"""
LangGraph Basics

Demonstrates how to build a stateful multi-step agent using LangGraph,
the official framework for complex Agent workflows in the LangChain ecosystem.
Covers StateGraph, nodes, edges, and conditional routing.
"""

import operator
from collections.abc import Sequence
from typing import Annotated, Literal

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from utils.config import load_settings

# Load environment variables from .env file and application settings
settings = load_settings()


# --- Example 1: Simple LangGraph with tools ---
# Define the state schema for the graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


# Define a tool the agent can use
@tool
def get_current_temperature(city: str) -> str:
    """Return the current temperature for a given city."""
    temperatures = {
        "Beijing": "22°C",
        "Shanghai": "28°C",
        "Guangzhou": "32°C",
        "London": "15°C",
    }
    return temperatures.get(city, f"Unknown temperature for {city}")


@tool
def get_weather_description(temperature: str) -> str:
    """Given a temperature, return a weather description."""
    try:
        temp = int(temperature.replace("°C", ""))
    except ValueError:
        return "unknown weather"
    if temp > 30:
        return "It's a hot day. Stay hydrated!"
    elif temp > 20:
        return "Pleasant weather. Good day for a walk!"
    elif temp > 10:
        return "A bit chilly. Bring a jacket."
    else:
        return "It's cold outside. Bundle up!"


# Initialize the model with tool binding
model = init_chat_model(
    model=settings.model,
    model_provider="openai",
    base_url=settings.base_url,
    api_key=settings.api_key,
    temperature=0.1,
)
model_with_tools = model.bind_tools([get_current_temperature, get_weather_description])


# Define the agent node: call the model with the current state
def call_model(state: AgentState) -> dict:
    """Invoke the model with the message history and return its response."""
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# Define the conditional edge: should we call tools or end?
def should_use_tools(state: AgentState) -> Literal["tools", END]:
    """Check if the model's last message contains tool calls."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


# Build the graph
workflow = StateGraph(AgentState)

# Add nodes (each node is a function that takes state and returns state updates)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode([get_current_temperature, get_weather_description]))

# Set the entry point
workflow.set_entry_point("agent")

# Add conditional edge: agent -> tools (if tool calls exist) or -> END
workflow.add_conditional_edges("agent", should_use_tools, {"tools": "tools", END: END})

# After tools execute, go back to the agent for a final response
workflow.add_edge("tools", "agent")

# Compile the graph into a runnable object
app = workflow.compile()

# Visualize the graph (optional, requires graphviz)
# print(app.get_graph().draw_mermaid())

print("=== Example 1: LangGraph with Tools ===")
result = app.invoke(
    {"messages": [HumanMessage("What's the weather like in Beijing? Describe it.")]}
)
print(f"Final response: {result['messages'][-1].content}")


# --- Example 2: Multi-node workflow (non-agent) ---
# A workflow where each node does a different transformation step
class ResearchState(TypedDict):
    topic: str
    summary: str
    key_points: str
    final_report: str


def gather_info(state: ResearchState) -> dict:
    """Step 1: Generate a summary of the topic."""
    model = init_chat_model(
        model=settings.model,
        model_provider="openai",
        base_url=settings.base_url,
        api_key=settings.api_key,
        temperature=0.1,
    )
    response = model.invoke(f"Provide a brief summary of: {state['topic']}")
    return {"summary": response.content}


def extract_key_points(state: ResearchState) -> dict:
    """Step 2: Extract key points from the summary."""
    model = init_chat_model(
        model=settings.model,
        model_provider="openai",
        base_url=settings.base_url,
        api_key=settings.api_key,
        temperature=0.1,
    )
    response = model.invoke(
        f"Given this summary, extract 3 key points:\n{state['summary']}"
    )
    return {"key_points": response.content}


def write_report(state: ResearchState) -> dict:
    """Step 3: Write a final report combining summary and key points."""
    model = init_chat_model(
        model=settings.model,
        model_provider="openai",
        base_url=settings.base_url,
        api_key=settings.api_key,
        temperature=0.1,
    )
    response = model.invoke(
        f"Write a concise report.\nSummary: {state['summary']}\nKey Points:\n{state['key_points']}"
    )
    return {"final_report": response.content}


# Build the research workflow
research_graph = StateGraph(ResearchState)
research_graph.add_node("gather", gather_info)
research_graph.add_node("extract", extract_key_points)
research_graph.add_node("report", write_report)

# Linear flow: gather -> extract -> report -> END
research_graph.set_entry_point("gather")
research_graph.add_edge("gather", "extract")
research_graph.add_edge("extract", "report")
research_graph.add_edge("report", END)

research_app = research_graph.compile()

print("\n=== Example 2: Multi-Node Workflow ===")
result = research_app.invoke({"topic": "quantum computing"})
print(f"Final Report:\n{result['final_report']}")


# --- Example 3: Conditional branching ---
# Route to different nodes based on the input type
class RouterState(TypedDict):
    question: str
    category: str
    answer: str


def classify_question(state: RouterState) -> dict:
    """Classify the question into a category."""
    model = init_chat_model(
        model=settings.model,
        model_provider="openai",
        base_url=settings.base_url,
        api_key=settings.api_key,
        temperature=0.1,
    )
    response = model.invoke(
        f"Classify this question into one word: 'code' or 'math'. Question: {state['question']}. Respond with ONLY the word."
    )
    category = response.content.strip().lower()
    return {"category": category}


def answer_code(state: RouterState) -> dict:
    """Answer a coding-related question."""
    model = init_chat_model(
        model=settings.model,
        model_provider="openai",
        base_url=settings.base_url,
        api_key=settings.api_key,
        temperature=0.1,
    )
    response = model.invoke(f"As a coding expert, answer: {state['question']}")
    return {"answer": response.content}


def answer_math(state: RouterState) -> dict:
    """Answer a math-related question."""
    model = init_chat_model(
        model=settings.model,
        model_provider="openai",
        base_url=settings.base_url,
        api_key=settings.api_key,
        temperature=0.1,
    )
    response = model.invoke(f"As a math tutor, answer: {state['question']}")
    return {"answer": response.content}


def route_question(state: RouterState) -> Literal["code_node", "math_node"]:
    """Route based on the classified category."""
    if state["category"] == "code":
        return "code_node"
    return "math_node"


# Build the routing workflow
router_graph = StateGraph(RouterState)
router_graph.add_node("classify", classify_question)
router_graph.add_node("code_node", answer_code)
router_graph.add_node("math_node", answer_math)

router_graph.set_entry_point("classify")
router_graph.add_conditional_edges(
    "classify",
    route_question,
    {"code_node": "code_node", "math_node": "math_node"},
)
router_graph.add_edge("code_node", END)
router_graph.add_edge("math_node", END)

router_app = router_graph.compile()

print("\n=== Example 3: Conditional Routing ===")
result = router_app.invoke({"question": "What is a decorator in Python?"})
print(f"Category: {result['category']}")
print(f"Answer: {result['answer'][:100]}...")
