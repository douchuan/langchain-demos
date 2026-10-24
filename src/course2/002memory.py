import uuid

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from utils.config import load_settings

# Load environment variables from .env file and application settings
settings = load_settings()

model = init_chat_model(
    model=settings.model,
    model_provider="openai",
    base_url=settings.base_url,
    api_key=settings.api_key,
)


def prompt_llm(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}


graph_builder = StateGraph(MessagesState)

graph_builder.add_node(prompt_llm)
graph_builder.add_edge(START, "prompt_llm")
graph_builder.add_edge("prompt_llm", END)

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": uuid.uuid4()}}


response = graph.invoke(
    {"messages": [{"role": "user", "content": "Hello, My name is Big Boss"}]},
    config=config,
)
print(response)

response = graph.invoke(
    {"messages": [{"role": "user", "content": "Do you know who I am?"}]}, config=config
)
print(response)
