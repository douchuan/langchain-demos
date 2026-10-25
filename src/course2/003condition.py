import uuid
from typing import Annotated, Literal, TypedDict

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from utils.config import load_settings

settings = load_settings()

llm = init_chat_model(
    model=settings.model,
    model_provider="openai",
    base_url=settings.base_url,
    api_key=settings.api_key,
)


class IntentClassifier(BaseModel):
    message_intent: Literal["chat", "knowledge", "code"] = Field(
        ...,
        description="Classify whether the user wants to just chat, ask for knowledge or change code in the project.",
    )


class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_intent: str | None


def classify_intent(state: State):
    structured_llm = llm.with_structured_output(IntentClassifier)
    result = structured_llm.invoke(
        [
            {
                "role": "system",
                "content": 'Determine / classify whether the user wants to chat ("chat"), retrieve knowledge ("knowledge") or change code ("code"). ',
            },
            {"role": "user", "content": state["messages"][-1].content},
        ]
    )

    return {"message_intent": result.message_intent}


def prompt_llm_chat(state: State):
    messages = [
        {"role": "system", "content": "You are a talkative chatbot for fun. Be nice."}
    ] + state["messages"]

    response = llm.invoke(messages)

    return {"messages": [{"role": "assistant", "content": response.content}]}


def prompt_llm_rag(state: State):
    messages = [
        {
            "role": "system",
            "content": 'No matter what the user says, always say "I am the RAG agent"',
        }
    ] + state["messages"]

    response = llm.invoke(messages)

    return {"messages": [{"role": "assistant", "content": response.content}]}


def prompt_llm_code(state: State):
    messages = [
        {
            "role": "system",
            "content": 'No matter what the user says, always say "I am the CODING agent"',
        }
    ] + state["messages"]

    response = llm.invoke(messages)

    return {"messages": [{"role": "assistant", "content": response.content}]}


graph_builder = StateGraph(State)
graph_builder.add_node("classifier", classify_intent)
graph_builder.add_node("chat_agent", prompt_llm_chat)
graph_builder.add_node("rag_agent", prompt_llm_rag)
graph_builder.add_node("coding_agent", prompt_llm_code)

graph_builder.add_edge(START, "classifier")
graph_builder.add_conditional_edges(
    "classifier",
    lambda state: state["message_intent"],
    {"chat": "chat_agent", "knowledge": "rag_agent", "code": "coding_agent"},
)

graph_builder.add_edge("chat_agent", END)
graph_builder.add_edge("rag_agent", END)
graph_builder.add_edge("coding_agent", END)

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": uuid.uuid4()}}


for msg in [
    "Hello, I am Big Boss",
    "Do you know who I am?",
    "Check database and return the result.",
    "I want to change the code in the project.",
]:
    result = graph.invoke(
        {"messages": [{"role": "user", "content": msg}]},
        config=config,
    )
    print(result["messages"][-1].content)
