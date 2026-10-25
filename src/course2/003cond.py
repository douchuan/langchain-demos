import uuid
from typing import Annotated, Literal, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, AnyMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, ValidationError

from utils.config import load_settings

settings = load_settings()

llm = init_chat_model(
    model=settings.model,
    model_provider="openai",
    base_url=settings.base_url,
    api_key=settings.api_key,
)

# Single source of truth for intent values, derived from the Literal type.
Intent = Literal["chat", "knowledge", "code"]
INTENT_TO_NODE: dict[Intent, str] = {
    "chat": "chat_agent",
    "knowledge": "rag_agent",
    "code": "coding_agent",
}
DEFAULT_INTENT: Intent = "chat"


class IntentClassifier(BaseModel):
    message_intent: Intent = Field(
        ...,
        description="Classify whether the user wants to just chat, ask for knowledge or change code in the project.",
    )


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    message_intent: Intent | None


def classify_intent(state: State):
    structured_llm = llm.with_structured_output(IntentClassifier)
    try:
        result = structured_llm.invoke(
            [
                {
                    "role": "system",
                    "content": "Classify the user's intent.",
                },
                {"role": "user", "content": state["messages"][-1].content},
            ]
        )
        intent = result.message_intent
    except (ValidationError, AttributeError):
        # Fallback when the endpoint doesn't support structured output.
        intent = DEFAULT_INTENT

    return {"message_intent": intent}


def _last_user_content(state: State) -> str:
    return state["messages"][-1].content


def prompt_llm_chat(state: State):
    messages = [
        {"role": "system", "content": "You are a talkative chatbot for fun. Be nice."},
        {"role": "user", "content": _last_user_content(state)},
    ]
    response = llm.invoke(messages)
    return {"messages": [AIMessage(content=response.content)]}


def prompt_llm_rag(state: State):
    # Stub: replace with real RAG retrieval + generation.
    messages = [
        {
            "role": "system",
            "content": 'No matter what the user says, always say "I am the RAG agent"',
        },
        {"role": "user", "content": _last_user_content(state)},
    ]
    response = llm.invoke(messages)
    return {"messages": [AIMessage(content=response.content)]}


def prompt_llm_code(state: State):
    # Stub: replace with real coding agent logic.
    messages = [
        {
            "role": "system",
            "content": 'No matter what the user says, always say "I am the CODING agent"',
        },
        {"role": "user", "content": _last_user_content(state)},
    ]
    response = llm.invoke(messages)
    return {"messages": [AIMessage(content=response.content)]}


graph_builder = StateGraph(State)
graph_builder.add_node("classifier", classify_intent)
graph_builder.add_node("chat_agent", prompt_llm_chat)
graph_builder.add_node("rag_agent", prompt_llm_rag)
graph_builder.add_node("coding_agent", prompt_llm_code)

graph_builder.add_edge(START, "classifier")
graph_builder.add_conditional_edges(
    "classifier",
    lambda state: state["message_intent"],
    INTENT_TO_NODE,
)
for node in INTENT_TO_NODE.values():
    graph_builder.add_edge(node, END)

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": str(uuid.uuid4())}}

for msg in [
    "Hello, I am Big Boss",
    "Tell me our return policy.",
    "I want to refact the code.",
]:
    result = graph.invoke(
        {"messages": [{"role": "user", "content": msg}]},
        config=config,
    )
    print(f"[intent={result['message_intent']}] {result['messages'][-1].content}")
