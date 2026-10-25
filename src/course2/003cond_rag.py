import uuid
from typing import Annotated, Literal, TypedDict

from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, AnyMessage
from langchain_huggingface import HuggingFaceEmbeddings
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

# Create the vector store.
KNOWLEDGE = [
    "We want you to be completely satisfied with your purchase. If you are not happy with your order for any reason, you may return it within **30 days** of delivery for a full refund or exchange.",
    "Items must be unused, in their original condition, and returned in the original packaging with all tags and accessories included. Proof of purchase (order confirmation or receipt) is required. Items that are damaged, washed, altered, or missing parts may not be eligible for return.",
    "LangGraph is very low-level, and focused entirely on agent orchestration.",
    "Use LangSmith to trace, debug, and evaluate agents built with any of these frameworks. Follow the tracing quickstart to get set up. We recommend you also set up LangSmith Engine which monitors your traces, detects issues, and proposes fixes.",
    "Use LangChain (create_agent) for a highly customizable harness, easily tailored to your use case and data.",
    "Start with Deep Agents for a “batteries-included” agent with features like automatic context compression, a virtual filesystem, and subagent-spawning. Deep Agents are built on LangChain agents which you can also use directly.",
]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

vector_store = FAISS.from_texts(
    KNOWLEDGE,
    embedding=embeddings,
)


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
    query = _last_user_content(state)
    documents = vector_store.similarity_search(query, k=2)
    context = "\n".join(f"- {doc.page_content}" for doc in documents)

    messages = [
        {
            "role": "system",
            "content": f"You are a RAG agent. Answer the user using only the context below. If the answer is not in it, say you don't know. \n\nContext:\n{context}",
        },
        {"role": "user", "content": query},
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
    "Tell me our return policy.",
]:
    result = graph.invoke(
        {"messages": [{"role": "user", "content": msg}]},
        config=config,
    )
    print(f"[intent={result['message_intent']}] {result['messages'][-1].content}")
