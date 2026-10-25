"""
LangChain Expression Language (LCEL)

Demonstrates the Runnable composition pattern using the | operator,
RunnablePassthrough, RunnableParallel, and RunnableLambda.
LCEL is the core compositional paradigm in modern LangChain.
"""

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from utils.config import load_settings
from utils.debug import DebugPromptHandler

# Load environment variables from .env file and application settings
settings = load_settings()

# Initialize the chat model
model = init_chat_model(
    model=settings.model,
    model_provider="openai",
    base_url=settings.base_url,
    api_key=settings.api_key,
    temperature=0.1,
)

# --- Example 1: Basic chain with | operator ---
# prompt -> model -> string output (most common pattern)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a concise assistant."),
        ("human", "{question}"),
    ]
)

# The | operator chains runnables left to right
chain = prompt | model | StrOutputParser()
response = chain.invoke({"question": "What is the time complexity of binary search?"})
print("=== Example 1: Basic Chain ===")
print(response)


# --- Example 2: RunnablePassthrough ---
# Passes the input through unchanged, useful for merging data into a later step
# Here we compute a "context" field and pass the original input alongside it
def build_context(input_text: str) -> str:
    """Simulate a retrieval step that returns relevant context."""
    return f"Binary search works on sorted arrays by halving the search space. Topic: {input_text}"


chain_with_passthrough = (
    RunnablePassthrough.assign(context=build_context)
    | ChatPromptTemplate.from_messages(
        [
            ("system", "Answer using this context: {context}"),
            ("human", "{question}"),
        ]
    )
    | model
    | StrOutputParser()
)

response = chain_with_passthrough.invoke(
    {
        "question": "How does it work?",
    },
    config={"callbacks": [DebugPromptHandler()]},
)
print("\n=== Example 2: RunnablePassthrough ===")
print(response)

# --- Example 3: RunnableParallel ---
# Runs multiple branches in parallel, collecting results into a dict
chain_parallel = RunnableParallel(
    {
        "joke": (
            ChatPromptTemplate.from_template("Tell a short joke about {topic}")
            | model
            | StrOutputParser()
        ),
        "fact": (
            ChatPromptTemplate.from_template("Tell an interesting fact about {topic}")
            | model
            | StrOutputParser()
        ),
    }
)

results = chain_parallel.invoke(
    {"topic": "Python programming"}, config={"callbacks": [DebugPromptHandler()]}
)
print("\n=== Example 3: RunnableParallel ===")
print(f"Joke: {results['joke']}")
print(f"Fact: {results['fact']}")


# --- Example 4: RunnableLambda ---
# Wrap any Python function into a Runnable for use in chains
def summarize_and_translate(text: str) -> dict:
    """Post-process the model output: extract and label it."""
    return {"original": text, "word_count": len(text.split())}


chain_with_lambda = (
    prompt | model | StrOutputParser() | RunnableLambda(summarize_and_translate)
)

result = chain_with_lambda.invoke({"question": "Name three benefits of exercise."})
print("\n=== Example 4: RunnableLambda ===")
print(result)


# --- Example 5: Full RAG-style chain ---
# Simulate retrieval -> format -> answer -> parse
def fake_retriever(query: str) -> str:
    """Simulate a vector store retrieval."""
    return "Exercise improves cardiovascular health, boosts mood through endorphins, and strengthens muscles."


rag_chain = (
    {"context": RunnableLambda(fake_retriever), "question": RunnablePassthrough()}
    | ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Use the context below to answer the question.\nContext: {context}",
            ),
            ("human", "{question}"),
        ]
    )
    | model
    | StrOutputParser()
)

response = rag_chain.invoke(
    "What are the benefits of exercise?", config={"callbacks": [DebugPromptHandler()]}
)
print("\n=== Example 5: RAG-style Chain ===")
print(response)
