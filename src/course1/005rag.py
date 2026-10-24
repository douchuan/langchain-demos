"""
Since I cannot access the OpenAI API or find a compatible embedding service,
I use a local embedding model instead.
"""

from langchain.agents import create_agent
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from utils.config import load_settings

texts = [
    "Apple makes very good computers.",
    "I believe Apple is innovative!",
    "I love apples.",
    "I am a fan of MacBooks.",
    "I enjoy oranges.",
    "1",
    "I like Lenovo Thinkpads.",
    "I think pears taste very good.",
]

# Load the embedding model locally.
# The first run downloads the model; subsequent runs use the local cache.
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# Build the local FAISS vector store.
vector_store = FAISS.from_texts(
    texts,
    embedding=embeddings,
)

# print(vector_store.similarity_search("Apple are my favorite food."))
# print(vector_store.similarity_search("Linux is a great operating system."))

retriever = vector_store.as_retriever(search_kwargs={"k": 3})
retriever_tool = create_retriever_tool(
    retriever,
    name="kb_search",
    description="Search the small product / fruit database for information.",
)

settings = load_settings()
# Initialize the chat model using OpenAI-compatible API settings
client = ChatOpenAI(
    api_key=settings.api_key,
    model=settings.model,
    base_url=settings.base_url,
)

agent = create_agent(
    model=client,
    tools=[retriever_tool],
    system_prompt=(
        "You are a helpful assistant. For questions about Macs, apples, or laptops,"
        "first call the kb_search tool to retrieve context, then answer succinctly."
        "Maybe you have to use it multiple times before answering."
    ),
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What three fruits does the person like and what three fruits does the person dislike?",
            }
        ]
    }
)

print(result)
print(result["messages"][-1].content)
