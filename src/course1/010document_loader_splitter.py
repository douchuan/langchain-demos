"""
Document Loaders and Text Splitters

Demonstrates how to load documents from various sources (text files, CSV, web)
and split them into chunks suitable for embedding in a RAG pipeline.
"""

from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
from langchain_community.document_loaders import (
    CSVLoader,
    TextLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool
from langchain_huggingface import HuggingFaceEmbeddings

from utils.config import load_settings

# Load environment variables from .env file and application settings
settings = load_settings()

# --- Example 1: Load a text file ---
# Create a sample text file for demonstration
BASE_DIR = Path(__file__).resolve().parent
SAMPLE_TEXT = BASE_DIR / "sample_doc.txt"
SAMPLE_TEXT.write_text(
    "Python is a high-level programming language. "
    "It was created by Guido van Rossum and first released in 1991. "
    "Python emphasizes code readability with its notable use of significant whitespace. "
    "Its language constructs and object-oriented approach aim to help programmers write clear, logical code. "
    "Python is dynamically typed and garbage-collected. "
    "It supports multiple programming paradigms, including structured, object-oriented, and functional programming. "
    "Python is often described as a batteries-included language due to its comprehensive standard library. "
    "The language has a large standard library that provides tools suited to many tasks. "
    "Python consistently ranks as one of the most popular programming languages."
)

loader = TextLoader(str(SAMPLE_TEXT))
docs = loader.load()
print("=== Example 1: TextLoader ===")
print(f"Loaded {len(docs)} document(s)")
print(f"First doc metadata: {docs[0].metadata}")
print(f"First doc content (first 100 chars): {docs[0].page_content[:100]}...")

# --- Example 2: Split documents into chunks ---
# RecursiveCharacterTextSplitter is the most commonly used splitter
# It splits by characters in this order: \n\n, \n, " ", ""
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,  # Maximum characters per chunk
    chunk_overlap=20,  # Overlap between chunks to preserve context
    separators=["\n\n", "\n", "。", ".", " ", ""],  # Split boundaries in priority order
)

chunks = text_splitter.split_documents(docs)
print("\n=== Example 2: RecursiveCharacterTextSplitter ===")
print(f"Split into {len(chunks)} chunks")
for i, chunk in enumerate(chunks[:3]):
    print(f"  Chunk {i + 1}: {chunk.page_content}")

# --- Example 3: Token-based splitter ---
# Splits by token count instead of character count (more accurate for LLMs)
token_splitter = TokenTextSplitter(
    chunk_size=50,  # tokens per chunk
    chunk_overlap=5,  # overlap tokens
)

token_chunks = token_splitter.split_documents(docs)
print("\n=== Example 3: TokenTextSplitter ===")
print(f"Split into {len(token_chunks)} chunks (by tokens)")
print(f"  First chunk: {token_chunks[0].page_content[:80]}...")

# --- Example 4: Create a CSV file and load it ---
SAMPLE_CSV = BASE_DIR / "sample_data.csv"
SAMPLE_CSV.write_text(
    "name,age,city\nAlice,30,Beijing\nBob,25,Shanghai\nCharlie,35,Guangzhou\n"
)

csv_loader = CSVLoader(file_path=str(SAMPLE_CSV))
csv_docs = csv_loader.load()
print("\n=== Example 4: CSVLoader ===")
print(f"Loaded {len(csv_docs)} rows from CSV")
for doc in csv_docs[:2]:
    print(f"  {doc.page_content[:60]}...")

# --- Example 5: Full RAG pipeline with Document Loader + Splitter + Vector Store ---
print("\n=== Example 5: Full RAG Pipeline ===")

# Use the Python knowledge text from Example 1
all_chunks = text_splitter.split_documents(docs)

# Create embeddings using local model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# Build vector store from chunks
vector_store = FAISS.from_documents(all_chunks, embedding=embeddings)

# Create retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# Initialize chat model
model = init_chat_model(
    model=settings.model,
    model_provider="openai",
    base_url=settings.base_url,
    api_key=settings.api_key,
    temperature=0.1,
)

# Create retriever tool for an agent
retriever_tool = create_retriever_tool(
    retriever,
    name="python_knowledge",
    description="Search for information about the Python programming language.",
)

# Create an agent that uses the retriever
from langchain.agents import create_agent

agent = create_agent(
    model=model,
    tools=[retriever_tool],
    system_prompt=(
        "You are a Python expert. Use the python_knowledge tool to look up "
        "information before answering. Always cite the retrieved context."
    ),
)

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What programming paradigms does Python support?",
            }
        ]
    }
)
print(response["messages"][-1].content)
