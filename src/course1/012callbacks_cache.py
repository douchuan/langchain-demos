"""
Callbacks and Token Tracking

Demonstrates the callback system for monitoring agent/model execution,
token usage tracking for cost estimation, and LLM result caching.
"""

import time

from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate

from utils.config import load_settings

# Load environment variables from .env file and application settings
settings = load_settings()


# --- Example 1: Custom Callback Handler ---
# Track timing, token counts, and model invocations
class DemoCallback(BaseCallbackHandler):
    """Custom callback that prints lifecycle events."""

    def __init__(self):
        super().__init__()
        self.model_start_time = None
        self.total_tokens = 0

    def on_llm_start(self, serialized, prompts, **kwargs):
        """Called when the LLM is about to be invoked."""
        self.model_start_time = time.time()
        print("[on_llm_start] Model is being called...")
        print(f"  Prompt length: {len(prompts[0]) if prompts else 0} chars")

    def on_llm_end(self, response, **kwargs):
        """Called when the LLM finishes generating."""
        elapsed = time.time() - self.model_start_time
        token_usage = (
            response.llm_output.get("token_usage", {}) if response.llm_output else {}
        )
        print(f"[on_llm_end] Done in {elapsed:.2f}s")
        print(
            f"  Tokens - prompt: {token_usage.get('prompt_tokens', '?')}, "
            f"completion: {token_usage.get('completion_tokens', '?')}, "
            f"total: {token_usage.get('total_tokens', '?')}"
        )

    def on_chain_start(self, serialized, inputs, **kwargs):
        """Called when a chain/runnable starts."""
        print(f"[on_chain_start] {serialized.get('name', 'unknown')}")

    def on_chain_end(self, outputs, **kwargs):
        """Called when a chain/runnable finishes."""
        print("[on_chain_end] Finished")

    def on_tool_start(self, serialized, input_str, **kwargs):
        """Called when a tool is invoked."""
        print(f"[on_tool_start] Tool: {serialized.get('name', 'unknown')}")

    def on_tool_end(self, output, **kwargs):
        """Called when a tool finishes."""
        print(f"[on_tool_end] Tool returned: {str(output)[:50]}...")


callback = DemoCallback()

model = init_chat_model(
    model=settings.model,
    model_provider="openai",
    base_url=settings.base_url,
    api_key=settings.api_key,
    temperature=0.1,
)

print("=== Example 1: Custom Callbacks ===")
response = model.invoke(
    "Explain what a closure is in one sentence.",
    config={"callbacks": [callback]},
)
print(f"\nResponse: {response.content}")

# --- Example 2: Token tracking with get_openai_callback ---
# For OpenAI-compatible APIs, track token consumption and estimated cost
print("\n=== Example 2: Token Tracking ===")

from langchain_community.callbacks import get_openai_callback

# get_openai_callback tracks all token usage within its context manager
with get_openai_callback() as cb:
    response1 = model.invoke("What is recursion? Answer in one sentence.")
    response2 = model.invoke("Give me 3 examples of recursive algorithms.")
    print(f"Total tokens used: {cb.total_tokens}")
    print(f"Prompt tokens: {cb.prompt_tokens}")
    print(f"Completion tokens: {cb.completion_tokens}")
    # Note: cost may show as 0 for non-OpenAI endpoints
    print(f"Estimated cost (USD): ${cb.total_cost:.6f}")

# --- Example 3: LLM Caching ---
# Cache model responses to avoid redundant API calls
print("\n=== Example 3: LLM Caching ===")

from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache

# Enable in-memory caching (resets when the process restarts)
set_llm_cache(InMemoryCache())

prompt = ChatPromptTemplate.from_template("What is the capital of {country}?")
chain = prompt | model

# First call: hits the API
print("First call (cache miss)...")
start = time.time()
result1 = chain.invoke({"country": "France"})
print(f"  Took {time.time() - start:.3f}s: {result1.content[:50]}...")

# Second call with the same input: served from cache (near-instant)
print("Second call (cache hit)...")
start = time.time()
result2 = chain.invoke({"country": "France"})
print(f"  Took {time.time() - start:.3f}s: {result2.content[:50]}...")

# Different input: cache miss again
print("Third call with different input (cache miss)...")
start = time.time()
result3 = chain.invoke({"country": "Germany"})
print(f"  Took {time.time() - start:.3f}s: {result3.content[:50]}...")

# Disable cache for subsequent calls
set_llm_cache(None)

# --- Example 4: SQLite Cache (persistent across restarts) ---
# Uncomment to use persistent file-based caching
# from langchain_community.cache import SQLiteCache
# set_llm_cache(SQLiteCache(database_path=".langchain_cache.db"))
print("\n=== Example 4: SQLite Cache (conceptual) ===")
print("SQLiteCache persists responses across process restarts.")
print("Usage: set_llm_cache(SQLiteCache(database_path='.langchain_cache.db'))")
