# LangChain & LangGraph Learning Path

A hands-on collection of runnable examples covering LangChain and LangGraph fundamentals. Each file is a standalone, runnable script — open it, read it, and run it.

## Quick Start

```bash
# Install dependencies
uv sync

# Run any example
uv run python src/course1/001agent.py
```

Before running, create a `.env` file:
```
API_KEY=your_api_key
LLM_MODEL=your_model_name
LLM_MODEL2=your_advanced_model_name
BASE_URL=your_api_base_url
```

## Course List

| Course | Source | Topics |
|---|---|---|
| [LangChain Full Crash Course - AI Agents in Python](https://www.youtube.com/watch?v=J7j5tCB_y4w) | [src/course1](src/course1) | Model invocation, chat, streaming, agents & tools, multimodal (image), embeddings, RAG, middleware (hooks, dynamic prompts, model routing), prompt templates, LCEL, output parsing, document loaders & splitters, callbacks & caching |
| [LangGraph Crash Course - Agent Workflows in Python](https://www.youtube.com/watch?v=uWLJAtMOVT0) | [src/course2](src/course2) | StateGraph, conditional routing, multi-agent workflows, tool integration |

## Course 1 — Topic Index

| File | Topic | Key Concepts |
|---|---|---|
| `001agent.py` | Agent & Tools | `create_agent`, `@tool`, tool invocation loop |
| `002chat.py` | Basic Model Call | `init_chat_model`, single invoke |
| `002conversation.py` | Multi-turn Chat | System/Human/AI messages, conversation history |
| `002stream.py` | Streaming Output | `model.stream()`, token-by-token output |
| `003session.py` | Advanced Agent | Context schema, structured response, checkpointer (memory) |
| `004multimodal1.py` | Multimodal (URL) | Image analysis via URL |
| `004multimodal2.py` | Multimodal (Base64) | Local image via base64 encoding |
| `004multimodal3.py` | Multimodal (HumanMessage) | Base64 image with typed message |
| `005embedding.py` | Embeddings | HuggingFace embeddings + FAISS vector store |
| `005rag.py` | RAG | Retriever tool + agent-based retrieval |
| `006middleware_agent.py` | Middleware Hooks | `AgentMiddleware` lifecycle (before/after agent/model) |
| `006middleware_dynamic_prompt.py` | Dynamic Prompt | `@dynamic_prompt` — context-driven system prompts |
| `006middleware_model.py` | Dynamic Model | `@wrap_model_call` — conditional model routing |
| `007prompt_template.py` | Prompt Templates | `ChatPromptTemplate`, `MessagesPlaceholder`, few-shot prompting |
| `008lcel.py` | LCEL | `\|` operator, `RunnablePassthrough`, `RunnableParallel`, `RunnableLambda` |
| `009output_parsing.py` | Output Parsing | `PydanticOutputParser`, `JsonOutputParser`, `with_structured_output` |
| `010document_loader_splitter.py` | Docs & Splitters | Text/CSV loaders, `RecursiveCharacterTextSplitter`, `TokenTextSplitter`, full RAG pipeline |
| `011langgraph.py` | LangGraph Basics | `StateGraph`, `ToolNode`, conditional edges, multi-node workflows, routing |
| `012callbacks_cache.py` | Callbacks & Cache | `BaseCallbackHandler`, token tracking, `InMemoryCache`, `SQLiteCache` |