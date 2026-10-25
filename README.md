# LangChain & LangGraph Learning Path

A hands-on collection of runnable examples covering LangChain and LangGraph fundamentals. Each file is a standalone, runnable script — open it, read it, and run it.

## Characteristics

- **Modern Python project** — dependency management, virtual environments, and packaging all handled by [uv](https://github.com/astral-sh/uv)
- **Python best practices** — type hints, docstrings, clean imports, and consistent code style enforced by Ruff
- **LangChain / LangGraph / LangSmith** — progressive examples from elementary model invocation to advanced multi-agent orchestration

## Quick Start

```bash
# Install dependencies
uv sync

# Run any example
uv run src/course1/001agent.py
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