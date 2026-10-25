# AI Era Learning Path

> **LangChain, LangGraph & Machine Learning — become an expert Agent builder in 1 day**

A hands-on learning path covering **three courses**: LangChain, LangGraph, and Machine Learning. Designed for people who want to pick up the core knowledge of the AI era in a short time, without wading through scattered documentation.

Every file is a standalone, runnable script — open it, read it, and run it.

<div align="center">
<img src="./src/res/march_of_ai_progress.png" alt="AI Era Learning Path" width="600">
</div>


## Characteristics

- **Modern Python project** — dependency management, virtual environments, and packaging all handled by [uv](https://github.com/astral-sh/uv)
- **Python best practices** — type hints, docstrings, clean imports, and consistent code style enforced by Ruff
- **LangChain / LangGraph / LangSmith** — progressive examples from elementary model invocation to advanced multi-agent orchestration
- **Detailed comments** — every file explains the what and why inline, so you learn by reading, not just by running

## Quick Start

```bash
# Install dependencies
uv sync

# Run any example
uv run src/course1/001agent.py
```

Before running, create a `.env` file:
```
# OpenAI-compatible
API_KEY=your_api_key
LLM_MODEL=your_model_name
LLM_MODEL2=your_advanced_model_name
BASE_URL=your_api_base_url
```

## Course List

| Course | Source | Topics |
|---|---|---|
| LangChain Crash Course | [src/course1](src/course1) | Model invocation, chat, streaming, agents & tools, multimodal (image), embeddings, RAG, middleware (hooks, dynamic prompts, model routing), prompt templates, LCEL, output parsing, document loaders & splitters, callbacks & caching |
| LangGraph Crash Course | [src/course2](src/course2) | StateGraph, conditional routing, multi-agent workflows, tool integration |
| Machine Learning Crash Course | [src/course3](src/course3) | Become an Expert in Data Analysis in 1 Day|
