
## Course List

|Course|Codes directory|
|---|---|
|[LangChain Full Crash Course - AI Agents in Python](https://www.youtube.com/watch?v=J7j5tCB_y4w)|[src/course1](src/course1)||
[LangGraph Crash Course - Agent Workflows in Python](https://www.youtube.com/watch?v=uWLJAtMOVT0)|[src/course2](src/course2)||

## Common Commands

This project uses [uv](https://docs.astral.sh/uv/) for dependency and environment management.

```bash
#####################################
# environment setup
#####################################

# Sync all dependencies (including dev group) and create/update the virtualenv
uv sync --all-groups

# Regenerate the lockfile
uv lock

#####################################
# Dependency Management
#####################################

# Add runtime dependencies
uv add requests httpx

# Add a dev dependency (e.g. Ruff)
uv add --dev ruff

# Remove a dependency
uv remove requests

#####################################
# Code Quality
#####################################

# Check code without modifying files
uv run ruff check .
uv run ruff check --fix .

# Format code
uv run ruff format .

#####################################
# Running Scripts
#####################################

# Run a script inside the project environment
uv run src/step1/006multimodal.py

#####################################
# Testing
#####################################

# Run all tests
uv run pytest -v

# Run tests with coverage
uv run pytest --cov=src --cov-report=term-missing


#####################################
# Maintenance
#####################################

# List dependencies with newer versions available
uv pip list --outdated

# Show dependency tree
uv tree

# Remove caches and the virtualenv (use with caution)
rm -rf .venv __pycache__ .pytest_cache .ruff_cache
```