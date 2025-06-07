# Project AGENTS.md Guide for OpenAI Codex

This AGENTS.md file provides comprehensive guidance for OpenAI Codex and other AI agents working with this codebase.

## Project Overview for OpenAI Codex

This repository contains a multi-module Python project for knowledge graph, RAG, agent-based applications, and data processing, including Streamlit-based interfaces and Neo4j integration.

## Project Structure for OpenAI Codex Navigation

- **Core Application Modules**: Python source code (primary focus for AI assistants)
  - `/unit12`: Main GraphRAG application (indexing, querying, Neo4j integration, Streamlit UI)
  - `/unit11`: Multi-agent system for stock analysis (planning, execution, Streamlit UI)
  - `/unit10`: Story RAG chatbot (graph-based QA)
  - `/unit08`: HR/Finance QA system (routing, data processing)
  - `/unit04`, `/unit05`, `/unit06`, `/unit07`, `/unit09`, `/unit13`: Additional NLP, data, and ML modules
- **Data**: Input and output data folders (e.g., `/unit12/data/`, `/unit12/index_data/`)
- **Testing**: Unit and integration tests in `tests/` subfolders within each module
- **Infrastructure**:
  - `Makefile`: Task automation (setup, test, build, run)
  - `docker-compose.yml`: Neo4j and other service orchestration
  - `setup_neo4j.sh`: Neo4j management script
- **Documentation**: Markdown files in each module, this AGENTS.md, and module-specific READMEs

## Scope and Focus Areas for OpenAI Codex

### Primary Focus for OpenAI Codex
- **Python backend development**: Data processing, knowledge graph, RAG, agent logic, and Streamlit UI
- **Database operations**: Neo4j integration, Cypher queries, and graph management
- **Testing**: Python unit tests (`test_*.py` files) and integration tests
- **Task automation**: Makefile targets for setup, build, test, and run
- **Dependency management**: Poetry for Python package/version control

### Out of Scope for OpenAI Codex
- **Frontend development**: No React/Next.js/TypeScript UI (except Streamlit)
- **Go/Node.js code**: Not present in this project
- **Legacy/unused scripts**: Unless specifically requested
- **Infrastructure changes**: Docker, deployment, or system-level changes unless requested

## Coding Conventions for OpenAI Codex

### General Conventions for OpenAI Codex
- Use Python 3.11.12 for all scripts and modules
- Manage dependencies and virtual environments with Poetry (`pyproject.toml`)
- Follow PEP8 and Black formatting standards
- Use meaningful variable and function names
- Add comments for complex business logic
- Keep functions and methods small and focused
- Must use type hints for function signatures
- Handle errors with try/except and informative messages

### Python Project Guidelines for OpenAI Codex
- Organize code into logical modules and packages
- Use relative imports within modules
- Separate core logic, utilities, and configuration
- Use environment variables for secrets and configuration (see `.env` usage)
- Implement CLI entry points where appropriate

### Database Conventions for OpenAI Codex
- Use Cypher queries for Neo4j operations
- Follow naming conventions for nodes, relationships, and properties
- Use parameterized queries to prevent injection
- Document any schema or index changes

## Formatting Standards for OpenAI Codex

- Use two spaces for indentation in Markdown files
- Use Black for Python code formatting (`black .`)
- Use isort for import sorting
- Use standard YAML/JSON formatting for config files

## Testing Requirements

### Prerequisites
Before running Python tests, ensure the following services are installed and running:

1. **Neo4j**: Install and start Neo4j (via Docker Compose)
   ```bash
   make neo4j-start
   make neo4j-status
   ```

2. **Python 3.11.12**: Use this version for all development and testing
   ```bash
   pyenv install 3.11.12
   pyenv local 3.11.12
   ```

3. **Poetry**: Install Poetry for dependency management
  ```bash
  pip install poetry
  poetry install --no-root --with dev
  ```

### Database Setup
- Ensure Neo4j is running and accessible (see `.env` for connection details)
- Use `make neo4j-reset` to clear and reinitialize the database if needed

### Running Tests
From the relevant module directory (e.g., `unit12`), run:
```bash
# Run all Python unit tests
make test

# Or directly with unittest/pytest
python -m unittest discover -v
pytest  # if pytest is used
```

## Pull Request Guidelines

When OpenAI Codex creates a PR, ensure it:

1. Includes a clear description of the changes and their purpose
2. References any related issues being addressed
3. Ensures all Python tests pass (`make test`)
4. Includes appropriate test coverage for new functionality
5. Keeps PRs focused on a single concern or feature
6. Updates documentation if API or CLI changes are made

## Required Checks and Quality Gates for OpenAI Codex

### Programmatic Checks for OpenAI Codex
Before submitting changes, run:

```bash
# Format code
black .
isort .

# Lint code
flake8 .
mypy .  # if type hints are used

# Run all tests
make test

# Poetry checks
poetry check
poetry lock --check
```

### Failure Handling for OpenAI Codex
- If any command fails due to missing dependencies or unavailable services, note the failure in the PR but do not block the commit
- All checks must pass before OpenAI Codex generated code can be merged
- If tests fail, attempt to fix them with a maximum of 50 iterations
- Document any unresolved issues in the PR description

## Database/Indexing Guidelines

- Use Cypher for Neo4j schema/index changes
- Document any changes to the graph schema in module README or a dedicated schema.md
- Test all graph operations for idempotency and correctness

## Security Considerations

- Validate all user inputs (especially for web/Streamlit interfaces)
- Use parameterized Cypher queries to prevent injection
- Handle sensitive data (API keys, passwords) via environment variables
- Log security-relevant events where appropriate

## Performance Guidelines

- Use Neo4j indexes appropriately for large graphs
- Implement caching strategies where beneficial (e.g., community summaries)
- Monitor and optimize slow queries
- Use batch operations for large data loads
- Implement pagination for large result sets in UIs

## Python & Dependency Management

- Always use Python 3.11.12
- Manage all dependencies and versions with Poetry (`pyproject.toml`, `poetry.lock`)
- Do not use `requirements.txt` for new dependencies; update Poetry files instead
- Use `poetry add <package>` to add new packages

---

**Note:** This guide is for OpenAI Codex and similar AI agents. Please follow these instructions to ensure high-quality, maintainable, and secure contributions to this Python project.