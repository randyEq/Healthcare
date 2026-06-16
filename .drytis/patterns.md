# Patterns

## Coding Standards
- Python 3.10+ with type hints
- Pydantic models for config and data validation
- LangGraph TypedDict for state
- Async/await throughout
- Structured logging via loguru

## Error Handling
- Try/except with meaningful error messages
- Graceful degradation (if RAG retrieval fails, still provide reasoning)
- Never expose raw stack traces to user

## Agent Pattern
- Each agent is a class with an `async def run(state)` method
- Agents receive and return the shared LangGraph State
- Agents use Azure OpenAI via LangChain's ChatOpenAI (OpenAI-compatible)

## Testing
- pytest for unit tests
- Test each agent in isolation
- Mock LLM calls in tests
