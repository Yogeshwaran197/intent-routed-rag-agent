# Intent-Routed RAG Agent

A LangChain agent that classifies incoming queries and routes them to one of three paths: tool-calling (weather/calculator), RAG retrieval, or general chat. Session-based conversation memory is maintained across turns.

## How it works

1. **Intent classification** — every user query is passed through an LLM prompt that labels it as `weather`, `calculate`, `rag`, or `general`.
2. **Routing** — based on that label, the query goes to one of three handlers:
   - `run_with_tools` — binds `get_weather` and `calculate` as tools to the LLM. If the model requests a tool call, the tool runs and its result is fed back to the LLM for a final answer.
   - `run_rag` — retrieves relevant chunks from a Chroma vector store (MMR search) and answers strictly from that context.
   - `run_general` — plain conversational LLM call, no retrieval or tools.
3. **Memory** — each session's message history is stored in an in-memory dict (`ChatMessageHistory`) keyed by `session_id`, so follow-up questions have context.

## Stack

- **LLM**: Groq (`openai/gpt-oss-120b`)
- **Embeddings**: `BAAI/bge-m3` via `langchain_huggingface`
- **Vector store**: ChromaDB, MMR retrieval (`fetch_k=6, k=3, lambda_mult=0.7`)
- **Framework**: LangChain Core (LCEL, `RunnableLambda`/`Passthrough`, tool binding)

## Known limitations (being honest about this)

- **Intent classification is a single LLM call with no fallback** — if the classifier returns something unexpected, it silently falls through to `general`. There's no confidence score or retry logic.
- **Memory is in-process only** — the `store` dict resets every time the runtime restarts. No persistence to disk or a database.
- **`calculate` uses `eval()`** — sandboxed with empty builtins, but this is not a secure expression evaluator. Fine for a demo, not for production or untrusted input.
- **RAG context is small and hardcoded** — the demo corpus is 30 short `Document` objects about RAG concepts, not a real ingested dataset. This project demonstrates the *pipeline*, not a production knowledge base.
- **No streaming for tool-calling responses** — `run_rag` and `run_general` stream token-by-token; `run_with_tools` returns the full response at once, so the UX is inconsistent between paths.
- **No automated tests.** Everything here was verified manually by running queries through each path.

## Setup

```bash
pip install langchain_community langchain_core langchain_groq chromadb langchain_huggingface
```

Requires a `GROQ_API_KEY` (loaded via `google.colab.userdata` in this version — swap for `os.environ` outside Colab).

## Example

```
Ask Anything(Weather or Calculate or General): weather of trichy?
The weather in Tiruchirappalli is ...

Ask Anything(Weather or Calculate or General): what is chunk overlap?
Chunk overlap preserves context across chunk boundaries by repeating a small portion of text...
```

## Why this project

Built as part of a structured LLM engineering study plan, to demonstrate LCEL composition, tool binding, retrieval, and conversation memory in one working system rather than in isolated tutorial snippets.
