"""No-framework agent loop — the real-model twin of POST /agent.

The Week 2 /agent endpoint (app/main_chat_agent.py) uses a *deterministic*
policy so the wiring is inspectable offline. This script is the same agent
with the decision-making handed to a real model via OpenAI function calling,
and no FastAPI anywhere — just stdlib + `openai`.

    python agent_loop.py "What is RAG? Give me the exact details."

With OPENAI_API_KEY set (and `pip install openai`), the model itself decides
which tools to call and when to stop. Without a key it falls back to the
deterministic policy from /agent so you can still watch the loop mechanics.
Pass --offline to force the fallback even with a key.

.NET map for what's new here:
    Agent Framework FunctionTool  ->  the TOOLS list below (name + schema + fn)
    ChatCompletionAgent loop      ->  run_agent(): model -> tool_calls -> tools
    IChatClient with tools        ->  OpenAI().chat.completions.create(tools=...)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from app.rag import build_store

logger = logging.getLogger(__name__)

# --- Tools -----------------------------------------------------------------
# Same pair as /agent: search_docs + read_chunk. Here they live in one file
# so this script is copy-paste portable; only the store is shared.

def _search_docs(query: str, top_k: int = 3) -> str:
    results = _STORE.search(query, top_k=top_k)
    if not results:
        return "no matches"
    return "\n".join(
        f"[{r.chunk.id}] ({r.chunk.source}, score={r.score:.3f}) {r.chunk.text}"
        for r in results
    )


def _read_chunk(chunk_id: str) -> str:
    for chunk in _STORE._chunks:  # small demo store; a real one queries by id
        if chunk.id == chunk_id:
            return f"[{chunk.id}] {chunk.source}:\n{chunk.text}"
    return f"unknown chunk id: {chunk_id}"


# name -> (openai tool schema, implementation)
TOOLS = {
    "search_docs": (
        {
            "type": "function",
            "function": {
                "name": "search_docs",
                "description": "Search the knowledge base for chunks relevant to a query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "default": 3},
                    },
                    "required": ["query"],
                },
            },
        },
        _search_docs,
    ),
    "read_chunk": (
        {
            "type": "function",
            "function": {
                "name": "read_chunk",
                "description": "Read the full text of one knowledge-base chunk by its id.",
                "parameters": {
                    "type": "object",
                    "properties": {"chunk_id": {"type": "string"}},
                    "required": ["chunk_id"],
                },
            },
        },
        _read_chunk,
    ),
}


# --- The loop --------------------------------------------------------------

SYSTEM = (
    "You answer questions using the provided tools. Search the knowledge "
    "base first, then answer from what you observe. Cite chunk ids like "
    "[rag-1]. If you have what you need, answer directly with no tool call."
)


def run_real(question: str, max_steps: int, model: str) -> str:
    """Model-driven loop: the model chooses tools until it answers."""
    from openai import OpenAI  # lazy: script loads fine without the package

    client = OpenAI()  # reads OPENAI_API_KEY from the environment
    schemas = [schema for schema, _ in TOOLS.values()]
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]

    for step in range(1, max_steps + 1):
        response = client.chat.completions.create(
            model=model, messages=messages, tools=schemas
        )
        msg = response.choices[0].message
        messages.append(msg)  # SDK message objects append cleanly

        if not msg.tool_calls:
            print(f"  step {step}: no tool call — answering")
            return msg.content or "(empty answer)"

        for call in msg.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments or "{}")
            _, fn = TOOLS[name]
            observation = str(fn(**args))
            print(f"  step {step}: {name}({json.dumps(args)})")
            print(f"    -> {observation[:160]}{'...' if len(observation) > 160 else ''}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": name,
                    "content": observation,
                }
            )

    return "(hit max_steps without a final answer)"


def run_offline(question: str, max_steps: int) -> str:
    """Deterministic policy — identical decisions to /agent, no key needed."""
    steps: list[str] = []
    retrieved = None

    for step in range(1, max_steps + 1):
        if retrieved is None:
            name, args = "search_docs", {"query": question, "top_k": 3}
        elif any(c in question.lower() for c in ("detail", "exact", "quote", "full text", "more about", "verbatim")) \
                and not any(s.startswith("read_chunk") for s in steps):
            name, args = "read_chunk", {"chunk_id": retrieved[0].chunk.id}
        else:
            print(f"  step {step}: no tool call — answering")
            break
        _, fn = TOOLS[name]
        observation = str(fn(**args))
        print(f"  step {step}: {name}({json.dumps(args)})")
        print(f"    -> {observation[:160]}{'...' if len(observation) > 160 else ''}")
        steps.append(f"{name}({json.dumps(args)})")
        if name == "search_docs":
            retrieved = _STORE.search(question, top_k=3)

    context = "\n".join(steps) or "no context"
    return f"(offline) Answer assembled from {context}. Set OPENAI_API_KEY for real generation."


# --- CLI -------------------------------------------------------------------

_STORE = build_store()


def main() -> None:
    parser = argparse.ArgumentParser(description="No-framework agent loop over the Week 2 doc store.")
    parser.add_argument("question", help="the question to answer")
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--offline", action="store_true", help="force the deterministic policy")
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()

    has_key = bool(os.environ.get("OPENAI_API_KEY"))
    if has_key and not args.offline:
        try:
            mode, answer = "real model", run_real(args.question, args.max_steps, args.model)
        except ImportError:
            sys.exit("openai package not installed: pip install openai")
    else:
        if not has_key:
            print("(no OPENAI_API_KEY — offline policy; set the key for the real loop)")
        mode, answer = "offline policy", run_offline(args.question, args.max_steps)

    print(f"\n[{mode}] {answer}")


if __name__ == "__main__":
    main()
