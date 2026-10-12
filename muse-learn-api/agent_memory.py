"""Multi-turn memory for the no-framework agent loop.

agent_loop.py is stateless: every run starts with a fresh message list, so a
follow-up like "give me the exact details" has nothing to refer back to. This
script adds the one missing piece from the agent article's "Adding memory"
section — a class-held message list that persists across turns — so turn 2
can resolve against turn 1.

    python agent_memory.py                 # scripted 2-turn demo, memory ON
    python agent_memory.py --no-memory     # same demo, memory OFF (watch turn 2)
    python agent_memory.py --interactive   # REPL; /save, /load, /quit
    python agent_memory.py --long-run      # 12 turns, watch the trim policy

.NET map for what's new here:
    Agent Framework AgentThread / ChatHistory -> Conversation.messages
    thread persistence (IThreadStateStore)    -> Conversation.save()/load()
    context-window management                 -> Conversation._trim()

The offline policy is identical to agent_loop.py's, except the search query
for a follow-up is grounded in the last substantive user question. That
reformulation is the offline stand-in for what a real model does with the
full message list — with OPENAI_API_KEY set, the model itself resolves
"it" / "the exact details" from history and no heuristic is needed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field

from app.rag import build_store


# --- Tools -----------------------------------------------------------------
# Same pair as agent_loop.py / /agent, duplicated so this file stays
# copy-paste portable (see agent_loop.py's module docstring).

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


# --- Loop ------------------------------------------------------------------

SYSTEM = (
    "You answer questions using the provided tools. Search the knowledge "
    "base first, then answer from what you observe. Cite chunk ids like "
    "[rag-1]. If you have what you need, answer directly with no tool call. "
    "Follow-up questions may refer to earlier turns (\"it\", \"the exact "
    "details\") — resolve them against the conversation history."
)

# Tokens that mark a question as a follow-up with a dangling referent.
FOLLOWUP_HINTS = (
    "it ", " it", "that ", "those ", "the exact", "exact details",
    "more about", "tell me more", "and then", "what about",
)

DETAIL_KEYWORDS = ("detail", "exact", "quote", "full text", "more about", "verbatim")


def _looks_like_followup(question: str) -> bool:
    q = f" {question.lower()} "
    return any(h in q for h in FOLLOWUP_HINTS)


def _resolve_query(question: str, prior_messages: list[dict]) -> str:
    """Ground a follow-up's search in the last substantive user question.

    Offline stand-in for real-model behaviour: with the full message list,
    the model itself would search on the resolved referent. Here we do the
    cheap deterministic version so the memory ON/OFF contrast is visible
    without a key.
    """
    if not _looks_like_followup(question):
        return question
    for msg in reversed(prior_messages):
        if msg["role"] == "user" and not _looks_like_followup(msg["content"]):
            return msg["content"]
    return question


@dataclass
class TurnResult:
    """What one ask() did — the inspectable trace, like /agent's response."""

    answer: str
    steps: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    read_chunk_ids: list[str] = field(default_factory=list)


class Conversation:
    """One agent thread: system prompt + every turn, trimmed to a window.

    The entire trick is that `messages` survives between ask() calls. The
    real-model path sends this whole list each turn; the offline path uses
    it for follow-up query resolution. _trim() keeps the list bounded.
    """

    def __init__(self, max_history: int = 8, memory: bool = True) -> None:
        self.max_history = max_history  # exchanges (user turns) retained
        self.memory = memory
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM}]

    # -- public API ------------------------------------------------------

    def ask(
        self,
        question: str,
        max_steps: int = 4,
        model: str = "gpt-4o-mini",
        offline: bool = False,
    ) -> TurnResult:
        """Run one turn. With memory, history persists; without, each turn
        starts from the bare system prompt (the agent_loop.py behaviour)."""
        has_key = bool(os.environ.get("OPENAI_API_KEY"))
        use_offline = offline or not has_key

        if self.memory:
            working = self.messages
            prior = list(working)  # snapshot before this turn's user message
        else:
            working = [{"role": "system", "content": SYSTEM}]
            prior = list(working)

        working.append({"role": "user", "content": question})

        if use_offline:
            result = self._run_offline_turn(working, prior, question, max_steps)
        else:
            result = self._run_real_turn(working, question, max_steps, model)

        working.append({"role": "assistant", "content": result.answer})
        if self.memory:
            self.messages = working
            self._trim()
        return result

    def save(self, path: str) -> None:
        """Persist the thread. .NET: IThreadStateStore / serialized ChatHistory."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.messages, f, indent=2)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            self.messages = json.load(f)

    def history_summary(self) -> str:
        users = sum(1 for m in self.messages if m["role"] == "user")
        return f"{len(self.messages)} messages, {users} user turn(s)"

    # -- internals ---------------------------------------------------------

    def _trim(self) -> None:
        """Context-window guard: keep the system prompt + the most recent
        max_history exchanges. Truncation is the naive policy — Exercise 1
        in WEEK2_MEMORY.md replaces it with summarisation."""
        user_idx = [i for i, m in enumerate(self.messages) if m["role"] == "user"]
        if len(user_idx) <= self.max_history:
            return
        cut = user_idx[-self.max_history]
        dropped = cut - 1  # everything between system and the kept window
        self.messages = [self.messages[0]] + self.messages[cut:]
        print(f"  [trim] dropped {dropped} message(s), kept last {self.max_history} exchange(s)")

    def _run_offline_turn(
        self,
        working: list[dict],
        prior: list[dict],
        question: str,
        max_steps: int,
    ) -> TurnResult:
        result = TurnResult(answer="")
        steps: list[str] = []
        retrieved = None

        for step in range(1, max_steps + 1):
            if retrieved is None:
                query = _resolve_query(question, prior) if self.memory else question
                name, args = "search_docs", {"query": query, "top_k": 3}
                result.search_queries.append(query)
            elif any(c in question.lower() for c in DETAIL_KEYWORDS) \
                    and not any(s.startswith("read_chunk") for s in steps):
                name, args = "read_chunk", {"chunk_id": retrieved[0].chunk.id}
            else:
                print(f"  step {step}: no tool call — answering")
                break

            _, fn = TOOLS[name]
            observation = str(fn(**args))
            print(f"  step {step}: {name}({json.dumps(args)})")
            print(f"    -> {observation[:160]}{'...' if len(observation) > 160 else ''}")

            tool_call_id = f"offline-{step}-{name}"
            working.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)},
                        }
                    ],
                }
            )
            working.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": name,
                    "content": observation,
                }
            )
            steps.append(f"{name}({json.dumps(args)})")
            if name == "read_chunk":
                result.read_chunk_ids.append(args["chunk_id"])
            if name == "search_docs":
                retrieved = _STORE.search(args["query"], top_k=3)

        context = "\n".join(steps) or "no context"
        result.steps = steps
        result.answer = f"(offline) Answer assembled from {context}."
        return result

    def _run_real_turn(
        self, working: list[dict], question: str, max_steps: int, model: str
    ) -> TurnResult:
        from openai import OpenAI  # lazy: script loads fine without the package

        client = OpenAI()
        schemas = [schema for schema, _ in TOOLS.values()]
        result = TurnResult(answer="")

        for step in range(1, max_steps + 1):
            response = client.chat.completions.create(
                model=model, messages=working, tools=schemas
            )
            msg = response.choices[0].message
            # Dict form, not the SDK object: keeps messages JSON-serialisable
            # so save()/load() and _trim() work on plain data.
            working.append(msg.model_dump(exclude_unset=True))

            if not msg.tool_calls:
                print(f"  step {step}: no tool call — answering")
                result.answer = msg.content or "(empty answer)"
                return result

            for call in msg.tool_calls:
                name = call.function.name
                args = json.loads(call.function.arguments or "{}")
                _, fn = TOOLS[name]
                observation = str(fn(**args))
                print(f"  step {step}: {name}({json.dumps(args)})")
                print(f"    -> {observation[:160]}{'...' if len(observation) > 160 else ''}")
                result.steps.append(f"{name}({json.dumps(args)})")
                if name == "search_docs":
                    result.search_queries.append(args.get("query", ""))
                if name == "read_chunk":
                    result.read_chunk_ids.append(args.get("chunk_id", ""))
                working.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": name,
                        "content": observation,
                    }
                )

        result.answer = "(hit max_steps without a final answer)"
        return result


# --- CLI -------------------------------------------------------------------

_STORE = build_store()

DEMO_TURNS = [
    "What is RAG?",
    "Give me the exact details.",
]

LONG_RUN_TURNS = [
    "What is RAG?",
    "Give me the exact details.",
    "What are embeddings?",
    "How do embeddings relate to RAG?",
    "What is MCP?",
    "Give me the exact details.",
    "What is a vector database?",
    "Which vector DB options were mentioned?",
    "Summarise RAG in one sentence.",
    "And embeddings?",
    "What was the first thing I asked?",
    "Give me the exact details.",
]


def _run_demo(turns: list[str], args: argparse.Namespace) -> Conversation:
    conv = Conversation(max_history=args.max_history, memory=not args.no_memory)
    if args.load:
        conv.load(args.load)
        print(f"(loaded history from {args.load}: {conv.history_summary()})")
    mode = "memory ON" if conv.memory else "memory OFF"
    print(f"=== {mode} ===")
    for i, question in enumerate(turns, 1):
        print(f"\n--- turn {i}: {question!r}")
        result = conv.ask(
            question, max_steps=args.max_steps, model=args.model, offline=args.offline
        )
        print(f"answer: {result.answer}")
        if result.search_queries:
            print(f"searched: {result.search_queries}")
        if result.read_chunk_ids:
            print(f"read_chunk: {result.read_chunk_ids}")
    print(f"\nthread: {conv.history_summary()}")
    if args.save:
        conv.save(args.save)
        print(f"(saved history to {args.save})")
    return conv


def _run_interactive(args: argparse.Namespace) -> None:
    conv = Conversation(max_history=args.max_history, memory=not args.no_memory)
    print("Multi-turn agent REPL. Commands: /save <path>, /load <path>, /history, /quit")
    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line == "/quit":
            break
        if line == "/history":
            print(conv.history_summary())
            continue
        if line.startswith("/save "):
            conv.save(line[6:].strip())
            print("(saved)")
            continue
        if line.startswith("/load "):
            conv.load(line[6:].strip())
            print(f"(loaded: {conv.history_summary()})")
            continue
        result = conv.ask(
            line, max_steps=args.max_steps, model=args.model, offline=args.offline
        )
        print(f"agent> {result.answer}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-turn agent loop: the agent_loop.py agent with conversation memory."
    )
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--max-history", type=int, default=8,
                        help="exchanges retained by the trim policy")
    parser.add_argument("--offline", action="store_true", help="force the deterministic policy")
    parser.add_argument("--no-memory", action="store_true",
                        help="reset the message list every turn (agent_loop.py behaviour)")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--interactive", action="store_true", help="REPL instead of the scripted demo")
    parser.add_argument("--long-run", action="store_true", help="12-turn scripted session")
    parser.add_argument("--save", metavar="PATH", help="save thread history to JSON after the demo")
    parser.add_argument("--load", metavar="PATH", help="load thread history from JSON before the demo")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY") and not args.offline:
        print("(no OPENAI_API_KEY — offline policy; set the key for the real loop)")
    if args.interactive:
        _run_interactive(args)
    else:
        _run_demo(LONG_RUN_TURNS if args.long_run else DEMO_TURNS, args)


if __name__ == "__main__":
    main()
