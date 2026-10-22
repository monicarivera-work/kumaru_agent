"""`ChatAgent` — orchestrates one conversational turn.

Its job, in order: normalize the system prompt, fit the history into the context
budget, call the backend, run the tools the model calls (asking the approver
first whenever one would change something), and measure what happened. It owns
*policy*; the backend owns *mechanism*.

Statelessness is a design decision, not an omission
---------------------------------------------------
`ChatAgent` stores no conversation history. Callers pass the full message list
on every turn, exactly like the OpenAI API. This is what makes horizontal
scaling trivial: any replica can serve any request, so you scale by starting
more processes behind a load balancer with no session affinity, no shared cache,
and no sticky routing. Persistence, when you want it, belongs in a store the
caller owns — not smuggled into the agent. A turn that runs tools adds several
messages (the calls, their results, the answer); pass a ``transcript`` list to
receive them, and append it to your history.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

from kumaru.agent.knowledge import load_pack, reference_context
from kumaru.agent.memory import TokenCounter, get_policy
from kumaru.agent.prompt import build_system_prompt, ensure_system_message
from kumaru.agent.tools import Approver, ToolBelt
from kumaru.backends.base import Backend
from kumaru.core.config import KumaruConfig
from kumaru.core.errors import ContextLengthExceededError
from kumaru.core.logging import get_logger
from kumaru.core.types import (
    ChatResult,
    FinishReason,
    GenerationParams,
    Message,
    Role,
    StreamChunk,
    Usage,
)

logger = get_logger(__name__)


class ChatAgent:
    """Turns a message list into a reply, using a backend and a memory policy."""

    def __init__(
        self, backend: Backend, config: KumaruConfig, *, approver: Approver | None = None
    ) -> None:
        """Read config, knowledge packs and tools; never touch the backend.

        Callers build the agent *before* `backend.load()`, so an unreadable or
        invalid pack, a missing project folder, or a backend that cannot call
        the enabled tools fails while there is still no model to release.

        `approver` is asked before every tool call that would change something.
        Without one, those tools and the project-folder tools are not offered.
        """
        self.backend = backend
        self.config = config
        self.memory = get_policy(config.agent.memory_policy)
        self.tools = ToolBelt(
            config.agent.enabled_tools, workspace=config.agent.workspace, approver=approver
        )
        if self.tools.withheld:
            logger.warning(
                "not offering %s: they need approval, and nobody here can give it",
                ", ".join(self.tools.withheld),
            )
        self._specs = self.tools.specs() if self.tools else None
        # The chat template renders the declarations into the prompt, so they
        # use context too; estimated like the default token counter, rounded up.
        self._tool_tokens = 0
        if self._specs:
            backend.check_tools(self._specs)
            self._tool_tokens = int(len(json.dumps(self._specs)) / 3) + 150
        self.knowledge = [load_pack(Path(path)) for path in config.agent.knowledge_files]

    # -- prompt assembly ----------------------------------------------------

    def prepare(
        self,
        messages: Sequence[Message],
        *,
        extra_context: str | None = None,
        reserve_tokens: int | None = None,
    ) -> list[Message]:
        """Normalize and budget the conversation. Pure and side-effect free.

        Kept public and separate from generation so tests can assert on the
        exact prompt without running a model — the fastest way to catch the
        class of bug where the model is fine but the prompt is wrong.

        `reserve_tokens` is the room kept free for the reply; it defaults to the
        configured `max_tokens`. Thinking replies are far longer, so the
        generation methods pass the effective value for the current request.
        At most half the window is reserved: `max_tokens` is a ceiling, not a
        forecast, and the generation methods trim it to the room actually left.
        """
        references = reference_context(self.knowledge, self.config.agent.max_knowledge_chars)
        context = "\n\n".join(part for part in (references, extra_context) if part)
        # No tool list here: the declarations travel in the request (`tools`),
        # and the chat template renders them, so listing them again would only
        # spend history budget.
        system_prompt = build_system_prompt(self.config.agent, extra_context=context)
        normalized = ensure_system_message(messages, system_prompt)

        # Reserve room for the answer: the budget must cover prompt *and*
        # completion, or generation is truncated right after it starts. Capped
        # at half the window: reserving qwen36's 32k cap in full would leave a
        # 32k window only the 512-token floor for history. Tool declarations
        # take their share of the window too.
        window = self.backend.context_length
        reserved = reserve_tokens if reserve_tokens is not None else self.config.generation.max_tokens
        reserved = min(reserved, window // 2)
        budget = min(
            self.config.agent.max_history_tokens,
            max(512, window - reserved - 128 - self._tool_tokens),
        )

        prepared = _keep_well_formed(
            self.memory.apply(normalized, budget, self.backend.count_tokens),
            normalized,
            budget,
            self.backend.count_tokens,
        )
        if (self.config.agent.expertise or self.knowledge) and self.backend.count_tokens(prepared) > budget:
            raise ContextLengthExceededError(
                "Engineering guidance, references and recent messages exceed the prompt budget.",
                hint="Reduce agent.max_knowledge_chars or expertise, shorten the request, "
                "or increase the context/history budget.",
            )
        return prepared

    def resolve_params(self, **overrides) -> GenerationParams:
        """Resolve the decoding parameters for one request.

        Precedence, lowest to highest: ``generation`` (config), then
        ``thinking_mode`` (config, applied only when thinking is on), then the
        per-request overrides. Reasoning models want different sampling when
        they think — Qwen recommends temperature 1.0 thinking vs 0.7 not — and
        many more output tokens, so one flat preset cannot serve both modes.

        Public so the benchmark runner decodes exactly the way chat does.
        """
        base = self.config.generation
        thinking = overrides.get("thinking")
        if thinking is None:
            thinking = base.thinking
        if thinking:
            base = base.merged(**self.config.thinking_mode.model_dump(exclude_none=True))
        return base.merged(**overrides)

    def _fit(
        self, messages: Sequence[Message], extra_context: str | None, overrides: dict
    ) -> tuple[list[Message], GenerationParams]:
        """Resolve the params, budget the history, then cap the reply to the room left.

        Strict servers (vLLM, OpenAI) reject a request whose prompt plus
        `max_tokens` exceeds the window instead of stopping at its end.
        """
        params = self.resolve_params(**overrides)
        prepared = self.prepare(messages, extra_context=extra_context, reserve_tokens=params.max_tokens)
        room = self.backend.context_length - self.backend.count_tokens(prepared) - self._tool_tokens
        if params.max_tokens > room:
            params = params.merged(max_tokens=max(room, 1))
        return prepared, params

    # -- generation ---------------------------------------------------------

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        extra_context: str | None = None,
        transcript: list[Message] | None = None,
        **overrides,
    ) -> ChatResult:
        """Produce a full reply. With tools, this runs the `stream` loop and gathers it."""
        if self._specs:
            return await self._gather(messages, extra_context, transcript, overrides)
        prepared, params = self._fit(messages, extra_context, overrides)
        result = await self.backend.generate(prepared, params)
        logger.info(
            "completed: %d prompt + %d completion tokens in %.0fms (%.1f tok/s)",
            result.usage.prompt_tokens,
            result.usage.completion_tokens,
            result.latency_ms,
            result.tokens_per_second,
        )
        if transcript is not None:
            transcript.append(Message.assistant(result.message.content))
        return result

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        extra_context: str | None = None,
        transcript: list[Message] | None = None,
        **overrides,
    ) -> AsyncIterator[StreamChunk]:
        """Yield the reply incrementally.

        This is the path the UI uses. Time-to-first-token dominates *perceived*
        speed far more than total tokens/second does, so it is measured and
        logged on every request.

        With tools offered, the model may call them before it answers. The
        stream then also carries each round's calls (`tool_calls`) and each
        result (`tool_result`) between the text, and exactly one final chunk
        carries the finish reason: "tool_calls" if ``agent.max_tool_iterations``
        stopped the loop. `transcript`, when given, receives every message the
        turn adds, in order.
        """
        started = time.perf_counter()
        ttft_ms: float | None = None
        emitted = 0
        turn: list[Message] = []

        def record(message: Message) -> None:
            turn.append(message)
            if transcript is not None:
                transcript.append(message)

        if not self._specs:
            prepared, params = self._fit(messages, extra_context, overrides)
            parts: list[str] = []
            async for chunk in self.backend.stream(prepared, params):
                # With thinking on, the first token is a thought, and that is what
                # the user perceives as "it started answering".
                if chunk.delta or chunk.reasoning:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - started) * 1000
                    emitted += 1
                parts.append(chunk.delta)
                yield chunk
            record(Message.assistant("".join(parts)))
        else:
            cap = self.config.agent.max_tool_iterations
            rounds = 0
            usage: Usage | None = None
            while True:
                prepared, params = self._fit([*messages, *turn], extra_context, overrides)
                text: list[str] = []
                calls: list[dict[str, Any]] = []
                finish: FinishReason | None = None
                round_usage: Usage | None = None
                # Tools run only after the model's reply is complete.
                async for chunk in self.backend.stream(prepared, params, tools=self._specs):
                    if chunk.delta or chunk.reasoning:
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - started) * 1000
                        emitted += 1
                        text.append(chunk.delta)
                        yield StreamChunk(delta=chunk.delta, reasoning=chunk.reasoning)
                    calls.extend(chunk.tool_calls or [])
                    finish = chunk.finish_reason or finish
                    round_usage = chunk.usage or round_usage
                if round_usage is not None:
                    usage = _add_usage(usage, round_usage)
                content = "".join(text)

                # Calls cut off by the token limit are incomplete: never run them.
                if not calls or finish == "length":
                    record(Message.assistant(content))
                    yield StreamChunk(finish_reason=finish or "stop", usage=usage)
                    break

                calls = _normalize_calls(calls)
                record(Message(role=Role.ASSISTANT, content=content, tool_calls=calls))
                yield StreamChunk(tool_calls=calls)
                if rounds >= cap:
                    # Every call gets a result, or the next request is malformed.
                    for call in calls:
                        result = _tool_message(call, _NOT_RUN.format(rounds=_rounds(cap)))
                        record(result)
                        yield StreamChunk(tool_result=result)
                    yield StreamChunk(
                        delta=f"\n[Stopped after {_rounds(cap)} (agent.max_tool_iterations). "
                        "Reply 'continue' to let it keep going.]"
                    )
                    yield StreamChunk(finish_reason="tool_calls", usage=usage)
                    break
                rounds += 1
                for call in calls:
                    result = await self._run_call(call)
                    record(result)
                    yield StreamChunk(tool_result=result)

        total_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "streamed: %d chunks in %.0fms (ttft %.0fms)",
            emitted,
            total_ms,
            ttft_ms if ttft_ms is not None else -1,
        )

    async def _run_call(self, call: dict[str, Any]) -> Message:
        """Run one call and return its result; failures become text for the model."""
        name = call["function"]["name"]
        arguments = _parse_arguments(name, call["function"]["arguments"])
        if isinstance(arguments, str):
            return _tool_message(call, arguments)
        try:
            content = await self.tools.invoke(name, arguments)
        except Exception as exc:  # noqa: BLE001 - reported to the model, like any tool failure
            logger.exception("tool call %s failed", name)
            content = f"ERROR: tool '{name}' failed: {type(exc).__name__}: {exc}"
        return _tool_message(call, _clip(content, self.config.agent.workspace.max_output_chars))

    async def _gather(
        self,
        messages: Sequence[Message],
        extra_context: str | None,
        transcript: list[Message] | None,
        overrides: dict,
    ) -> ChatResult:
        started = time.perf_counter()
        first_at: float | None = None
        text: list[str] = []
        reasoning: list[str] = []
        finish: FinishReason = "stop"
        usage = Usage()
        async for chunk in self.stream(
            messages, extra_context=extra_context, transcript=transcript, **overrides
        ):
            if (chunk.delta or chunk.reasoning) and first_at is None:
                first_at = time.perf_counter()
            text.append(chunk.delta)
            reasoning.append(chunk.reasoning)
            finish = chunk.finish_reason or finish
            usage = chunk.usage or usage
        thought = "".join(reasoning)
        return ChatResult(
            model=self.backend.config.model,
            message=Message.assistant("".join(text), reasoning=thought or None),
            finish_reason=finish,
            usage=usage,
            latency_ms=(time.perf_counter() - started) * 1000,
            time_to_first_token_ms=(first_at - started) * 1000 if first_at is not None else None,
        )

    # -- introspection ------------------------------------------------------

    def describe(self) -> dict:
        """Snapshot of the agent's effective configuration, shown in the UI."""
        return {
            "backend": self.backend.info().model_dump(),
            "memory_policy": self.memory.name,
            "max_history_tokens": self.config.agent.max_history_tokens,
            "tools": list(self.tools.tools),
            "expertise": self.config.agent.expertise,
            "knowledge_topics": [pack.topic for pack in self.knowledge],
            "generation": self.config.generation.model_dump(),
            "thinking_mode": self.config.thinking_mode.model_dump(exclude_none=True),
        }


_NOT_RUN = (
    "NOT RUN: this request already used its {rounds} (agent.max_tool_iterations), "
    "so nothing was done. The user can reply 'continue' to allow more."
)


def _rounds(n: int) -> str:
    return f"{n} tool round{'' if n == 1 else 's'}"


_MIN_RESULT_CHARS = 400


def _keep_well_formed(
    prepared: list[Message], full: Sequence[Message], budget: int, count_tokens: TokenCounter
) -> list[Message]:
    """Repair the ways trimming old history can break a tool conversation.

    Memory policies keep the newest messages, so the cut can fall between a
    call and its results: results that answer no call are dropped, since
    servers reject them. In a long tool loop the cut can drop the request
    itself, leaving only tool traffic; the latest user message is put back, as
    the model must see what it is working on (and Qwen's template requires it).
    When the results of the latest round alone outgrow the budget, the cut
    keeps nothing but some of them; that round is put back whole after the
    request. Whatever is put back, the tool results are then shortened (start
    and end kept) until the prompt fits the budget, or they are down to
    `_MIN_RESULT_CHARS`.
    """
    start = 1 if prepared and prepared[0].role == Role.SYSTEM else 0
    head, body = prepared[:start], list(prepared[start:])
    while body and body[0].role == Role.TOOL:
        body.pop(0)
    request = next((m for m in reversed(full) if m.role == Role.USER), None)
    if request is None or any(m.role == Role.USER for m in body):
        return [*head, *body]
    body = [request, *(body or _last_round(full))]
    kept = [*head, *body]
    limit = max((len(m.content) for m in body if m.role == Role.TOOL), default=0)
    while limit > _MIN_RESULT_CHARS and count_tokens(kept) > budget:
        limit = max(limit * 2 // 3, _MIN_RESULT_CHARS)
        kept = [*head, *(
            m.model_copy(update={"content": _clip(m.content, limit)}) if m.role == Role.TOOL else m
            for m in body
        )]  # fmt: skip
    return kept


def _last_round(full: Sequence[Message]) -> list[Message]:
    """The last message that called tools and its results, if `full` ends with them."""
    start = len(full)
    while start and full[start - 1].role == Role.TOOL:
        start -= 1
    if start == len(full) or not start or full[start - 1].role != Role.ASSISTANT:
        return []
    return list(full[start - 1 :]) if full[start - 1].tool_calls else []


def _normalize_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Calls in the exact OpenAI shape: a unique id, a name, arguments as a JSON string."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for call in calls:
        function = call.get("function") or {}
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps({} if arguments is None else arguments, default=str)
        call_id = str(call.get("id") or "")
        while not call_id or call_id in seen:
            call_id = f"call_{uuid.uuid4().hex[:12]}"
        seen.add(call_id)
        out.append({
            "id": call_id,
            "type": "function",
            "function": {"name": str(function.get("name") or ""), "arguments": arguments},
        })  # fmt: skip
    return out


def _parse_arguments(name: str, raw: str) -> dict[str, Any] | str:
    """The call's arguments as a dict, or the ERROR text the model reads instead."""
    if not raw.strip():
        return {}
    try:
        value = json.loads(raw)
        if isinstance(value, str):  # some models encode the object twice
            value = json.loads(value)
    except json.JSONDecodeError as exc:
        return (
            f"ERROR: the arguments for {name} are not valid JSON ({exc.msg}, at character "
            f"{exc.pos}). Nothing was run. Send them again as one JSON object."
        )
    if not isinstance(value, dict):
        return f"ERROR: the arguments for {name} must be a JSON object ({{...}}). Nothing was run."
    return value


def _tool_message(call: dict[str, Any], content: str) -> Message:
    return Message(
        role=Role.TOOL, content=content, tool_call_id=call["id"], name=call["function"]["name"]
    )


def _clip(text: str, limit: int) -> str:
    """Bound a tool result, keeping its start and its end, where errors usually are."""
    if len(text) <= limit:
        return text
    head, tail = limit * 2 // 5, limit * 3 // 5
    return f"{text[:head]}\n… {len(text) - head - tail:,} characters omitted …\n{text[-tail:]}"


def _add_usage(total: Usage | None, more: Usage) -> Usage:
    if total is None:
        return more
    return Usage(
        prompt_tokens=total.prompt_tokens + more.prompt_tokens,
        completion_tokens=total.completion_tokens + more.completion_tokens,
    )
