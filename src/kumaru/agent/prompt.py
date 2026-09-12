"""System-prompt assembly.

Why a module rather than a string constant
------------------------------------------
The system prompt is the highest-leverage, lowest-cost lever you have over
answer quality — it is the one part of model behaviour you can change in
milliseconds instead of hours of fine-tuning. Isolating it here means you can
A/B test prompts, version them, and inject runtime facts (today's date, enabled
tools) without touching the orchestration code.

The order of the assembled sections is deliberate. Models attend most reliably
to the beginning and the end of their context (the "lost in the middle" effect),
so stable identity goes first and volatile, task-specific context goes last —
closest to the user's question.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Sequence

from kumaru.core.config import AgentConfig
from kumaru.core.types import Message, Role

# Appended to every system prompt. These are behavioural guardrails that apply
# regardless of which persona the user configures.
QUALITY_DIRECTIVES = """\
Operating rules:
- Lead with the direct answer. Put reasoning after it, not before.
- Match answer length to question complexity. Do not pad.
- If you do not know, say "I don't know" and name what would settle it.
- Never fabricate citations, URLs, APIs, file paths, version numbers, or statistics.
- When a question is ambiguous in a way that changes the answer, ask one clarifying question.
- Use fenced code blocks with a language tag for all code."""


def build_system_prompt(
    config: AgentConfig,
    *,
    tool_descriptions: Sequence[str] = (),
    extra_context: str | None = None,
    include_date: bool = True,
) -> str:
    """Compose the full system prompt from its parts."""
    sections: list[str] = [config.system_prompt.strip(), QUALITY_DIRECTIVES]

    if include_date:
        today = _dt.datetime.now().strftime("%Y-%m-%d")
        # Models have a training cutoff and will otherwise confidently assume a
        # stale date. One line removes a whole class of wrong answers.
        sections.append(f"Today's date is {today}.")

    if tool_descriptions:
        listed = "\n".join(f"- {d}" for d in tool_descriptions)
        sections.append(f"You can call these tools:\n{listed}")

    if extra_context:
        # Retrieved documents land here — last, so they sit nearest the question.
        sections.append(f"Relevant context:\n{extra_context.strip()}")

    return "\n\n".join(s for s in sections if s)


def ensure_system_message(
    messages: Sequence[Message], system_prompt: str
) -> list[Message]:
    """Return `messages` with exactly one system message, at index 0.

    Normalizing here means every downstream component can assume the invariant,
    and a client that forgets to send a system message still gets correct
    behaviour instead of a bare, unguided model.
    """
    body = [m for m in messages if m.role != Role.SYSTEM]
    client_system = next((m for m in messages if m.role == Role.SYSTEM), None)

    if client_system and client_system.content.strip():
        # A client-supplied system message augments the configured one rather
        # than silently replacing it, so server-side guardrails always apply.
        merged = f"{system_prompt}\n\n{client_system.content.strip()}"
    else:
        merged = system_prompt

    return [Message.system(merged), *body]
