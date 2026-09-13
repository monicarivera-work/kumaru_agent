"""Prompt assembly - the one place that decides what the model actually sees.

Keeping this separate from :mod:`kumaru.agent.chat` matters because prompt
changes are the highest-leverage, highest-risk edits in the whole system. When
answers get worse, you want exactly one file to diff.

Rules enforced here:

* there is always exactly one system message, and it comes first;
* history is appended in order, with any stray system messages dropped;
* the new user turn is always last.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from kumaru.core.types import Message, Role


def render_system_prompt(template: str, *, now: datetime | None = None) -> str:
    """Fill ``{date}``/``{time}`` placeholders in a system prompt.

    Models have no clock; giving them the date prevents confident nonsense
    about "today". Unknown placeholders are left untouched rather than raising,
    so a prompt containing literal braces (JSON examples) still works.
    """
    stamp = now or datetime.now()
    return template.replace("{date}", stamp.strftime("%Y-%m-%d")).replace(
        "{time}", stamp.strftime("%H:%M")
    )


def build_messages(
    system_prompt: str,
    history: Iterable[Message],
    user_input: str,
    *,
    now: datetime | None = None,
) -> list[Message]:
    """Return the final message list to send to a backend."""
    messages: list[Message] = []
    rendered = render_system_prompt(system_prompt, now=now).strip()
    if rendered:
        messages.append(Message.system(rendered))
    messages.extend(m for m in history if m.role is not Role.SYSTEM)
    if user_input.strip():
        messages.append(Message.user(user_input.strip()))
    return messages
