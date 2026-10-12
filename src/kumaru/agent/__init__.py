"""Agent layer: everything between "a user typed something" and "call the model".

Responsibilities kept here on purpose:

* assembling the system prompt (`prompt.py`)
* fitting the conversation into the context window (`memory.py`)
* the tool-calling seam for future capabilities (`tools.py`)
* orchestrating the whole turn (`chat.py`)

Keeping these out of the server means the exact same agent runs unchanged from
the CLI, from a test, or from a future batch-evaluation harness.
"""

from kumaru.agent.chat import ChatAgent

__all__ = ["ChatAgent"]
