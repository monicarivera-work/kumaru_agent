"""Text from outside Kumaru, made safe for the model to read.

Files in the project folder, and the output of the programs `run_command`
starts, are untrusted: whoever could write them could try to steer the model
(prompt injection). Your approval of every change and command is the backstop;
this module takes away one lever on top of it. Knowledge packs and documentation
excerpts are untrusted reference data, and are escaped the same way.

Chat models mark the turns of a conversation with control tokens, such as
Qwen's ``<|im_start|>`` and ``<|im_end|>``. llama-server, like the transformers
backend, turns the rendered prompt into tokens with these recognised anywhere in
it, so a file holding ``<|im_end|>`` and then ``<|im_start|>system`` would open
a real system turn. `neutralize_control_tokens` escapes such text
(``&lt;|im_end|&gt;``): the model can still read it, but it no longer becomes a
control token. The change is visible, not a zero-width character, so code the
model copies from such text cannot carry a hidden character into a program.
`restore_control_tokens` turns the escaped forms back, for the one place that
needs the real characters: `edit_file`, when the model copies text it read and
the file holds the real tokens.

It imports nothing from Kumaru, so any module can use it.
"""

from __future__ import annotations

import re

CONTROL_TOKEN = re.compile(
    r"<\|[^\s|<>&]{1,64}\|>|<\uff5c[^\s\uff5c<>&]{1,64}\uff5c>"
    r"|</?(?:s|bos|eos|start_of_turn|end_of_turn|think|tool_call|tool_response)>"
    r"|<</?SYS>>"
    r"|\[/?(?:INST|SYSTEM_PROMPT|AVAILABLE_TOOLS|TOOL_RESULTS|TOOL_CALLS|THINK)\]"
)
MAX_CONTROL_TOKEN_CHARS = 68  # "<|", a 64-character name and "|>"
_ENTITIES = str.maketrans({"<": "&lt;", ">": "&gt;", "[": "&#91;", "]": "&#93;"})

# What `neutralize_control_tokens` turns each kind of CONTROL_TOKEN into, in the
# same order. A name between bars may hold [ and ], which are escaped as well.
_ESCAPED_TOKEN = re.compile(
    r"&lt;\|(?:[^\s|<>&\[\]]|&#91;|&#93;){1,64}\|&gt;"
    r"|&lt;\uff5c(?:[^\s\uff5c<>&\[\]]|&#91;|&#93;){1,64}\uff5c&gt;"
    r"|&lt;/?(?:s|bos|eos|start_of_turn|end_of_turn|think|tool_call|tool_response)&gt;"
    r"|&lt;&lt;/?SYS&gt;&gt;"
    r"|&#91;/?(?:INST|SYSTEM_PROMPT|AVAILABLE_TOOLS|TOOL_RESULTS|TOOL_CALLS|THINK)&#93;"
)
_ENTITY = re.compile(r"&lt;|&gt;|&#91;|&#93;")
_REAL = {"&lt;": "<", "&gt;": ">", "&#91;": "[", "&#93;": "]"}


def neutralize_control_tokens(text: str) -> str:
    """`text` with every chat control token escaped, and nothing else changed.

    Only the ``<``, ``>``, ``[`` and ``]`` of a token are replaced, by ``&lt;``,
    ``&gt;``, ``&#91;`` and ``&#93;``; the result holds no control token, so
    escaping it again changes nothing.
    """
    return CONTROL_TOKEN.sub(lambda match: match.group().translate(_ENTITIES), text)


def restore_control_tokens(text: str) -> str:
    """`text` with the escaped control tokens turned back into the real ones.

    This undoes `neutralize_control_tokens` for text that had no ``&`` in it.
    Text that writes ``&lt;s&gt;`` on purpose, as HTML may, looks exactly like
    escaped text, so callers try text as it is before they restore it.
    """
    return _ESCAPED_TOKEN.sub(
        lambda match: _ENTITY.sub(lambda entity: _REAL[entity.group()], match.group()), text
    )
