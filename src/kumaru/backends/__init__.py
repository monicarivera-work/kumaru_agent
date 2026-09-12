"""Inference backends: the one seam that makes model scale a config change.

Every engine — a 1.5B model on your CPU, a 7B on your 3090, a 70B sharded across
a node, or a remote vLLM cluster — is hidden behind the same four methods. The
agent, server, and UI above this line never learn which one is loaded.

That is the whole scalability story for inference in one sentence: *upgrading
your compute is editing a YAML file, not rewriting your application.*
"""

from kumaru.backends.base import Backend, iterate_in_thread

__all__ = ["Backend", "iterate_in_thread"]
