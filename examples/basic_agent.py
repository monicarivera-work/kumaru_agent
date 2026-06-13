"""
examples/basic_agent.py
------------------------
The simplest possible Kumaru agent.

Run:
    ollama pull llama3.1
    ollama serve
    python examples/basic_agent.py

What this demonstrates:
  * Creating an Agent with default settings.
  * Sending a single message and printing the reply.
  * How the ReAct loop works when no tools are needed.
"""

from kumaru.agent import Agent
from kumaru.config import AgentConfig

# ── 1. Configure the agent ────────────────────────────────────────────────────
#
# AgentConfig bundles all the settings in one place.
# verbose=True prints the agent's reasoning to stdout for learning purposes.
config = AgentConfig(verbose=True)

# ── 2. Create the agent ───────────────────────────────────────────────────────
#
# The Agent constructor wires together:
#   * The LLM client (OllamaClient by default)
#   * The conversation memory
#   * The tool registry
agent = Agent(config=config)

# ── 3. Run a query ────────────────────────────────────────────────────────────
query = "Explain what a large language model is in two sentences."
print(f"\nUser: {query}\n")
reply = agent.run(query)
print(f"Kumaru: {reply}\n")

# ── 4. Continue the conversation (multi-turn) ─────────────────────────────────
#
# The agent remembers the previous exchange.  You can keep calling run()
# with new messages and it will maintain context.
follow_up = "What are three practical applications of LLMs?"
print(f"User: {follow_up}\n")
reply2 = agent.run(follow_up)
print(f"Kumaru: {reply2}\n")
