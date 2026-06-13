"""
examples/agent_with_tools.py
-----------------------------
A Kumaru agent equipped with the built-in Calculator and WebSearch tools.

Run:
    ollama pull llama3.1
    ollama serve
    python examples/agent_with_tools.py

What this demonstrates:
  * Registering tools with the agent.
  * The full ReAct loop: the model calls a tool, receives the result, and
    incorporates it into a final answer.
  * How to inspect the conversation memory after a run.
"""

from kumaru.agent import Agent
from kumaru.config import AgentConfig
from kumaru.tools import CalculatorTool, WebSearchTool

# ── 1. Create tools ───────────────────────────────────────────────────────────
#
# Pass your Serper/Brave/SerpAPI key here for live results.
# Without a key the WebSearchTool returns labelled stub data.
calculator = CalculatorTool()
web_search = WebSearchTool(api_key=None)  # stub mode

# ── 2. Create the agent ───────────────────────────────────────────────────────
config = AgentConfig(verbose=True, max_iterations=6)
agent = Agent(config=config, tools=[calculator, web_search])

print("=== Agent with tools ===\n")

# ── 3. Arithmetic query (uses calculator tool) ────────────────────────────────
query1 = "If I invest $1,000 at 8% annual interest for 5 years, what is the final amount? Use the calculator."
print(f"User: {query1}\n")
reply1 = agent.run(query1)
print(f"Kumaru: {reply1}\n")
print("-" * 60)

# ── 4. Reset and do a web-search query ───────────────────────────────────────
agent.reset()  # start a fresh conversation

query2 = "What is the latest news about AI agents?"
print(f"\nUser: {query2}\n")
reply2 = agent.run(query2)
print(f"Kumaru: {reply2}\n")
print("-" * 60)

# ── 5. Inspect the conversation memory ───────────────────────────────────────
print("\n=== Conversation memory dump ===")
for i, msg in enumerate(agent.memory.get_messages()):
    role = msg.role.upper()
    content_preview = (msg.content or "")[:80]
    print(f"[{i}] {role}: {content_preview}")
