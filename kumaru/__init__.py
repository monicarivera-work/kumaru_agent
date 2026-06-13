"""
kumaru – An enterprise LLM agent built from scratch.

Quick start::

    from kumaru.agent import Agent
    from kumaru.tools import CalculatorTool

    agent = Agent(tools=[CalculatorTool()])
    print(agent.run("What is 42 * 137?"))
"""

from kumaru.agent import Agent
from kumaru.config import AgentConfig, LLMConfig

__all__ = ["Agent", "AgentConfig", "LLMConfig"]
