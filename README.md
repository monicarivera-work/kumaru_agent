# Kumaru Agent

An enterprise LLM agent built from scratch — for learning and production use.

> **Goal:** Understand *how* an AI agent works by building every layer yourself,
> from the LLM API call all the way up to the reasoning loop that decides what
> to do next.

---

## Table of Contents

1. [What is an LLM Agent?](#what-is-an-llm-agent)
2. [Architecture overview](#architecture-overview)
3. [Key concepts](#key-concepts)
4. [Project structure](#project-structure)
5. [Quick start](#quick-start)
6. [Running the examples](#running-the-examples)
7. [Running the tests](#running-the-tests)
8. [How to extend the agent](#how-to-extend-the-agent)
9. [Enterprise patterns used](#enterprise-patterns-used)

---

## What is an LLM Agent?

A **Large Language Model (LLM)** is a neural network trained on vast amounts
of text.  It takes a sequence of tokens as input and predicts the next token.
Repeat that prediction many times and you get coherent prose, code, or plans.

An **LLM Agent** wraps that model inside a loop:

```
User message
     │
     ▼
┌────────────┐    tool call    ┌──────────┐
│  LLM call  │───────────────▶│   Tool   │
│ (reason)   │◀──── result ───│ (act)    │
└────────────┘                └──────────┘
     │ final answer
     ▼
  Response
```

The key insight: **the model decides what to do; Python does it**.
The model never executes code directly.  It asks for a tool (e.g. "call the
calculator with `6 * 7`"), and the agent framework runs the tool and sends the
result back.

---

## Architecture overview

```
kumaru/
├── agent.py          ← The ReAct loop (reason → act → observe → repeat)
├── config.py         ← All settings in one place (model, temperature, …)
├── logger.py         ← Structured JSON logging for production
├── llm/
│   ├── base.py       ← Abstract interface (swap providers without code changes)
│   └── openai_client.py ← OpenAI implementation with retry logic
├── memory/
│   └── conversation.py  ← Rolling list of messages = agent's working memory
└── tools/
    ├── base.py       ← Abstract tool interface + ToolError
    ├── calculator.py ← Safe arithmetic (no eval!)
    └── web_search.py ← Web search (stub + live Serper implementation)
```

---

## Key concepts

### 1 · Messages and roles

LLM APIs communicate via a list of **messages**.  Each message has a `role`:

| Role | Who sends it | Purpose |
|------|-------------|---------|
| `system` | You | Sets the model's persona and rules |
| `user` | The human (or agent) | The question or instruction |
| `assistant` | The model | The model's response or tool request |
| `tool` | Your code | The result of running a tool |

Keeping this list *is* the agent's short-term memory.

### 2 · The ReAct pattern

**Re**asoning + **Act**ing.  The agent loop:

1. **Reason** — send all messages to the LLM, get a response.
2. **Act** — if the response contains a tool call, run the tool.
3. **Observe** — append the tool result as a `tool` role message.
4. **Repeat** — until the model gives a plain text answer (no tool call).

### 3 · Tool / function calling

When you register a tool with the agent, its JSON Schema description is sent
to the model on every call.  The model can then "call" a tool by returning a
structured `tool_calls` response instead of plain text.

### 4 · Conversation memory

Every message — user, assistant, tool call, tool result — is stored in
`ConversationMemory` and replayed on every LLM call.  This is why the model
can "remember" previous turns.  In production you'd persist this to a database.

### 5 · LLM abstraction

`BaseLLMClient` is an interface.  `OpenAIClient` is one implementation.  To
swap to Anthropic, Cohere, or a local Ollama model you create a new subclass
and pass it to the `Agent` constructor — no other code changes needed.

---

## Project structure

```
kumaru_agent/
├── kumaru/               ← Main package
│   ├── agent.py
│   ├── config.py
│   ├── logger.py
│   ├── llm/
│   ├── memory/
│   └── tools/
├── tests/                ← Pytest test suite
├── examples/             ← Runnable demo scripts
├── requirements.txt
└── setup.py
```

---

## Quick start

```bash
# 1. Clone the repo
git clone https://github.com/monicarivera-work/kumaru_agent.git
cd kumaru_agent

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -e .             # installs the kumaru package in editable mode

# 4. Set your OpenAI API key
export OPENAI_API_KEY="sk-..."   # or add it to a .env file
```

Minimal usage:

```python
from kumaru.agent import Agent
from kumaru.tools import CalculatorTool

agent = Agent(tools=[CalculatorTool()])
reply = agent.run("What is 123 * 456?")
print(reply)  # → "123 * 456 = 56088"
```

---

## Running the examples

```bash
# Bare-bones agent (no tools)
python examples/basic_agent.py

# Agent with calculator and web-search tools
python examples/agent_with_tools.py
```

---

## Running the tests

The test suite uses mocked LLM responses so **no API key is required**.

```bash
pip install pytest pytest-mock
pytest tests/ -v
```

---

## How to extend the agent

### Add a new tool

```python
# my_tools/datetime_tool.py
from datetime import datetime
from kumaru.tools.base import BaseTool

class DateTimeTool(BaseTool):
    name = "get_current_datetime"
    description = "Returns the current date and time."
    parameters = {"type": "object", "properties": {}, "required": []}

    def run(self, **_):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

```python
from kumaru.agent import Agent
from my_tools.datetime_tool import DateTimeTool

agent = Agent(tools=[DateTimeTool()])
print(agent.run("What time is it?"))
```

### Add a new LLM provider

```python
from kumaru.llm.base import BaseLLMClient, LLMResponse, Message

class AnthropicClient(BaseLLMClient):
    def chat(self, messages, tools=None):
        # ... call Anthropic API ...
        return LLMResponse(content="...", tool_calls=[], usage={})

agent = Agent(llm=AnthropicClient())
```

---

## Enterprise patterns used

| Pattern | Where | Why |
|---------|-------|-----|
| Dependency Inversion | `BaseLLMClient` interface | Swap providers without touching agent code |
| Strategy pattern | `BaseTool` interface | Add new capabilities without modifying the agent |
| Retry with exponential back-off | `OpenAIClient._call_with_retry` | Handles transient API failures gracefully |
| Structured logging (JSON) | `kumaru/logger.py` | Searchable logs in production (Datadog, CloudWatch) |
| Safety limits | `AgentConfig.max_iterations` | Prevents runaway loops that waste money |
| Secrets via env vars | `LLMConfig.api_key` | Never hard-code credentials |
| Separation of concerns | `agent / llm / memory / tools` | Each module has one job; easy to test and replace |
