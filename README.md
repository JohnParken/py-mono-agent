# pi-ai and pi-agent-core: Python AI Agent Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Python 复刻版** [badlogic/pi-mono](https://github.com/badlogic/pi-mono) - 一个简洁而强大的 AI Agent 框架。

这个项目将 Mario Zechner 的 TypeScript AI Agent 框架移植到 Python，提供统一的 LLM API 抽象层和有状态/无状态的 Agent 实现。

## 项目结构

```
pi_ai/          # 统一的 LLM API 抽象层
  ├── event_stream.py    # 事件流系统
  ├── llm.py             # LLM 模型抽象
  ├── config.py          # 配置管理
  ├── model_router.py    # 模型路由
  ├── types.py           # 核心类型定义
  └── exceptions.py      # 异常处理

pi_agent_core/  # Agent 核心实现
  ├── agent.py           # 有状态 Agent
  ├── agent_loop.py      # 无状态 Agent Loop
  ├── types.py           # Agent 类型定义
  └── skills.py          # Skill 管理系统
```

## 灵感来源

本项目受到 [badlogic/pi-mono](https://github.com/badlogic/pi-mono) 的深刻启发，特别是：

- `@mariozechner/pi-ai` - TypeScript LLM 抽象层
- `@mariozechner/pi-agent-core` - TypeScript Agent 框架

原项目的优雅设计和清晰的架构理念是我进行 Python 移植的动力。我保持了核心设计思想，同时充分利用 Python 的异步特性和类型提示。

## 安装

```bash
# 从 GitHub 安装
pip install git+https://github.com/encyc/py-mono.git

# 或本地开发模式
git clone https://github.com/encyc/py-mono.git
cd py-mono
pip install -e .
```

### 依赖

- Python 3.10+
- pydantic 2.0+
- httpx (异步 HTTP 客户端)
- python-dotenv

## 快速开始

### 1. 使用 pi-ai - 统一的 LLM API

`pi-ai` 提供了统一的接口来调用多种 LLM Provider：

```python
from pi_ai import Model, stream_simple, get_model
from pi_ai.types import UserMessage, TextContent

# 创建模型实例
model = get_model(
    provider="openai",  # 或 "anthropic", "google"
    model_id="gpt-4o",
    api_key="your-api-key"
)

# 发送消息并流式接收响应
async def chat():
    message = UserMessage(content=[TextContent(text="Hello, world!")])
  
    async for event in stream_simple(model, message):
        if hasattr(event, 'content'):
            print(event.content.text, end='', flush=True)
```

### 2. 使用 Agent - 有状态的对话

`pi-agent-core` 提供了有状态的 Agent 实现：

```python
from pi_agent_core import Agent
from pi_ai import get_model

# 创建 Agent
agent = Agent(
    system_prompt="You are a helpful assistant.",
    model=get_model("openai", "gpt-4o", api_key="your-key")
)

# 订阅事件
agent.subscribe(lambda event: print(f"Event: {event.type}"))

# 运行对话
async def main():
    response = await agent.run("What is the capital of France?")
    print(response)

# 保持状态继续对话
response2 = await agent.run("And what about Germany?")
```

### 3. 使用 agent_loop - 无状态的函数式调用

对于不需要保持状态的场景，使用无状态的 `agent_loop`：

```python
from pi_agent_core import agent_loop
from pi_agent_core.types import AgentLoopConfig
from pi_ai import get_model

async def chat():
    config = AgentLoopConfig(
        model=get_model("openai", "gpt-4o"),
        system_prompt="You are a helpful assistant."
    )
  
    messages = [{"role": "user", "content": "Hello!"}]
  
    async for event in agent_loop(messages, config):
        if event.type == "message_update":
            print(event.message, end='', flush=True)
```

### 4. 添加工具 (Tools)

Agent 可以调用外部工具：

```python
from pydantic import BaseModel
from pi_agent_core import Agent, AgentTool

class GetWeatherArgs(BaseModel):
    city: str

async def get_weather(city: str, tool_call_id, context, update_callback):
    # 调用天气 API
    return AgentToolResult(
        content=[TextContent(text=f"Weather in {city}: Sunny")]
    )

agent = Agent(
    system_prompt="You are a helpful assistant.",
    model=get_model("openai", "gpt-4o"),
    tools=[
        AgentTool(
            name="get_weather",
            label="Get Weather",
            description="Get current weather for a city",
            parameters=GetWeatherArgs,
            execute=get_weather
        )
    ]
)
```

### 5. 使用配置文件

通过 YAML 配置文件管理多个 LLM 配置：

```yaml
# llm.yaml
use_llm: glm_4_7

llms:
  glm_4_7:
    provider: openai
    api_key: ${ZHIPU_API_KEY:}
    base_url: https://open.bigmodel.cn/api/paas/v4
    model: glm-4-plus

  openai_gpt4o:
    provider: openai
    api_key: ${OPENAI_API_KEY:}
    model: gpt-4o
```

```python
from pi_ai import get_model_from_config

model = get_model_from_config()
```

## 事件系统

框架提供了丰富的事件流，可以监控 Agent 的整个生命周期：

| 事件类型                  | 描述             |
| ------------------------- | ---------------- |
| `agent_start`           | Agent 开始运行   |
| `turn_start`            | 新的对话轮次开始 |
| `message_start`         | 新消息开始生成   |
| `message_update`        | 消息流式更新     |
| `tool_execution_start`  | 工具开始执行     |
| `tool_execution_update` | 工具执行进度更新 |
| `turn_end`              | 对话轮次结束     |
| `agent_end`             | Agent 运行结束   |

```python
agent.subscribe(lambda event: {
    "agent_start": lambda: print("Agent started"),
    "message_update": lambda e: print(f"Streaming: {e.message}"),
    "tool_execution_end": lambda e: print(f"Tool {e.tool_name} finished")
}.get(event.type, lambda: None)())
```

## 支持的 LLM Provider

- **OpenAI** - GPT-4o, GPT-4o-mini, etc.
- **Anthropic** - Claude Sonnet 4, Opus 4, Haiku 4
- **Google** - Gemini Pro, Gemini Flash
- **自定义** - 实现自己的 Provider 适配器

## 高级功能

### 思考模式 (Thinking)

```python
from pi_ai.types import ThinkingLevel

agent = Agent(
    system_prompt="You are a helpful assistant.",
    model=get_model("anthropic", "claude-sonnet-4-20250514"),
    thinking_level=ThinkingLevel.HIGH  # 启用深度思考
)
```

### 模型路由 (Model Router)

根据任务复杂度自动选择模型：

```python
from pi_ai import ModelRouter, create_model_router_from_config

router = create_model_router_from_config()
model = router.get_model_for_intent("complex_analysis")
```

### 上下文转换

```python
async def transform_context(messages, metadata):
    # 裁剪上下文、注入 RAG 检索内容等
    return messages[:10] + [retrieved_context]

agent = Agent(
    system_prompt="You are a helpful assistant.",
    model=get_model("openai", "gpt-4o"),
    transform_context=transform_context
)
```

## 与原版的区别

| 特性     | TypeScript 原版 | Python 版本       |
| -------- | --------------- | ----------------- |
| 异步模型 | Promise         | asyncio           |
| 类型系统 | TypeScript      | typing + Pydantic |
| 流式处理 | ReadableStream  | AsyncGenerator    |
| 配置管理 | JSON            | YAML + 环境变量   |
| 参数验证 | TypeScript 类型 | Pydantic 模型     |

## 许可证

MIT License - 与原项目保持一致

## 致谢

- [Mario Zechner](https://github.com/badlogic) - 原始 pi-mono 框架作者
- [badlogic/pi-mono](https://github.com/badlogic/pi-mono) - TypeScript 原版项目

## 相关链接

- 原项目: https://github.com/badlogic/pi-mono
- 问题反馈: https://github.com/encyc/py-mono/issues
