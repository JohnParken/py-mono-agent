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

macOS / Linux:

```bash
# 从 GitHub 安装
pip install git+https://github.com/encyc/py-mono.git

# 或本地开发模式
git clone https://github.com/encyc/py-mono.git
cd py-mono
pip install -e .
```

Windows PowerShell:

```powershell
# 从 GitHub 安装
pip install git+https://github.com/encyc/py-mono.git

# 或本地开发模式
git clone https://github.com/encyc/py-mono.git
cd py-mono
pip install -e .
```

Windows `cmd.exe`:

```bat
REM 从 GitHub 安装
pip install git+https://github.com/encyc/py-mono.git

REM 或本地开发模式
git clone https://github.com/encyc/py-mono.git
cd py-mono
pip install -e .
```

### 使用 uv 开发

本项目已经配置好 `uv`，推荐把它作为默认的 Python 环境和依赖管理工具。

#### 1. 初始化开发环境

macOS / Linux:

```bash
git clone https://github.com/encyc/py-mono.git
cd py-mono

# 根据 pyproject.toml 和 uv.lock 创建虚拟环境并同步依赖
uv sync
```

如果你需要使用代理流相关功能，可以安装额外依赖：

```bash
uv sync --extra proxy
```

Windows PowerShell:

```powershell
git clone https://github.com/encyc/py-mono.git
cd py-mono

# 根据 pyproject.toml 和 uv.lock 创建虚拟环境并同步依赖
uv sync

# 如果需要代理流功能
uv sync --extra proxy
```

Windows `cmd.exe`:

```bat
git clone https://github.com/encyc/py-mono.git
cd py-mono

REM 根据 pyproject.toml 和 uv.lock 创建虚拟环境并同步依赖
uv sync

REM 如果需要代理流功能
uv sync --extra proxy
```

执行完成后，项目根目录会生成 `.venv/`，后续所有开发命令都建议通过 `uv run` 执行。

#### 2. 日常写代码的常用命令

macOS / Linux:

```bash
# 运行一个 Python 脚本
uv run python your_script.py

# 直接运行一小段调试代码
uv run python -c "import pi_ai; print(pi_ai.__version__)"

# 进入项目虚拟环境的 Python 交互式解释器
uv run python
```

Windows PowerShell:

```powershell
# 运行一个 Python 脚本
uv run python your_script.py

# 直接运行一小段调试代码
uv run python -c "import pi_ai; print(pi_ai.__version__)"

# 进入项目虚拟环境的 Python 交互式解释器
uv run python
```

Windows `cmd.exe`:

```bat
REM 运行一个 Python 脚本
uv run python your_script.py

REM 直接运行一小段调试代码
uv run python -c "import pi_ai; print(pi_ai.__version__)"

REM 进入项目虚拟环境的 Python 交互式解释器
uv run python
```

如果你习惯先激活虚拟环境，也可以执行。

macOS / Linux:

```bash
source .venv/bin/activate
python your_script.py
```

不过对这个项目来说，更推荐 `uv run ...`，这样不容易混用系统 Python。

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python your_script.py
```

Windows `cmd.exe`:

```bat
.venv\Scripts\activate.bat
python your_script.py
```

#### 3. 新增依赖

开发过程中如果需要安装新包，推荐直接用 `uv add`，它会自动更新 `pyproject.toml` 和 `uv.lock`：

macOS / Linux:

```bash
# 添加运行时依赖
uv add requests

# 添加可选依赖到 proxy 组
uv add --optional proxy aiohttp
```

Windows PowerShell:

```powershell
# 添加运行时依赖
uv add requests

# 添加可选依赖到 proxy 组
uv add --optional proxy aiohttp
```

Windows `cmd.exe`:

```bat
REM 添加运行时依赖
uv add requests

REM 添加可选依赖到 proxy 组
uv add --optional proxy aiohttp
```

如果你只是修改了 `pyproject.toml`，也可以重新同步：

```bash
uv sync
```

#### 4. 运行项目代码

仓库当前主要提供两个包：`pi_ai` 和 `pi_agent_core`。开发时可以直接这样验证导入是否正常：

macOS / Linux:

```bash
uv run python -c "import pi_ai, pi_agent_core; print(pi_ai.__version__, pi_agent_core.__version__)"
```

如果你想新建一个本地实验脚本，例如 `examples/demo.py`，可以直接：

```bash
uv run python examples/demo.py
```

Windows PowerShell:

```powershell
uv run python -c "import pi_ai, pi_agent_core; print(pi_ai.__version__, pi_agent_core.__version__)"
uv run python examples/demo.py
```

Windows `cmd.exe`:

```bat
uv run python -c "import pi_ai, pi_agent_core; print(pi_ai.__version__, pi_agent_core.__version__)"
uv run python examples/demo.py
```

#### 5. 更新锁文件

当依赖发生变更时，可以重新生成锁文件并同步环境：

macOS / Linux:

```bash
uv lock
uv sync
```

Windows PowerShell:

```powershell
uv lock
uv sync
```

Windows `cmd.exe`:

```bat
uv lock
uv sync
```

这样团队成员拉取代码后，只需要执行一次 `uv sync`，就能得到一致的开发环境。

### 作为代码助手运行

这个仓库现在带了一个最小可用的代码助手 CLI，参考了 `badlogic/pi-mono` 的基础闭环思路：让模型通过工具来读文件、搜代码、改文件和执行命令。

内置工具包括：

- `list_files`：列出工作区文件
- `read_file`：按行读取文件
- `search_code`：搜索代码内容
- `write_file`：写入文件
- `edit_file`：按精确文本替换内容
- `run_command`：在工作区内执行命令

#### 1. 准备模型配置

你可以直接复用 `pi_ai/llm.yaml.example` 作为模板，并通过环境变量提供 API Key。

如果使用 OpenAI：

```bash
export OPENAI_API_KEY=your-key
```

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your-key"
```

Windows `cmd.exe`:

```bat
set OPENAI_API_KEY=your-key
```

如果你想用 OpenRouter 来验证，可以新建 `pi_ai/llm.yaml`：

```yaml
use_llm: openrouter

llms:
  openrouter:
    provider: openai
    api_key: ${OPENROUTER_API_KEY:}
    base_url: https://openrouter.ai/api/v1
    model: openai/gpt-5.2
    description: "OpenRouter"
```

然后设置环境变量：

```bash
export OPENROUTER_API_KEY=your-openrouter-key
```

Windows PowerShell:

```powershell
$env:OPENROUTER_API_KEY="your-openrouter-key"
```

Windows `cmd.exe`:

```bat
set OPENROUTER_API_KEY=your-openrouter-key
```

如果你有自己的配置文件，也可以在启动时通过 `--config` 指定。

#### 2. 启动交互式代码助手

macOS / Linux:

```bash
uv run py-mono-agent --workspace .
```

Windows PowerShell:

```powershell
uv run py-mono-agent --workspace .
```

Windows `cmd.exe`:

```bat
uv run py-mono-agent --workspace .
```

启动后可以直接输入类似下面的请求：

```text
找出项目里和 model router 相关的代码，然后解释它是怎么工作的
```

#### 3. 单次执行一个任务

macOS / Linux:

```bash
uv run py-mono-agent --workspace . "读取 README，然后帮我总结当前项目结构"
```

Windows PowerShell:

```powershell
uv run py-mono-agent --workspace . "读取 README，然后帮我总结当前项目结构"
```

Windows `cmd.exe`:

```bat
uv run py-mono-agent --workspace . "读取 README，然后帮我总结当前项目结构"
```

#### 4. 指定模型

如果你不想走配置文件，也可以直接指定 provider 和 model。

直接指定 OpenAI：

```bash
uv run py-mono-agent \
  --workspace . \
  --provider openai \
  --model gpt-4o-mini \
  --api-key "$OPENAI_API_KEY" \
  "搜索项目里所有 AgentTool 的定义"
```

Windows PowerShell:

```powershell
uv run py-mono-agent `
  --workspace . `
  --provider openai `
  --model gpt-4o-mini `
  --api-key $env:OPENAI_API_KEY `
  "搜索项目里所有 AgentTool 的定义"
```

直接指定 OpenRouter：

```bash
uv run py-mono-agent \
  --workspace . \
  --provider openai \
  --model openai/gpt-5.2 \
  --api-key "$OPENROUTER_API_KEY" \
  --base-url https://openrouter.ai/api/v1 \
  "搜索项目里所有 AgentTool 的定义"
```

Windows PowerShell:

```powershell
uv run py-mono-agent `
  --workspace . `
  --provider openai `
  --model openai/gpt-5.2 `
  --api-key $env:OPENROUTER_API_KEY `
  --base-url https://openrouter.ai/api/v1 `
  "搜索项目里所有 AgentTool 的定义"
```

Windows `cmd.exe`:

```bat
uv run py-mono-agent ^
  --workspace . ^
  --provider openai ^
  --model openai/gpt-5.2 ^
  --api-key %OPENROUTER_API_KEY% ^
  --base-url https://openrouter.ai/api/v1 ^
  "搜索项目里所有 AgentTool 的定义"
```

如果你已经把 OpenRouter 写进了 `pi_ai/llm.yaml`，也可以直接：

macOS / Linux:

```bash
uv run py-mono-agent --workspace . --config pi_ai/llm.yaml
```

Windows PowerShell:

```powershell
uv run py-mono-agent --workspace . --config pi_ai/llm.yaml
```

Windows `cmd.exe`:

```bat
uv run py-mono-agent --workspace . --config pi_ai/llm.yaml
```

#### 5. 工作方式和边界

- 工具只允许操作 `--workspace` 指定的目录，默认是当前目录。
- `run_command` 会自动选择当前平台的本机 shell：Windows 优先 PowerShell，否则回退到 `cmd.exe`；macOS/Linux 使用当前 `$SHELL`。
- 这是一个“最小闭环”版本，适合本地代码阅读、简单修改和命令验证。
- 当前还没有 TUI/Web UI、多工作区、补丁审阅流或更细粒度权限系统。

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

### 4.1 Qwen `native/text` 模式开关

`QwenLLMProvider` 的工具调用模式开关是运行时参数 `tool_calling_mode`，不是 YAML 固定字段。

- `tool_calling_mode="auto"`: 默认值。若模型 ID 命中 `qwen3.6-35b-a3b*`，先走 `native`，失败后自动回退 `text`。
- `tool_calling_mode="native"`: 强制原生 Function Calling（请求体携带 `data.tools`）。
- `tool_calling_mode="text"`: 强制文本协议（不发送 `data.tools`，使用 `<tools>/<tool_call>` 提示词格式）。

直接调用 `stream_simple` 时：

```python
from pi_ai import stream_simple, get_model

model = get_model("QwenLLMprovider", "qwen3.6-35b-a3b-instruct")

response = await stream_simple(
    model,
    {
        "system_prompt": "You are a tool-using assistant.",
        "messages": messages,
        "tools": tools,
    },
    tool_calling_mode="text",  # "auto" | "native" | "text"
    static_memory=(
        "Project Facts:\\n"
        "- Repository root is /workspace\\n"
        "- Coding style: snake_case\\n"
        "- Never rename public APIs without migration notes"
    ),  # 固定不变上下文，一次定义，多轮复用
)
```

使用 `Agent` 时，可通过自定义 `stream_fn` 传入开关：

```python
from pi_ai import stream_simple, get_model
from pi_agent_core import Agent, AgentOptions

async def qwen_stream(model, context, **opts):
    return await stream_simple(
        model,
        context,
        tool_calling_mode="text",  # "auto" | "native" | "text"
        **opts,
    )

agent = Agent(
    AgentOptions(
        initial_state={
            "system_prompt": "You are a tool-using assistant.",
            "model": get_model("QwenLLMprovider", "qwen3.6-35b-a3b-instruct"),
            "tools": tools,
        },
        stream_fn=qwen_stream,
        strict_tool_arguments=True,  # 参数解析失败时返回结构化 tool error，触发模型自我修正
    )
)
```

当前 `llm.yaml` 不支持直接声明 `tool_calling_mode`，需要在运行时通过 `stream_simple(..., tool_calling_mode=...)` 或自定义 `stream_fn` 传入。
`strict_tool_arguments` 仅在 `Agent`/`agent_loop` 工具执行阶段生效，用于防止解析失败后以 `{}` 继续执行工具。
`static_memory` 会透传给 Qwen Provider 并注入到 `variable.static_memory`（若模板未包含，会自动把 `(static_memory)` 插入 `appInfo.prompt`）。

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

默认情况下 `Agent` 已启用一套轻量上下文治理（结构化摘要 + 最近窗口 + 近似 token 预算裁剪）。可通过以下参数调整：

```python
agent = Agent(AgentOptions(
    enable_context_memory=True,
    context_recent_messages=12,
    context_tool_results_to_keep=6,
    context_max_tokens=12000,
    context_summary_max_chars=6000,
))
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
