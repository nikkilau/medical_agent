# Micro-MDT

Micro-MDT 是一个低成本、多智能体医疗安全工作流实验项目。它基于 `multi-agent.md` 中的设想，把 test-time compute、Proposer/Verifier、Sequential Revision、Abstention 和 human-in-the-loop 做成一个可运行的 Python MVP。

> 重要声明：本项目仅用于计算机科学实验和医疗 AI 安全机制演示，不能用于真实诊断、处方或治疗决策。

## 功能

- 分诊 Agent：把病例路由到 `LOW`、`MEDIUM`、`HIGH` 三种算力路径。
- 全科医生 Agent：生成初步方案或修订方案。
- 临床药师 Agent：审查药物相互作用、禁忌和越权用药。
- 安全伦理 Agent：检查急症延误、高风险建议和拒答条件。
- 多轮辩论：高复杂度病例最多运行 3 轮 proposer/verifier/reviser。
- 安全拒答：无法达成共识或急症风险高时，生成分歧报告并要求人类医生接管。
- 人类医生回路：交互输入医生最终决策后，文书 Agent 生成复诊记录、患者说明和复诊提醒草稿。
- 零依赖默认演示：内置 deterministic mock provider，不需要 API key。
- OpenAI-compatible provider：可接 OpenAI、DeepSeek、Qwen 等兼容 `/chat/completions` 的服务。

## 快速开始

```powershell
python -m micro_mdt.cli --case-id case_high_001 --show-trace
```

如果没有安装为包，请先设置 `PYTHONPATH`：

```powershell
$env:PYTHONPATH="src"
python -m micro_mdt.cli --case-id case_high_001 --show-trace
```

运行全部示例病例：

```powershell
$env:PYTHONPATH="src"
python -m micro_mdt.cli
```

启用人类医生接管和文书生成：

```powershell
$env:PYTHONPATH="src"
python -m micro_mdt.cli --case-id case_high_003 --interactive-human
```

输入临时病例：

```powershell
$env:PYTHONPATH="src"
python -m micro_mdt.cli --case "患者 70 岁，胸痛伴呼吸困难 30 分钟，有高血压史。"
```

## 使用真实模型

本项目启动时会自动读取项目根目录的 `.env` 文件，也仍然支持直接使用系统环境变量。不要把真实 API key 提交到仓库。

推荐先复制环境模板：

```powershell
copy .env.example .env
```

然后在 `.env` 中填入自己的 DeepSeek 或其他 OpenAI-compatible 服务配置：

```env
MICRO_MDT_API_KEY=your_deepseek_api_key_here
MICRO_MDT_BASE_URL=https://api.deepseek.com/v1
MICRO_MDT_MODEL=deepseek-chat
```

Windows PowerShell 启动命令：

```powershell
$env:PYTHONPATH="src"
python -m micro_mdt.cli --provider openai-compatible --case-id case_high_001
python -m micro_mdt.cli --web --provider openai-compatible
```

Windows CMD 启动命令：

```bat
set PYTHONPATH=src
python -m micro_mdt.cli --provider openai-compatible --case-id case_high_001
python -m micro_mdt.cli --web --provider openai-compatible
```

也可以不用 `.env`，直接设置环境变量：

```powershell
$env:MICRO_MDT_API_KEY="你的 API key"
$env:MICRO_MDT_BASE_URL="https://api.openai.com/v1"
$env:MICRO_MDT_MODEL="gpt-4o-mini"
$env:PYTHONPATH="src"
python -m micro_mdt.cli --provider openai-compatible --case-id case_high_001
```

DeepSeek、Qwen 等服务如果兼容 OpenAI Chat Completions，也可以替换 `MICRO_MDT_BASE_URL` 和 `MICRO_MDT_MODEL`。注意 provider 名称必须写完整：`openai-compatible`。

## 测试

```powershell
$env:PYTHONPATH="src"
python -m unittest discover -s tests
```

## 项目结构

```text
src/micro_mdt/
  agents.py       Agent 封装、分级和审查结果解析
  cli.py          命令行入口
  io.py           病例加载
  models.py       数据结构
  prompts.py      各 Agent system prompts
  providers.py    mock 与 OpenAI-compatible provider
  reporting.py    终端报告渲染
  workflow.py     Micro-MDT 核心编排
examples/
  cases.json      演示病例
```

## 设计对应关系

- Compute-Optimal：`workflow.py` 根据分诊结果选择低、中、高算力路径。
- Proposer vs Verifier：全科医生提出方案，药师和安全伦理审查。
- Sequential Revisions：审查不通过时，把反馈交给修订 Agent 再生成新版方案。
- Abstention：急症风险或多轮仍不通过时，拒绝输出最终诊疗方案。
- Human-in-the-loop：`--interactive-human` 让医生输入最终决策，再由文书 Agent 执行后续草稿生成。
