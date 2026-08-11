# Nanobot、Codex 与 Claude Code 原生 Harness

这套实现把三个 coding-agent harness 接到同一个 `GeneralAgentBBOAlgorithm`
外层优化循环中：

- `nanobot` / `agentic_nanobot`
- `codex` / `agentic_codex`
- `claude_code` / `agentic_claude_code`

内层模型仍可统一使用 SGLang。三个 harness 分别走：

| Harness | SGLang 接口 | Harness 配置隔离 |
| --- | --- | --- |
| Nanobot | `/v1/chat/completions` | benchmark 生成的 Nanobot 配置 |
| Codex | 本地 `/v1/responses` 适配到 SGLang `/v1/chat/completions` | 每个 run 独立 `CODEX_HOME` |
| Claude Code | 本地 `/v1/messages` 适配到 SGLang `/v1/chat/completions` | 每个 run 独立 `CLAUDE_CONFIG_DIR` |

Codex 和 Claude Code 不直接依赖 SGLang 的实验性 Responses / Messages
实现。每次 agent call 会在 loopback 地址启动一个临时适配器，把
Codex Responses 或 Claude Code Anthropic Messages 请求、原生工具
schema 与流事件转换为 SGLang Chat Completions。
适配器显式关闭 Qwen separated reasoning，避免最终答案只落在
`reasoning_content` 后被 Messages 入口丢弃；同时保证每个
`text_delta` / `input_json_delta` 都指向类型正确的 Anthropic content
block。适配器不添加 BBO 工具，也不改变 Claude Code 的原生工具集合。

## `no_tool` 的准确含义

`no_tool` 只移除 benchmark 注入面：

- 不生成 `bbo_tool.py`、`bbo_tools.py`、`TOOLS.md` 或 BBO tool specs；
- 不复制 BBO skills；
- 不向 harness 注入 BBO function tools。

原生 harness 工具不做归一化，也不关闭：

- Nanobot 保持自己的文件、shell 与 agent-loop 工具；
- Codex 保持 Codex CLI 的原生工具；
- Claude Code 使用完整的 `claude_code` native tool preset。

为了避免用户环境影响结果，Codex 和 Claude Code 不加载用户已有的
settings、plugins、MCP servers 或 skills。这与关闭原生工具是两回事。

正式 Native run 现在强制 evaluator-sealed 黑盒边界，并采用 fail-closed
策略：Codex 与 Nanobot 只在 `bwrap` 创建的最小文件系统视图中启动，视图
只挂载 agent workspace、独立 state、系统运行库和 harness 可执行文件，
不会挂载仓库、task 实现或 evaluator 数据；继承环境也改为 allowlist，
避免把宿主凭据交给 agent。当前 Claude Agent SDK 是宿主进程内 transport，
还不能证明 file tools 看不到仓库；因此即使配置了 `permission_mode=default`、
强制 sandbox 和 `allowUnsandboxedCommands=false`，正式 Claude Code run
仍会 fail closed，直到改为独立 mount namespace runner。隔离失败使用专用
错误码并禁止随机 fallback。

强边界目前只支持 `no_tool` Native runs。`workspace_json` bridge 依赖宿主
Python/仓库路径，因此 Native engine 会在 agent 启动前拒绝该组合；需要
工具实验时使用 host-mediated `function_calling` engine。Native harness 的
CLI 默认也已改为 `no_tool`。本机若不能创建 bubblewrap namespace，smoke
与正式矩阵都会明确失败，不会生成可被误认为有效的 Agent 结果。

## 安装

```bash
cd /home/trx/agentic-bbo-multiharness-benchmark-v3
uv sync --extra general-agent --extra bo-tutorial --extra surrogate
```

Codex 需要系统中已有 `codex` 命令，也可以通过 `BBO_CODEX_BIN` 或
`--agent-executable` 指定。Claude Code 由 `claude-agent-sdk>=0.1.80`
自带的 CLI 驱动。

## 专用 SGLang 服务

不要复用当前为 OPRO 启动的 18300 服务。等 OPRO 结束后启动专用服务：

```bash
workflow/40_nanobot_no_tool_multifamily_20260724/start_dedicated_sglang_harness.sh
```

默认端口是 18301，并启用：

```text
--tool-call-parser qwen3_coder
--reasoning-parser qwen3
```

启动脚本检测到 OPRO multifamily runner 仍在运行时会直接拒绝启动，避免
争抢同一组 GPU。

检查所需 SGLang 接口：

```bash
workflow/40_nanobot_no_tool_multifamily_20260724/check_sglang_harness_api.py
```

这个检查只读 `/v1/models` 与 `/openapi.json`，不会发送推理请求。
Codex 和 Claude Code 的外部协议由进程内适配器提供，因此服务端只需
暴露 `/v1/chat/completions`；保留 `/v1/responses` 和 `/v1/messages`
不再是运行前提。

正式矩阵前，运行三个 harness 各一次 agent 调用的在线 smoke：

```bash
workflow/40_nanobot_no_tool_multifamily_20260724/run_native_harness_online_smoke.sh
```

## 96-run 矩阵

每个 harness 都运行相同的 96 个 task/seed 组合：

- synthetic：16 tasks × 2 seeds = 32；
- DBTune：6 × 2 = 12；
- molecule：10 × 2 = 20；
- BBOPlace：16 × 2 = 32。

先 dry-run：

```bash
workflow/40_nanobot_no_tool_multifamily_20260724/run_sglang_codex_no_tool.sh --dry-run
workflow/40_nanobot_no_tool_multifamily_20260724/run_sglang_claude_code_no_tool.sh --dry-run
workflow/40_nanobot_no_tool_multifamily_20260724/run_sglang_harness_no_tool.sh nanobot --dry-run
```

完整运行或选 family：

```bash
workflow/40_nanobot_no_tool_multifamily_20260724/run_sglang_codex_no_tool.sh
workflow/40_nanobot_no_tool_multifamily_20260724/run_sglang_claude_code_no_tool.sh molecule
workflow/40_nanobot_no_tool_multifamily_20260724/run_sglang_harness_no_tool.sh all synthetic
```

## Claude Code 与严格 OPRO 对齐的 96-run 矩阵

以下入口运行已确认的 Claude Code 完整矩阵：

```bash
workflow/40_nanobot_no_tool_multifamily_20260724/run_sglang_claude_code_opro_aligned.sh
```

它使用 seed 1/2、OPRO 相同的 family 预算，并逐条回放
`outputs/baselines/all` 中的初始化 observation：structured/BBOPlace
来自 `random_search`，molecule 来自 `graph_ga`。运行前会审计全部
96 个引用，初始化不重新调用 evaluator。默认 `jobs=8`、禁止 fallback，
结果写入 `outputs/claude_code/qwen35_9b_opro_aligned_full96_20260728/`。

Claude Code 的 SGLang 兼容层将每次 completion 上限限制为 4096 token，
并给 agent 暴露不含审计元数据的精简 history；完整审计字段仍保留在
`trials.jsonl`。两个 196 维 DBTune `*_all` 任务只暴露完整 incumbent
和 search space，不附加 history 行，因为 Claude Code 原生 system/tool
上下文加上完整 196 维历史会超过 65,536 token。其他任务保持
`history_limit=200`。

默认结果根目录是：

```text
workflow/40_nanobot_no_tool_multifamily_20260724/outputs/sglang_native_harnesses/
```

每个具体 run 都写入 `run_setting.json`。使用 `--resume` 时，runner 会
严格比较该文件；harness、模型、接口、预算、任务参数或工具策略任一变化
都会拒绝续跑。每次计划的完整快照另存于结果根目录 `_plans/`。

`summary.json` 同时记录：

- `benchmark_injected_tool_calls`：`no_tool` 下应为 0；
- `native_harness_tool_calls`：允许大于 0；
- 按工具名拆分的原生调用计数。
