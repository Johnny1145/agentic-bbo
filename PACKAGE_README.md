# Portable Codex + Claude Code Benchmark Package

This source snapshot contains the unified native-harness implementation used
for Codex and Claude Code black-box optimization experiments with an SGLang
inner model.

Start with:

1. `workflow/42_codex_claude_full_matrix_agent_20260731/AGENT_WORKFLOW.md`
2. `workflow/42_codex_claude_full_matrix_agent_20260731/README.md`
3. `workflow/42_codex_claude_full_matrix_agent_20260731/preflight.py`

The package includes source, tests, task descriptions, evaluator code,
configuration, fixed task data, and a compact strict shared-initialization
bundle. It deliberately excludes virtual environments, caches, experiment
outputs, logs, agent workspaces, credentials, and model weights.

The native transports are not direct protocol assumptions:

- Codex Responses requests are translated through
  `bbo/algorithms/agentic/codex_responses_compat.py`.
- Claude Code Anthropic Messages requests are translated through
  `bbo/algorithms/agentic/claude_messages_compat.py`.
- Both compatibility layers target SGLang Chat Completions and preserve the
  harness-native file/shell tool loop.

Run `uv sync --frozen --extra general-agent --extra bo-tutorial --extra
surrogate` after extracting the archive; the `.venv` directory is intentionally
not shipped.
