from __future__ import annotations

import sys
import types
from pathlib import Path

from bbo.algorithms.agentic.nanobot_runner import (
    _bbo_exec_arguments,
    _extract_bbo_function_exec_call,
    _extract_bbo_method_exec_calls,
    _patch_bbo_mode_isolation,
)


def test_extracts_malformed_bbo_validate_candidate_text_call() -> None:
    content = """
    <tool_call>
    <function=BBO().validate_candidate({"x1": 5.0, "x2": 0.0})
    </parameter>
    </function>
    </tool_call>
    """

    calls = _extract_bbo_method_exec_calls(content)

    assert calls == [("validate_candidate", {"candidate": {"x1": 5.0, "x2": 0.0}})]


def test_bbo_validate_alias_maps_to_workspace_cli_tool() -> None:
    calls = _extract_bbo_method_exec_calls('BBO().validate([{"x1": 1.0, "x2": 2.0}])')

    assert calls == [("validate_candidates", {"candidates": [{"x1": 1.0, "x2": 2.0}]})]


def test_bbo_function_parameter_form_maps_to_workspace_cli_tool() -> None:
    call = _extract_bbo_function_exec_call(
        "BBO",
        """
        <parameter=tool_name>summarize_objective_metrics</parameter>
        <parameter=arguments>{"recent_limit": 4}</parameter>
        """,
    )

    assert call == ("summarize_objective_metrics", {"recent_limit": 4})


def test_unknown_bbo_method_is_not_converted_to_exec() -> None:
    assert _extract_bbo_method_exec_calls("Do not call BBO().initialize_search().") == []


def test_bbo_exec_command_uses_relative_workspace_cli() -> None:
    args = _bbo_exec_arguments("validate_candidate", {"candidate": {"x1": 0.0, "x2": 1.0}})

    assert args["command"].startswith("python bbo_tool.py validate_candidate ")
    assert "/home/" not in args["command"]
    assert args["timeout"] == 60


def test_bbo_exec_command_propagates_agent_call_id(monkeypatch) -> None:
    monkeypatch.setenv("BBO_AGENT_CALL_ID", "agent_call_00002")

    args = _bbo_exec_arguments("get_history_overview", {})

    assert args["command"].startswith("BBO_AGENT_CALL_ID=agent_call_00002 python bbo_tool.py ")


def _install_fake_nanobot_modules(monkeypatch, workspace: Path) -> tuple[type, type, types.ModuleType]:
    class ContextBuilder:
        def __init__(self) -> None:
            self.workspace = workspace

        def _get_identity(self, channel=None):  # noqa: ANN001
            del channel
            return "base identity\nCustom skills:\n- alpha\nRead SKILL.md before use\nkept line"

    class SkillsLoader:
        def build_skills_summary(self, exclude=None):  # noqa: ANN001
            del exclude
            return "available custom skills"

        def get_always_skills(self):
            return ["alpha"]

        def list_skills(self, filter_unavailable=True):  # noqa: ANN001
            del filter_unavailable
            return ["alpha"]

    class AgentLoop:
        def _register_default_tools(self):
            return "registered"

    nanobot_module = types.ModuleType("nanobot")
    agent_module = types.ModuleType("nanobot.agent")
    context_module = types.ModuleType("nanobot.agent.context")
    skills_module = types.ModuleType("nanobot.agent.skills")
    loop_module = types.ModuleType("nanobot.agent.loop")
    cli_module = types.ModuleType("nanobot.cli")
    commands_module = types.ModuleType("nanobot.cli.commands")

    context_module.ContextBuilder = ContextBuilder
    skills_module.SkillsLoader = SkillsLoader
    loop_module.AgentLoop = AgentLoop
    commands_module.sync_workspace_templates = lambda workspace, silent=False: ["TOOLS.md", "skills"]  # noqa: ARG005

    agent_module.context = context_module
    agent_module.skills = skills_module
    agent_module.loop = loop_module
    cli_module.commands = commands_module
    nanobot_module.agent = agent_module
    nanobot_module.cli = cli_module

    for name, module in {
        "nanobot": nanobot_module,
        "nanobot.agent": agent_module,
        "nanobot.agent.context": context_module,
        "nanobot.agent.skills": skills_module,
        "nanobot.agent.loop": loop_module,
        "nanobot.cli": cli_module,
        "nanobot.cli.commands": commands_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return SkillsLoader, AgentLoop, commands_module


def test_no_skill_mode_disables_nanobot_skill_surface(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("BBO_NANOBOT_NO_TOOL_MODE", raising=False)
    monkeypatch.setenv("BBO_NANOBOT_NO_SKILL_MODE", "1")
    skills_loader_cls, agent_loop_cls, commands_module = _install_fake_nanobot_modules(monkeypatch, tmp_path)

    _patch_bbo_mode_isolation()

    loader = skills_loader_cls()
    assert loader.build_skills_summary() == ""
    assert loader.get_always_skills() == []
    assert loader.list_skills() == []
    assert "Custom skills" not in sys.modules["nanobot.agent.context"].ContextBuilder()._get_identity()
    assert "SKILL.md" not in sys.modules["nanobot.agent.context"].ContextBuilder()._get_identity()
    assert "kept line" in sys.modules["nanobot.agent.context"].ContextBuilder()._get_identity()
    assert agent_loop_cls()._register_default_tools() == "registered"
    assert commands_module.sync_workspace_templates(tmp_path) == []


def test_no_tool_mode_keeps_nanobot_native_tools_and_disables_skills(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BBO_NANOBOT_NO_TOOL_MODE", "1")
    monkeypatch.delenv("BBO_NANOBOT_NO_SKILL_MODE", raising=False)
    skills_loader_cls, agent_loop_cls, commands_module = _install_fake_nanobot_modules(monkeypatch, tmp_path)

    _patch_bbo_mode_isolation()

    assert skills_loader_cls().list_skills() == []
    assert agent_loop_cls()._register_default_tools() == "registered"
    identity = sys.modules["nanobot.agent.context"].ContextBuilder()._get_identity()
    assert "Custom skills" not in identity
    assert "SKILL.md" not in identity
    assert "kept line" in identity
    assert commands_module.sync_workspace_templates(tmp_path) == []
