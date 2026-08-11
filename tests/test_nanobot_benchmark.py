from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from bbo.benchmark import nanobot as nanobot_benchmark
from bbo.core import TrialSuggestion
from bbo.tasks import create_task


def _load_restricted_prior_workflow():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "workflow"
        / "exp1"
        / "run_prior_restriction_nanobot_skill_compare.py"
    )
    if not module_path.exists():
        pytest.skip(
            "restricted-prior workflow is local experiment orchestration and is not published"
        )
    spec = importlib.util.spec_from_file_location("prior_restriction_workflow", module_path)
    assert spec is not None
    assert spec.loader is not None
    workflow = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workflow)
    return workflow


def test_effective_max_evaluations_uses_initial_plus_optimizer_budget() -> None:
    assert (
        nanobot_benchmark.effective_max_evaluations(
            max_evaluations=None,
            initial_random=20,
            optimizer_budget=100,
        )
        == 120
    )
    with pytest.raises(ValueError, match="either"):
        nanobot_benchmark.effective_max_evaluations(
            max_evaluations=120,
            initial_random=20,
            optimizer_budget=100,
        )


def test_matrix_dry_run_expands_tasks_seeds_and_skill_modes(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = nanobot_benchmark.main(
        [
            "matrix",
            "--tasks",
            "bbob_f01_d10",
            "--seeds",
            "1,2",
            "--skill-modes",
            "both",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["planned_cases"] == [
        {"seed": 1, "skill_mode": "no-skill", "task_name": "bbob_f01_d10"},
        {"seed": 2, "skill_mode": "no-skill", "task_name": "bbob_f01_d10"},
        {"seed": 1, "skill_mode": "skill", "task_name": "bbob_f01_d10"},
        {"seed": 2, "skill_mode": "skill", "task_name": "bbob_f01_d10"},
    ]


def test_run_command_passes_nanobot_full_prior_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def fake_run_single_experiment(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        run_dir = kwargs["results_root"] / kwargs["task_name"] / kwargs["algorithm_name"] / f"seed_{kwargs['seed']}"
        run_dir.mkdir(parents=True)
        summary = {
            "task_name": kwargs["task_name"],
            "algorithm_name": kwargs["algorithm_name"],
            "seed": kwargs["seed"],
            "run_dir": str(run_dir),
            "results_jsonl": str(run_dir / "trials.jsonl"),
            "trial_count": 0,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return summary

    monkeypatch.setattr(nanobot_benchmark, "run_single_experiment", fake_run_single_experiment)

    exit_code = nanobot_benchmark.main(
        [
            "run",
            "--task",
            "bbob_f01_d10",
            "--seed",
            "3",
            "--skill-mode",
            "skill",
            "--initial-random",
            "5",
            "--optimizer-budget",
            "10",
            "--results-root",
            str(tmp_path),
            "--no-plots",
        ]
    )

    assert exit_code == 0
    assert len(calls) == 1
    call = calls[0]
    assert call["algorithm_name"] == "nanobot"
    assert call["max_evaluations"] == 15
    assert call["agent_initial_random"] == 5
    assert call["agent_tool_mode"] == "workspace_json"
    assert call["agent_enable_bbo_skills"] is True
    assert call["results_root"] == tmp_path / "full-prior" / "skill"


def test_run_command_accepts_no_tool_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def fake_run_single_experiment(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        run_dir = kwargs["results_root"] / kwargs["task_name"] / kwargs["algorithm_name"] / f"seed_{kwargs['seed']}"
        run_dir.mkdir(parents=True)
        summary = {
            "task_name": kwargs["task_name"],
            "algorithm_name": kwargs["algorithm_name"],
            "seed": kwargs["seed"],
            "run_dir": str(run_dir),
            "results_jsonl": str(run_dir / "trials.jsonl"),
            "trial_count": 0,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return summary

    monkeypatch.setattr(nanobot_benchmark, "run_single_experiment", fake_run_single_experiment)

    exit_code = nanobot_benchmark.main(
        [
            "run",
            "--task",
            "bbob_f01_d10",
            "--seed",
            "3",
            "--skill-mode",
            "no-skill",
            "--agent-tool-mode",
            "no-tool",
            "--results-root",
            str(tmp_path),
            "--no-plots",
        ]
    )

    assert exit_code == 0
    assert calls[0]["agent_tool_mode"] == "no_tool"
    assert calls[0]["agent_enable_bbo_skills"] is False


def test_skill_mode_rejects_no_tool_mode(tmp_path: Path) -> None:
    exit_code = nanobot_benchmark.main(
        [
            "run",
            "--task",
            "bbob_f01_d10",
            "--seed",
            "3",
            "--skill-mode",
            "skill",
            "--agent-tool-mode",
            "no-tool",
            "--results-root",
            str(tmp_path),
            "--no-plots",
        ]
    )

    assert exit_code == 1


def test_restricted_workflow_removes_task_identity_and_prior_section() -> None:
    workflow = _load_restricted_prior_workflow()

    task = create_task("bbob_f01_d10", max_evaluations=4, seed=7)
    wrapped = workflow.RestrictedPriorTask(task, agent_task_id="restricted_task_001")
    description = wrapped.get_description()

    assert wrapped.spec.name == "restricted_task_001"
    assert "prior_knowledge" not in description.section_map
    assert "bbob_f01_d10" not in description.rendered_context
    assert "Branin" not in description.rendered_context
    assert "objective expression" not in description.rendered_context
    assert "cos(x1)" not in description.rendered_context
    assert "known_optimum" not in wrapped.spec.metadata

    config = task.spec.search_space.defaults()
    config.update({"x1": 2.5, "x2": 4.5})
    result = wrapped.evaluate(TrialSuggestion(config=config))
    assert "problem_key" not in result.metadata
    assert "task_name" not in result.metadata
    assert "regret" not in result.metrics


def test_restricted_workflow_smoke_runs_both_variants_without_prior(tmp_path: Path) -> None:
    workflow = _load_restricted_prior_workflow()

    exit_code = workflow.main(
        [
            "--tasks",
            "bbob_f01_d10",
            "--seeds",
            "1",
            "--variant",
            "both",
            "--initial-random",
            "1",
            "--max-evaluations",
            "1",
            "--results-root",
            str(tmp_path),
            "--no-plots",
        ]
    )

    assert exit_code == 0
    summary = json.loads((tmp_path / "benchmark_summary.json").read_text(encoding="utf-8"))
    assert summary["failures"] == []
    assert [result["benchmark_metadata"]["skill_mode"] for result in summary["results"]] == ["no-skill", "skill"]

    for result in summary["results"]:
        assert result["task_name"] == "restricted_task_001"
        assert result["output_task_name"] == "restricted_task_001"
        assert result["best_trial_id"] == 0
        assert isinstance(result["best_config"], dict)
        assert result["best_config"]
        assert result["benchmark_metadata"]["exposure_policy"] == "restricted-prior"
        run_dir = Path(result["run_dir"])
        manifest = json.loads((run_dir / "agent_workspace" / "manifest.json").read_text(encoding="utf-8"))
        task_md = (run_dir / "agent_workspace" / "task.md").read_text(encoding="utf-8")
        assert manifest["task_id"] == "restricted_task_001"
        assert "prior_knowledge" not in task_md
        assert "bbob_f01_d10" not in task_md
        assert "Branin" not in task_md
        assert "objective expression" not in task_md
        assert "cos(x1)" not in task_md
        history_path = run_dir / "agent_workspace" / "history.jsonl"
        if history_path.exists() and history_path.read_text(encoding="utf-8").strip():
            history_text = history_path.read_text(encoding="utf-8")
            assert "bbob_f01_d10" not in history_text
            assert "problem_key" not in history_text


def test_collect_tool_usage_reports_skill_and_numeric_evidence_counts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "agent_tool_calls.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "agent_call_id": "agent_call_00000",
                        "tool_name": "summarize_objective_metrics",
                        "success": True,
                    }
                ),
                json.dumps(
                    {
                        "agent_call_id": "agent_call_00000",
                        "tool_name": "validate_candidate",
                        "success": True,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    llm_log_dir = run_dir / "llm_logs" / "agent_call_00000"
    llm_log_dir.mkdir(parents=True)
    (llm_log_dir / "mock_agent-end.json").write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "read_file",
                                    "arguments": json.dumps({"path": "skills/refine-incumbent/SKILL.md"}),
                                }
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "agent_calls.jsonl").write_text(
        json.dumps(
            {
                "accepted_candidates": 1,
                "accepted_search_actions": [
                    {
                        "skill": "refine-incumbent",
                        "search_intent": "exploitation",
                        "skill_audit": {
                            "declared_skill": "refine-incumbent",
                            "read_skills": ["refine-incumbent"],
                            "inferred_skill": "refine-incumbent",
                            "compliance": "declared_and_supported",
                        },
                    }
                ],
            }
        )
        + "\n"
        + json.dumps(
            {
                "accepted_candidates": 1,
                "accepted_search_actions": [
                    {
                        "skill": None,
                        "search_intent": "hypothesis_test",
                        "skill_audit": {
                            "declared_skill": None,
                            "read_skills": ["isolate-variable-effect"],
                            "inferred_skill": "isolate-variable-effect",
                            "compliance": "read_and_inferred_but_not_declared",
                        },
                    }
                ],
            }
        )
        + "\n"
        + json.dumps(
            {
                "validation_error": (
                    "Agent declared BBO skill `refine-incumbent` but did not call the "
                    "required BBO evidence tools in this same attempt."
                )
            }
        )
        + "\n",
        encoding="utf-8",
    )

    usage = nanobot_benchmark.collect_tool_usage(run_dir)

    assert usage["bbo_workspace_tool_calls"] == 2
    assert usage["bbo_workspace_non_validation_numeric_tool_calls"] == 1
    assert usage["skill_read_counts"] == {"refine-incumbent": 1}
    assert usage["accepted_skill_counts"] == {"refine-incumbent": 1}
    assert usage["accepted_search_intent_counts"] == {"exploitation": 1, "hypothesis_test": 1}
    assert usage["skill_evidence_failure_count"] == 1
    assert usage["skill_audit_count"] == 2
    assert usage["skill_audit_compliance_counts"] == {
        "declared_and_supported": 1,
        "read_and_inferred_but_not_declared": 1,
    }
    assert usage["skill_audit_declared_skill_counts"] == {"refine-incumbent": 1}
    assert usage["skill_audit_inferred_skill_counts"] == {
        "isolate-variable-effect": 1,
        "refine-incumbent": 1,
    }
    assert usage["skill_audit_read_skill_counts"] == {
        "isolate-variable-effect": 1,
        "refine-incumbent": 1,
    }
    assert usage["skill_audit_read_but_not_declared_count"] == 1
