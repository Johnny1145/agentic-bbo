from pathlib import Path
import pytest

from bbo.algorithms.agentic import create_agentic_method
from bbo.algorithms.agentic.prompt_profiles import PromptProfile, resolve_prompt_profile
from bbo.benchmark import BenchmarkRunConfig, run_named_benchmark


def test_prompt_profile_composes_protocol_and_round_layers() -> None:
    profile = PromptProfile(
        name="custom",
        protocol_instructions="Protocol addition.",
        round_instructions="Round addition.",
    )
    assert "Protocol addition." in profile.compose("base", stage="protocol")
    assert "Round addition." in profile.compose("base", stage="round")


def test_agentic_bo_owns_agentic_bo_prompt_profile(tmp_path: Path) -> None:
    decorated = create_agentic_method("agentic_bo", run_dir=tmp_path)
    assert decorated.algorithm.config.prompt_profile == resolve_prompt_profile("agentic_bo")


    profile = decorated.algorithm.config.prompt_profile
    protocol = profile.protocol_instructions
    rounds = profile.round_instructions
    assert "not an autopilot" in protocol
    assert "space-filling exploration" in protocol
    assert "Never record predictions as observations" in protocol
    assert "what this evaluation should learn" in rounds

def test_named_benchmark_api_runs_constructed_registry_entries(tmp_path: Path) -> None:
    result = run_named_benchmark(
        "bbob_f01_d10",
        "random_search",
        config=BenchmarkRunConfig(output_dir=tmp_path, seed=3),
        task_kwargs={"max_evaluations": 2},
    )
    assert result["trial_count"] == 2

    assert Path(result["results_jsonl"]).exists()

def test_workflow_profile_supports_distinct_multi_agent_roles() -> None:
    from bbo.algorithms.agentic.prompt_profiles import WorkflowPromptProfile

    workflow = WorkflowPromptProfile(
        name="team",
        roles={
            "planner": PromptProfile(name="planner", round_instructions="Plan only."),
            "worker": PromptProfile(name="worker", round_instructions="Execute only."),
        },
    )
    assert "Plan only." in workflow.for_role("planner").compose("base", stage="round")
    assert "Execute only." in workflow.for_role("worker").compose("base", stage="round")
    with pytest.raises(ValueError, match="has no role"):
        workflow.for_role("critic")


def test_multi_role_workflow_rejects_one_atomic_profile() -> None:
    from bbo.algorithms.agentic.prompt_profiles import resolve_workflow_prompt_profile

    with pytest.raises(ValueError, match="single PromptProfile"):
        resolve_workflow_prompt_profile(
            PromptProfile(name="one"),
            roles={"planner", "worker"},
        )


def test_registered_method_prompt_roles_match_declared_roles() -> None:
    from bbo.algorithms.agentic import AGENTIC_METHOD_REGISTRY

    for spec in AGENTIC_METHOD_REGISTRY.values():
        if spec.prompt_profile is not None:
            assert set(spec.prompt_profile.roles) == {role.name for role in spec.roles}


def test_pablo_uses_distinct_role_profiles(tmp_path: Path) -> None:
    from bbo.algorithms.agentic import PabloAlgorithm

    algorithm = PabloAlgorithm(provider="mock", run_dir=tmp_path)
    profiles = algorithm._prompt_profiles
    assert profiles.for_role("planner").name == "pablo.planner"
    assert profiles.for_role("explorer").name == "pablo.explorer"
    assert profiles.for_role("worker").name == "pablo.worker"
