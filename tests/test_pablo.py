from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

from bbo.algorithms import ALGORITHM_REGISTRY
from bbo.algorithms.agentic import (
    PabloAlgorithm,
    PabloProviderConfig,
    build_explorer_prompt,
    build_worker_prompt,
    create_llm_client,
)
from bbo.algorithms.agentic.model_routing import PabloModelRoutingConfig, build_routing_table, resolve_role_model
from bbo.algorithms.agentic.prompts import PromptBundle
from bbo.algorithms.agentic.validation import validate_candidate_payload
from bbo.core import EvaluationResult, TrialObservation, TrialStatus, TrialSuggestion
from bbo.run import build_arg_parser, run_single_experiment
from bbo.tasks import create_bboplace_task, create_molecule_qed_task, create_oer_task, create_task, default_bboplace_definition
from conftest import create_agent_test_task
from bbo.tasks.scientific import CACHE_ROOT_ENV, SOURCE_ROOT_ENV, VENDORED_SOURCE_ROOT


REQUIRED_ARTIFACT_KEYS = {
    "pablo_rounds_jsonl",
    "task_registry_json",
    "llm_calls_jsonl",
    "candidate_queue_jsonl",
    "resume_state_json",
}


def _require_bo_tutorial_source() -> Path:
    source_root = Path(os.environ.get(SOURCE_ROOT_ENV, str(VENDORED_SOURCE_ROOT)))
    if not source_root.exists():
        pytest.skip("Bundled scientific task datasets are not available in the workspace.")
    return source_root


@pytest.fixture
def scientific_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    source_root = _require_bo_tutorial_source()
    monkeypatch.setenv(SOURCE_ROOT_ENV, str(source_root))
    monkeypatch.setenv(CACHE_ROOT_ENV, str(tmp_path / "dataset_cache"))
    return source_root


def test_pablo_and_palbo_are_registered_and_cli_visible() -> None:
    parser = build_arg_parser()
    algorithm_action = next(action for action in parser._actions if action.dest == "algorithm")

    assert "pablo" in ALGORITHM_REGISTRY
    assert "palbo" in ALGORITHM_REGISTRY
    assert ALGORITHM_REGISTRY["pablo"].family == "agentic"
    assert ALGORITHM_REGISTRY["palbo"].family == "agentic"
    assert "pablo" in algorithm_action.choices
    assert "palbo" in algorithm_action.choices
    assert parser.parse_args(["--algorithm", "palbo"]).algorithm == "palbo"
    parsed = parser.parse_args(["--algorithm", "pablo", "--pablo-provider", "kimi"])
    assert parsed.pablo_provider == "kimi"


def test_pablo_model_routing_and_worker_prompt_boundary() -> None:
    routing = build_routing_table(
        PabloModelRoutingConfig(
            model="base-model",
            global_model="global-model",
            worker_model="worker-model",
            planner_model="planner-model",
        )
    )
    assert routing == {
        "planner": "planner-model",
        "explorer": "global-model",
        "worker": "worker-model",
    }
    assert resolve_role_model("worker", PabloModelRoutingConfig(model="base-model", worker_model=None)) == "base-model"

    task = create_agent_test_task(max_evaluations=3, seed=7)
    prompt = build_worker_prompt(
        task_spec=task.spec,
        planner_task_name="EXPLOIT_TOP",
        planner_task_text="TASK: refine around the current strong seed.",
        current_seed=task.spec.search_space.defaults(),
    )
    assert "current_seed" in prompt.user
    assert "c_global" not in prompt.user
    assert "c_global" not in prompt.system
    assert prompt.context["planner_task_name"] == "EXPLOIT_TOP"


def test_pablo_initial_points_are_random_not_defaults(tmp_path: Path) -> None:
    task = create_agent_test_task(max_evaluations=3, seed=7)
    algorithm = PabloAlgorithm(provider="mock", run_dir=tmp_path)
    algorithm.setup(task.spec, seed=7, task_description=task.get_description())

    suggestion = algorithm.ask()

    assert suggestion.metadata["pablo_source"] == "initial_random"
    assert suggestion.metadata["pablo_role"] == "init"
    assert suggestion.config != task.spec.search_space.defaults()


def test_pablo_c_global_uses_top_8_plus_coverage_12(tmp_path: Path) -> None:
    task = create_agent_test_task(max_evaluations=30, seed=7)
    algorithm = PabloAlgorithm(provider="mock", run_dir=tmp_path)
    algorithm.setup(task.spec, seed=7, task_description=task.get_description())

    for trial_id in range(25):
        suggestion = TrialSuggestion(
            config={"x1": -5.0 + trial_id * 0.5, "x2": float(trial_id % 16)},
            trial_id=trial_id,
            metadata={"pablo_task_name": "INIT"},
        )
        algorithm.tell(
            TrialObservation.from_evaluation(
                suggestion,
                EvaluationResult(
                    status=TrialStatus.SUCCESS,
                    objectives={task.spec.primary_objective.name: float(trial_id)},
                ),
            )
        )

    c_global = algorithm._build_c_global()

    assert len(c_global) == 20
    assert [entry["trial_id"] for entry in c_global[:8]] == list(range(8))
def _pablo_observation(suggestion: TrialSuggestion, task, score: float) -> TrialObservation:
    return TrialObservation.from_evaluation(
        suggestion,
        EvaluationResult(
            status=TrialStatus.SUCCESS,
            objectives={task.spec.primary_objective.name: score},
        ),
    )


def test_pablo_feedback_loop_waits_for_tell_and_advances_after_max_fails(tmp_path: Path) -> None:
    task = create_agent_test_task(max_evaluations=10, seed=7)
    algorithm = PabloAlgorithm(
        provider="mock", run_dir=tmp_path, init_points=1, max_fails=2,
        num_seeds=1, enable_planner=False,
    )
    algorithm.setup(task.spec, seed=7, task_description=task.get_description())
    candidates = iter([
        [{"x1": 0.1, "x2": 0.1}],
        [{"x1": 0.2, "x2": 0.2}],
        [{"x1": 0.3, "x2": 0.3}],
    ])
    algorithm._invoke_candidate_role = lambda role, prompt: next(candidates)  # type: ignore[method-assign]

    initial = algorithm.ask()
    algorithm.tell(_pablo_observation(initial, task, 0.0))
    explorer_one = algorithm.ask()
    assert explorer_one.metadata["pablo_role"] == "explorer"
    assert algorithm._active_search is not None
    algorithm.tell(_pablo_observation(explorer_one, task, 1.0))
    assert algorithm._active_search is not None
    assert algorithm._active_search.consecutive_failures == 1

    explorer_two = algorithm.ask()
    assert explorer_two.metadata["pablo_role"] == "explorer"
    algorithm.tell(_pablo_observation(explorer_two, task, 2.0))
    assert algorithm._active_search is None

    worker = algorithm.ask()
    assert worker.metadata["pablo_role"] == "worker"
    assert worker.metadata["pablo_feedback_step"] is True


def test_pablo_worker_improvement_updates_seed_and_resume_state(tmp_path: Path) -> None:
    task = create_agent_test_task(max_evaluations=10, seed=7)
    algorithm = PabloAlgorithm(
        provider="mock", run_dir=tmp_path, init_points=1, max_fails=2,
        num_seeds=1, enable_explorer=False, enable_planner=False,
    )
    algorithm.setup(task.spec, seed=7, task_description=task.get_description())
    candidates = iter([[{"x1": 0.4, "x2": 0.4}], [{"x1": 0.35, "x2": 0.35}]])
    algorithm._invoke_candidate_role = lambda role, prompt: next(candidates)  # type: ignore[method-assign]

    initial = algorithm.ask()
    algorithm.tell(_pablo_observation(initial, task, 5.0))
    worker = algorithm.ask()
    algorithm.tell(_pablo_observation(worker, task, 1.0))

    assert algorithm._active_search is not None
    assert algorithm._active_search.role == "worker"
    assert algorithm._active_search.current_seed == worker.config
    assert algorithm._active_search.consecutive_failures == 0

    resumed = PabloAlgorithm(
        provider="mock", run_dir=tmp_path, init_points=1, max_fails=2,
        num_seeds=1, enable_explorer=False, enable_planner=False, resume=True,
    )
    resumed.setup(task.spec, seed=7, task_description=task.get_description())
    resumed.replay([_pablo_observation(initial, task, 5.0), _pablo_observation(worker, task, 1.0)])
    assert resumed._active_search is not None
    assert resumed._active_search.current_seed == worker.config
    assert resumed._active_search.task_name == worker.metadata["pablo_task_name"]


def test_candidate_validation_accepts_wrapped_config_objects() -> None:
    task = create_agent_test_task(max_evaluations=3, seed=7)
    payload = """
    {
      "candidates": [
        {"config": {"x1": 0.5, "x2": 0.5}, "rationale": "keep near center"},
        {"x1": 0.1, "x2": 0.9}
      ]
    }
    """

    candidates = validate_candidate_payload(payload, task.spec.search_space)

    assert candidates == [{"x1": 0.5, "x2": 0.5}, {"x1": 0.1, "x2": 0.9}]


def test_candidate_validation_accepts_compact_bboplace_xy_arrays() -> None:
    task = create_bboplace_task(definition=default_bboplace_definition(n_macro=2), max_evaluations=3, seed=7)
    payload = """
    {
      "candidates": [
        {"x": [1.23456, 200.0], "y": [3.45678, 100]},
        {"config": {"x": [8.0, 9.0], "y": [10.0, 11.0]}, "rationale": "compact wrapped"}
      ]
    }
    """

    candidates = validate_candidate_payload(payload, task.spec.search_space)

    assert candidates == [
        {"x_0": 1.23456, "x_1": 200.0, "y_0": 3.45678, "y_1": 100.0},
        {"x_0": 8.0, "x_1": 9.0, "y_0": 10.0, "y_1": 11.0},
    ]


def test_worker_prompt_uses_compact_bboplace_xy_format() -> None:
    task = create_bboplace_task(definition=default_bboplace_definition(n_macro=2), max_evaluations=3, seed=7)
    prompt = build_worker_prompt(
        task_spec=task.spec,
        description=task.get_description(),
        planner_task_name="SIMILAR_LAYOUT",
        planner_task_text="TASK: generate similar layouts.",
        current_seed={"x_0": 1.23456, "x_1": 200.0, "y_0": 3.45678, "y_1": 100.0},
    )

    assert '"current_seed": {"x":[1.2346,200.0],"y":[3.4568,100.0]}' in prompt.user
    assert "exactly 2 x coordinates and 2 y coordinates" in prompt.user
    assert '"x":[75.1234,154.5678]' in prompt.user


def test_explorer_prompt_uses_one_compact_bboplace_candidate() -> None:
    task = create_bboplace_task(definition=default_bboplace_definition(n_macro=2), max_evaluations=3, seed=7)
    prompt = build_explorer_prompt(
        task_spec=task.spec,
        description=task.get_description(),
        c_global=[
            {
                "trial_id": 0,
                "status": "success",
                "primary_objective": 1.0,
                "config": {"x_0": 1.0, "x_1": 2.0, "y_0": 3.0, "y_1": 4.0},
            }
        ],
        best_objective=1.0,
        observed_trial_count=1,
    )
    client = create_llm_client(PabloProviderConfig(provider="mock"), seed=7)

    raw = client.complete(role="explorer", model="mock-model", prompt=prompt)
    candidates = validate_candidate_payload(raw, task.spec.search_space)

    assert "Return exactly 1 candidate in the candidates list." in prompt.user
    assert "Propose exactly 1 NEW compact coordinate configuration" in prompt.user
    assert len(candidates) == 1


def test_pablo_kimi_provider_uses_openai_compatible_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KIMI_TEST_KEY", "sk-test")
    captured: dict[str, object] = {}

    class _FakeCompletions:
        def create(self, **kwargs):
            captured["kwargs"] = kwargs
            message = types.SimpleNamespace(content='{"candidates":[]}')
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    class _FakeOpenAI:
        def __init__(self, *, api_key: str, base_url: str | None = None):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            self.chat = types.SimpleNamespace(completions=_FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_FakeOpenAI))

    client = create_llm_client(
        PabloProviderConfig(
            provider="kimi",
            base_url="https://api.kimi.com/coding/",
            api_key_env="KIMI_TEST_KEY",
        )
    )
    raw = client.complete(
        role="worker",
        model="kimi-for-coding",
        prompt=PromptBundle(role="worker", system="Return JSON.", user='{"current_seed":{}}'),
    )

    assert raw == '{"candidates":[]}'
    assert captured["api_key"] == "sk-test"
    assert captured["base_url"] == "https://api.kimi.com/coding/v1"
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["model"] == "kimi-for-coding"
    assert kwargs["messages"][0]["content"] == "Return JSON."
    assert kwargs["messages"][1]["content"] == '{"current_seed":{}}'
    assert kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("task_name", ["her_demo", "oer_demo"])
def test_pablo_mock_scientific_smoke(task_name: str, scientific_env: Path, tmp_path: Path) -> None:
    summary = run_single_experiment(
        task_name=task_name,
        algorithm_name="pablo",
        seed=5,
        max_evaluations=5,
        results_root=tmp_path,
        resume=False,
        pablo_provider="mock",
    )

    assert summary["trial_count"] == 5
    assert summary["best_primary_objective"] is not None
    assert Path(summary["results_jsonl"]).exists()
    assert REQUIRED_ARTIFACT_KEYS <= set(summary["internal_artifacts"])
    assert summary["role_model_routes"]["planner"] == "gpt-4.1-mini"
    for artifact_path in summary["internal_artifacts"].values():
        assert Path(artifact_path).exists()


def test_pablo_mock_molecule_and_alias_smoke(scientific_env: Path, tmp_path: Path) -> None:
    pytest.importorskip("rdkit")

    summary = run_single_experiment(
        task_name="molecule_qed_demo",
        algorithm_name="pablo",
        seed=5,
        max_evaluations=5,
        results_root=tmp_path,
        resume=False,
        pablo_provider="mock",
    )
    alias_summary = run_single_experiment(
        task_name="her_demo",
        algorithm_name="palbo",
        seed=6,
        max_evaluations=4,
        results_root=tmp_path,
        resume=False,
        pablo_provider="mock",
        pablo_model="base-model",
        pablo_global_model="global-model",
        pablo_worker_model="worker-model",
        pablo_planner_model="planner-model",
        pablo_explorer_model="explorer-model",
    )

    assert summary["trial_count"] == 5
    assert 0.0 <= float(summary["best_primary_objective"]) <= 1.0
    assert alias_summary["trial_count"] == 4
    assert alias_summary["algorithm_name"] == "pablo"
    assert alias_summary["role_model_routes"] == {
        "planner": "planner-model",
        "explorer": "explorer-model",
        "worker": "worker-model",
    }
    for artifact_path in alias_summary["internal_artifacts"].values():
        assert Path(artifact_path).exists()
