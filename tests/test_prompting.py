from __future__ import annotations

import json
from pathlib import Path

from bbo.core import (
    CategoricalParam,
    EvaluationResult,
    FloatParam,
    IntParam,
    ObjectiveDirection,
    ObjectiveSpec,
    SearchSpace,
    StringParam,
    TaskDescriptionBundle,
    TaskDescriptionDoc,
    TaskSpec,
    TrialObservation,
    TrialSuggestion,
    build_prompt_context,
    candidate_response_json_schema,
    compile_candidate_generation_prompt,
    compile_score_prediction_prompt,
    default_candidate_generation_contract,
    default_score_prediction_contract,
    format_candidate_response,
    parse_candidate_response,
    parse_score_prediction_response,
    render_history_block,
    render_task_context_block,
    search_space_to_schema,
)


def _task_spec(
    direction: ObjectiveDirection = ObjectiveDirection.MINIMIZE,
    *,
    metadata: dict | None = None,
) -> TaskSpec:
    return TaskSpec(
        name="prompt_demo",
        search_space=SearchSpace(
            [
                FloatParam("lr", low=1e-4, high=1e-1, log=True, default=1e-2),
                IntParam("depth", low=2, high=8, default=4),
                CategoricalParam("activation", choices=("relu", "gelu"), default="relu"),
                StringParam("tag", default="seed", min_length=1, max_length=12, pattern=r"[A-Za-z0-9_]+"),
            ]
        ),
        objectives=(ObjectiveSpec("loss", direction),),
        max_evaluations=9,
        metadata=dict(metadata or {}),
    )


def _description() -> TaskDescriptionBundle:
    return TaskDescriptionBundle(
        task_id="prompt_demo",
        primary=TaskDescriptionDoc(
            path=Path("background.md"),
            kind="background",
            title="Background",
            content="# Background\n\nThis task tunes a generic system.",
        ),
        extras=(
            TaskDescriptionDoc(
                path=Path("goal.md"),
                kind="goal",
                title="Goal",
                content="# Goal\n\nMinimize the observed loss.",
            ),
        ),
        section_map={
            "background": "# Background\n\nThis task tunes a generic system.",
            "goal": "# Goal\n\nMinimize the observed loss.",
        },
    )


def _observation(task_spec: TaskSpec, config: dict, value: float, trial_id: int) -> TrialObservation:
    return TrialObservation.from_evaluation(
        TrialSuggestion(config=config, trial_id=trial_id),
        EvaluationResult(objectives={task_spec.primary_objective.name: value}),
    )


def test_search_space_schema_and_candidate_json_schema_include_string_params() -> None:
    task_spec = _task_spec()

    schema = search_space_to_schema(task_spec.search_space)
    string_param = next(item for item in schema if item["name"] == "tag")

    assert string_param == {
        "name": "tag",
        "type": "string",
        "min_length": 1,
        "max_length": 12,
        "pattern": r"[A-Za-z0-9_]+",
        "default": "seed",
    }
    response_schema = candidate_response_json_schema(task_spec.search_space, num_candidates=2)
    assert response_schema["properties"]["candidates"]["minItems"] == 2
    assert response_schema["properties"]["candidates"]["maxItems"] == 2
    assert response_schema["properties"]["candidates"]["items"]["properties"]["tag"]["type"] == "string"


def test_history_block_sorts_successful_trials_best_to_worst() -> None:
    task_spec = _task_spec()
    history = [
        _observation(task_spec, {"lr": 0.01, "depth": 4, "activation": "relu", "tag": "a"}, 0.5, 0),
        _observation(task_spec, {"lr": 0.02, "depth": 5, "activation": "gelu", "tag": "b"}, 0.1, 1),
        _observation(task_spec, {"lr": 0.03, "depth": 6, "activation": "relu", "tag": "c"}, 0.3, 2),
    ]

    rows = json.loads(render_history_block(history, task_spec))

    assert [row["objective"] for row in rows] == [0.1, 0.3, 0.5]
    assert [row["trial_id"] for row in rows] == [1, 2, 0]


def test_default_candidate_prompt_contract_keeps_all_successful_history() -> None:
    task_spec = _task_spec()
    history = [
        _observation(
            task_spec,
            {"lr": 0.01, "depth": 4, "activation": "relu", "tag": f"t{index}"},
            float(index),
            index,
        )
        for index in range(25)
    ]

    context = build_prompt_context(
        task_spec=task_spec,
        description=_description(),
        history=history,
        contract=default_candidate_generation_contract(),
    )
    rows = json.loads(context.history_block)

    assert len(rows) == 25
    assert {row["trial_id"] for row in rows} == set(range(25))


def test_general_candidate_and_score_prompts_compile_from_context() -> None:
    task_spec = _task_spec()
    history = [
        _observation(task_spec, {"lr": 0.01, "depth": 4, "activation": "relu", "tag": "a"}, 0.5, 0),
    ]
    candidate_contract = default_candidate_generation_contract(max_history_trials=5)
    candidate_context = build_prompt_context(
        task_spec=task_spec,
        description=_description(),
        history=history,
        contract=candidate_contract,
    )

    prompt = compile_candidate_generation_prompt(candidate_context, num_candidates=3)

    assert "general black-box optimization task" in prompt
    assert "Task name: prompt_demo" in prompt
    assert "Better objective values are: lower" in prompt
    assert "Return exactly 3 candidates" in prompt
    assert "This task tunes a generic system." in prompt
    assert '"trial_id": 0' in prompt

    score_contract = default_score_prediction_contract(max_history_trials=5)
    score_context = build_prompt_context(
        task_spec=task_spec,
        description=_description(),
        history=history,
        contract=score_contract,
        candidate_config={"lr": 0.02, "depth": 5, "activation": "gelu", "tag": "b"},
    )
    score_prompt = compile_score_prediction_prompt(score_context)
    assert '"predicted_objective": <number>' in score_prompt
    assert '{"lr":0.02,"depth":5,"activation":"gelu","tag":"b"}' in score_prompt


def test_prompt_task_name_prefers_display_name_metadata() -> None:
    task_spec = _task_spec(metadata={"display_name": "Readable BBO task instance"})
    context = build_prompt_context(
        task_spec=task_spec,
        description=_description(),
        history=[],
        contract=default_candidate_generation_contract(max_history_trials=5),
    )

    prompt = compile_candidate_generation_prompt(context, num_candidates=1)

    assert "Task name: Readable BBO task instance" in prompt
    assert "Task name: prompt_demo" not in prompt


def test_default_task_context_only_includes_core_sections() -> None:
    docs = (
        TaskDescriptionDoc(Path("goal.md"), "# Goal\n\nGoal details.", "goal", "Goal"),
        TaskDescriptionDoc(Path("constraints.md"), "# Constraints\n\nConstraint details.", "constraints", "Constraints"),
        TaskDescriptionDoc(
            Path("prior_knowledge.md"),
            "# Domain Prior Knowledge\n\nPrior details.",
            "prior_knowledge",
            "Domain Prior Knowledge",
        ),
        TaskDescriptionDoc(
            Path("evaluation.md"),
            "# Evaluation Protocol\n\nEvaluation details.",
            "evaluation",
            "Evaluation Protocol",
        ),
        TaskDescriptionDoc(Path("submission.md"), "# Submission\n\nSubmission details.", "submission", "Submission"),
        TaskDescriptionDoc(Path("environment.md"), "# Environment\n\nEnvironment details.", "environment", "Environment"),
    )
    bundle = TaskDescriptionBundle(
        task_id="prompt_demo",
        primary=TaskDescriptionDoc(
            Path("background.md"),
            "# Background\n\nBackground details.",
            "background",
            "Background",
        ),
        extras=docs,
    )

    block = render_task_context_block(bundle)

    assert "Background details." in block
    assert "Goal details." in block
    assert "Constraint details." in block
    assert "Prior details." in block
    assert "Evaluation details." not in block
    assert "Submission details." not in block
    assert "Environment details." not in block


def test_candidate_and_score_output_parsers_accept_new_json_and_legacy_tags() -> None:
    task_spec = _task_spec()
    candidates = [
        {"lr": 0.02, "depth": 5, "activation": "gelu", "tag": "b"},
        {"lr": 0.03, "depth": 6, "activation": "relu", "tag": "c"},
    ]
    payload = format_candidate_response(task_spec.search_space, candidates)

    parsed = parse_candidate_response(payload, task_spec.search_space, expected_count=2)

    assert parsed == candidates
    legacy = '<candidate>{"lr":0.02,"depth":5,"activation":"gelu","tag":"b"}</candidate>'
    assert parse_candidate_response(legacy, task_spec.search_space) == [candidates[0]]
    wrapped = json.dumps({"candidates": [{"config": candidates[0], "rationale": "ok"}]})
    assert parse_candidate_response(wrapped, task_spec.search_space) == [candidates[0]]
    assert parse_score_prediction_response('{"predicted_objective": 0.125}') == 0.125
    assert parse_score_prediction_response("<score>0.25</score>") == 0.25
