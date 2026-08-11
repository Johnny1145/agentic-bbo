"""Prompt builders for the Pablo Planner/Explorer/Worker roles."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ...core import ObjectiveDirection, SearchSpace, TaskDescriptionBundle, TaskSpec, search_space_to_schema
from ...core.prompting import (
    SearchSpaceRenderPolicy,
    TaskContextRenderPolicy,
    render_search_space_block,
    render_task_context_block,
)
from .compact import COMPACT_COORD_DECIMALS, bboplace_macro_count, compact_xy_config, is_compact_xy_space


@dataclass(frozen=True)
class PromptBundle:
    role: str
    system: str
    user: str
    context: dict[str, Any] = field(default_factory=dict)


def summarize_search_space(search_space: SearchSpace, *, max_choices: int = 16) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for item in search_space_to_schema(search_space):
        if item.get("type") != "categorical":
            summary.append(item)
            continue
        choices = list(item.get("choices", []))
        summarized = dict(item)
        summarized["choice_count"] = len(choices)
        summarized["choices_preview"] = choices[:max_choices]
        summarized["choices_truncated"] = len(choices) > max_choices
        summarized.pop("choices", None)
        summary.append(summarized)
    return summary


def summarize_description(bundle: TaskDescriptionBundle, *, max_chars_per_section: int = 360) -> dict[str, str]:
    summary: dict[str, str] = {}
    for kind, text in bundle.section_map.items():
        compact = " ".join(text.split())
        if len(compact) > max_chars_per_section:
            compact = compact[: max_chars_per_section - 3] + "..."
        summary[kind] = compact
    return summary


def _is_bboplace_task(task_spec: TaskSpec) -> bool:
    return task_spec.metadata.get("task_family") == "bboplace" or task_spec.name.startswith("bboplace")


def _display_task_name(task_spec: TaskSpec) -> str:
    display_name = task_spec.metadata.get("display_name")
    if isinstance(display_name, str) and display_name.strip():
        return display_name.strip()
    return task_spec.name


def _better_relation(task_spec: TaskSpec) -> str:
    return "lower" if task_spec.primary_objective.direction == ObjectiveDirection.MINIMIZE else "higher"


def _format_score_data(c_global: list[dict[str, Any]], *, task_spec: TaskSpec) -> str:
    direction = task_spec.primary_objective.direction
    compact_xy = _is_bboplace_task(task_spec) and is_compact_xy_space(task_spec.search_space)

    def sort_key(item: dict[str, Any]) -> tuple[int, float]:
        score = item.get("primary_objective")
        if isinstance(score, (int, float)):
            value = float(score)
            if direction == ObjectiveDirection.MAXIMIZE:
                value = -value
            return (0, value)
        return (1, 0.0)

    rows: list[dict[str, Any]] = []
    for item in sorted(c_global, key=sort_key):
        row = {
            "score": item.get("primary_objective"),
        }
        config = item.get("config", {})
        if compact_xy and isinstance(config, dict):
            row.update(compact_xy_config(task_spec.search_space, config))
        else:
            row["config"] = config
        if "trial_id" in item:
            row["trial_id"] = item["trial_id"]
        if "status" in item:
            row["status"] = item["status"]
        rows.append(row)
    if compact_xy:
        return json.dumps(rows, separators=(",", ":"), sort_keys=True)
    return json.dumps(rows, indent=2, sort_keys=True)


def _format_json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True)


def _search_space_block(task_spec: TaskSpec) -> str:
    if _is_bboplace_task(task_spec) and is_compact_xy_space(task_spec.search_space):
        n_macro = bboplace_macro_count(task_spec.search_space)
        x_param = task_spec.search_space["x_0"]
        y_param = task_spec.search_space["y_0"]
        return (
            f"Compact BBOPlace coordinate format: return exactly {n_macro} x coordinates and "
            f"{n_macro} y coordinates.\n"
            f"- x: list of {n_macro} numbers; bounds=[{x_param.low}, {x_param.high}]\n"
            f"- y: list of {n_macro} numbers; bounds=[{y_param.low}, {y_param.high}]\n"
            f"- x[i] maps to internal parameter x_i; y[i] maps to internal parameter y_i.\n"
            f"- Round coordinates to at most {COMPACT_COORD_DECIMALS} decimal places."
        )
    return render_search_space_block(task_spec.search_space, SearchSpaceRenderPolicy())


def _task_context_block(description: TaskDescriptionBundle) -> str:
    return render_task_context_block(description, TaskContextRenderPolicy(max_chars_per_section=1200))


def _objective_header(
    *,
    task_spec: TaskSpec,
    num_observed_trials: int | None = None,
    best_objective: float | None = None,
) -> str:
    lines = [
        f"Task name: {_display_task_name(task_spec)}",
        f"Primary objective: {task_spec.primary_objective.name}",
        f"Optimization direction: {task_spec.primary_objective.direction.value}",
        f"Better objective values are: {_better_relation(task_spec)}",
    ]
    if num_observed_trials is not None:
        lines.extend(
            [
                f"Evaluation budget: {task_spec.max_evaluations}",
                f"Current number of observed trials: {num_observed_trials}",
                f"Current best objective value: {json.dumps(best_objective)}",
            ]
        )
    return "\n".join(lines)


def _display_config(task_spec: TaskSpec, config: dict[str, Any]) -> Any:
    if _is_bboplace_task(task_spec) and is_compact_xy_space(task_spec.search_space):
        return compact_xy_config(task_spec.search_space, config)
    return config


def _candidate_output_contract(task_spec: TaskSpec, *, candidate_count: int | None = None) -> str:
    if _is_bboplace_task(task_spec) and is_compact_xy_space(task_spec.search_space):
        n_macro = bboplace_macro_count(task_spec.search_space)
        count_line = ""
        if candidate_count is not None:
            noun = "candidate" if candidate_count == 1 else "candidates"
            count_line = f"\nReturn exactly {candidate_count} {noun} in the candidates list.\n"
        return f"""Return ONLY a JSON object with a list of compact BBOPlace coordinate configurations called "candidates".
{count_line}

Each candidate must use this compact schema:
{{"x":[... exactly {n_macro} numbers ...],"y":[... exactly {n_macro} numbers ...]}}

For this task, n_macro = {n_macro}. Therefore every candidate must contain exactly {n_macro} x values and exactly {n_macro} y values, corresponding to macro indices 0 through {n_macro - 1}. A candidate with any other array length, including {n_macro - 1} or {n_macro + 1}, is invalid. Before final output, count both arrays once and ensure len(x) == len(y) == {n_macro}.

Round coordinates to at most {COMPACT_COORD_DECIMALS} decimal places. Use numeric JSON values, not strings.

Example shape for n_macro=2:
{{"candidates":[{{"x":[75.1234,154.5678],"y":[88.4321,166.8765]}}]}}"""
    return """Return ONLY a JSON object with a list of complete configurations called "candidates".

Example:
{ "candidates": [{ ... complete configuration ... }] }"""


def _explorer_system(task_spec: TaskSpec) -> str:
    if _is_bboplace_task(task_spec):
        return (
            "You are an expert chip macro-placement optimizer. Return only data that satisfies the requested "
            "JSON output format."
        )
    return (
        "You are an expert black-box optimization assistant. Return only data that satisfies the requested "
        "JSON output format."
    )


def _planner_system(task_spec: TaskSpec) -> str:
    if _is_bboplace_task(task_spec):
        return (
            "You are a prompt generator for a chip macro-placement optimization system. Follow the requested "
            "JSON output format exactly."
        )
    return (
        "You are a prompt generator for a black-box optimization system. Follow the requested JSON output "
        "format exactly."
    )


def _worker_system(task_spec: TaskSpec) -> str:
    if _is_bboplace_task(task_spec):
        return (
            "You are an expert chip macro-placement candidate generator operating in BBOPlace coordinate "
            "space. Follow the requested JSON output format exactly."
        )
    return (
        "You are an expert black-box optimization candidate generator. Follow the requested JSON output "
        "format exactly."
    )


def build_explorer_prompt(
    *,
    task_spec: TaskSpec,
    description: TaskDescriptionBundle,
    c_global: list[dict[str, Any]],
    best_objective: float | None,
    observed_trial_count: int | None = None,
) -> PromptBundle:
    search_space_summary = summarize_search_space(task_spec.search_space)
    description_summary = summarize_description(description)
    score_data = _format_score_data(c_global, task_spec=task_spec)
    score_heading = (
        "PLACEMENT-SCORE DATA (sorted low HPWL to high HPWL)"
        if _is_bboplace_task(task_spec)
        else "CONFIGURATION-SCORE DATA (sorted best to worst)"
    )
    task_intro = (
        "Generate compact macro-coordinate configurations that you reason are most likely to score LOWER HPWL than any in "
        "the provided PLACEMENT-SCORE DATA."
        if _is_bboplace_task(task_spec)
        else "Generate complete configurations that you reason are most likely to improve over any in the provided "
        "CONFIGURATION-SCORE DATA."
    )
    analysis_focus = (
        "What coordinate patterns correlate with low HPWL? Form 2-3 hypotheses about what the evaluator rewards."
        if _is_bboplace_task(task_spec)
        else "What configuration patterns correlate with strong objective values? Form 2-3 hypotheses about what the evaluator rewards."
    )
    generation_focus = (
        "- Push your hypotheses to their logical extreme for lower HPWL\n"
        "- Combine best coordinate/relative-placement patterns from multiple top scorers\n"
        "- Explore creative new placement layouts that are still valid under bounds\n"
        "- Preserve useful relative macro structure when appropriate, instead of perturbing every coordinate independently"
        if _is_bboplace_task(task_spec)
        else "- Push your hypotheses to their logical extreme for better objective values\n"
        "- Combine best features from multiple top scorers\n"
        "- Explore creative new valid configurations"
    )
    candidate_kind = "compact coordinate configurations" if _is_bboplace_task(task_spec) else "complete configurations"
    candidate_count_text = (
        "exactly 1 NEW compact coordinate configuration"
        if _is_bboplace_task(task_spec)
        else f"10-20 NEW {candidate_kind}"
    )
    explorer_candidate_count = 1 if _is_bboplace_task(task_spec) else None
    goal_line = (
        f"Propose one new macro-coordinate configuration that could BEAT the current best HPWL ({json.dumps(best_objective)}). Lower is better."
        if _is_bboplace_task(task_spec)
        else f"Propose new configurations that could beat the current best objective value ({json.dumps(best_objective)})."
    )
    system = _explorer_system(task_spec)
    user = f"""You are helping with a general black-box optimization task.

{task_intro}

## OPTIMIZATION OBJECTIVE
{_objective_header(task_spec=task_spec, num_observed_trials=observed_trial_count if observed_trial_count is not None else len(c_global), best_objective=best_objective)}

## SEARCH SPACE
Each candidate must satisfy the following search space:

{_search_space_block(task_spec)}

## TASK CONTEXT
{_task_context_block(description)}

## {score_heading}
{score_data}

## TASK
Think step-by-step:
1. Analyze the {score_heading.split(' (', 1)[0]}: {analysis_focus}
2. Generate: Propose {candidate_count_text} that:
{generation_focus}

## OUTPUT FORMAT
{_candidate_output_contract(task_spec, candidate_count=explorer_candidate_count)}

## GOAL
{goal_line}"""
    return PromptBundle(
        role="explorer",
        system=system,
        user=user,
        context={
            "task_spec": task_spec,
            "search_space": task_spec.search_space,
            "c_global": c_global,
            "best_objective": best_objective,
            "description_summary": description_summary,
        },
    )


def build_planner_prompt(
    *,
    task_spec: TaskSpec,
    description: TaskDescriptionBundle,
    c_global: list[dict[str, Any]],
    performance_stats: dict[str, Any],
    existing_tasks_summary: list[dict[str, Any]],
) -> PromptBundle:
    description_summary = summarize_description(description)
    best_objective = performance_stats.get("best_primary_objective")
    score_data = _format_score_data(c_global, task_spec=task_spec)
    score_heading = (
        "PLACEMENT-SCORE DATA (sorted low HPWL to high HPWL)"
        if _is_bboplace_task(task_spec)
        else "CONFIGURATION-SCORE DATA (sorted best to worst)"
    )
    domain_goal = (
        "is trying to find the lowest-HPWL BBOPlace configurations"
        if _is_bboplace_task(task_spec)
        else "is trying to find the best-scoring configurations"
    )
    worker_candidate_kind = (
        "compact coordinate configurations" if _is_bboplace_task(task_spec) else "complete configurations"
    )
    gradient_question = (
        "What small coordinate or relative-placement change caused one to score lower than another?"
        if _is_bboplace_task(task_spec)
        else "What small configuration change caused one to score better than another?"
    )
    contrast_question = (
        "What placement patterns appear in low-HPWL configurations but not high-HPWL configurations? Consider macro separation, clustering, axis ordering, center vs boundary bias, diagonal/row/column structure, and preservation of relative geometry."
        if _is_bboplace_task(task_spec)
        else "What parameter patterns appear in top configurations but not weak configurations?"
    )
    gap_question = (
        "What placement transformations have NOT been tried yet? What regions of the canvas remain unexplored? Which changes could escape local optima caused by MGO decoding?"
        if _is_bboplace_task(task_spec)
        else "What configuration changes have NOT been tried yet? What regions of the search space remain unexplored?"
    )
    evidence_note = (
        "If there are too few observed trials to compare score gradients, do not over-analyze nonexistent contrasts; create broadly useful initial tasks from the task context and search-space structure."
        if _is_bboplace_task(task_spec)
        else "If there are too few observed trials to compare score gradients, create broadly useful initial tasks from the task context and search-space structure."
    )
    creative_ideas = (
        "- global shifts of a promising placement pattern\n"
        "- local coordinate nudges around the current best seed\n"
        "- preserving macro order while changing spread\n"
        "- changing cluster compactness or separation\n"
        "- moving selected macros toward/away from canvas boundaries\n"
        "- swapping or mirroring x/y spatial patterns\n"
        "- diagonal, row-like, or column-like layouts\n"
        "- perturbing only a small subset of macros\n"
        "- combining coordinate patterns from multiple top scorers\n"
        "- probing MGO decoding boundaries with moderate, structured moves"
        if _is_bboplace_task(task_spec)
        else "- local refinements around strong configurations\n- broad valid moves into underexplored regions\n- feature mixing across top configurations\n- boundary or interaction probes"
    )
    system = _planner_system(task_spec)
    user = f"""You are a prompt generator for a black-box optimization system that {domain_goal}.

The downstream Worker is a smaller LLM. It will see only one task prompt, the search-space bounds, and one seed configuration. Your job is to create task prompts that help it generate useful new {worker_candidate_kind}.

## OPTIMIZATION OBJECTIVE
{_objective_header(task_spec=task_spec, num_observed_trials=int(performance_stats.get("n_trials", len(c_global))), best_objective=best_objective if isinstance(best_objective, (int, float)) else None)}

## SEARCH SPACE
Downstream Workers must generate {worker_candidate_kind} in this search space:

{_search_space_block(task_spec)}

## TASK CONTEXT
{_task_context_block(description)}

## {score_heading}
{score_data}

## TASK PERFORMANCE
success rate = objective improvements / attempts
{_format_json(performance_stats)}

## EXISTING TASKS
You can reuse by name or create new ones.
{_format_json(existing_tasks_summary)}

---

## YOUR ANALYSIS PROCESS
1. Study the score gradient when enough observations exist: Compare configurations with SIMILAR objective values. {gradient_question} These small differences are highly informative.

2. Best vs weak contrast: {contrast_question}

3. Identify gaps: {gap_question}

4. Evidence check: {evidence_note}

## YOUR GOAL
Generate task prompts that help the smaller Worker:
- EXPLOIT: Make targeted modifications based on patterns you observe
- EXPLORE: Try diverse, creative transformations to discover new promising regions

We are often stuck at local optima. To escape, we need BOTH:
- Smart exploitation of what seems to work
- Broad exploration of untried transformation types

## YOUR OUTPUT FORMAT
Return a JSON object with task names as keys and task descriptions as values.
- To REUSE an existing task: "TASK NAME": "USE EXISTING"
- To CREATE a new task: "NEW NAME": "TASK: ... HINTS: ..."

## GUIDELINES
1. Output 8-10 tasks total.
2. Include 2-3 exploitation tasks.
3. Include 2-3 exploration tasks.
4. Include 2-4 reliable existing tasks that have (>0%) success rates when such tasks exist.
5. If no existing task has >0% success rate, skip the reuse quota and create improved new tasks instead.
6. If a task has 0 successes, avoid it or create an improved version.
7. Keep new task descriptions concise.
8. New task names: SHORT, DESCRIPTIVE, ALL CAPS.
9. Every task must be actionable for a Worker that receives only one seed configuration.

## CREATIVE EXPLORATION IDEAS
Consider tasks involving:
{creative_ideas}"""
    return PromptBundle(
        role="planner",
        system=system,
        user=user,
        context={
            "task_spec": task_spec,
            "search_space": task_spec.search_space,
            "description_summary": description_summary,
            "performance_stats": performance_stats,
            "existing_tasks_summary": existing_tasks_summary,
            "c_global": c_global,
        },
    )


def build_worker_prompt(
    *,
    task_spec: TaskSpec,
    description: TaskDescriptionBundle | None = None,
    planner_task_name: str,
    planner_task_text: str,
    current_seed: dict[str, Any],
) -> PromptBundle:
    search_space_summary = summarize_search_space(task_spec.search_space)
    description = description or TaskDescriptionBundle.empty(task_id=task_spec.name)
    system = _worker_system(task_spec)
    generation_rules = (
        "1. Follow the PLANNER TASK closely.\n"
        "2. Generate 5-10 NEW compact coordinate configurations.\n"
        "3. Keep all coordinates within bounds.\n"
        "4. Do not return the seed unchanged.\n"
        "5. Prefer structured changes over independent random noise unless the task asks for broad exploration.\n"
        "6. Preserve useful relative macro geometry when exploiting a promising seed."
        if _is_bboplace_task(task_spec)
        else "1. Follow the PLANNER TASK closely.\n2. Generate 5-10 NEW complete configurations.\n3. Respect all search-space constraints.\n4. Do not return the seed unchanged.\n5. Prefer purposeful transformations over independent random noise."
    )
    user = f"""INPUT: You will be given a single seed configuration in the prompt.

## OPTIMIZATION OBJECTIVE
{_objective_header(task_spec=task_spec)}

## SEARCH SPACE
Each candidate must satisfy the following search space:

{_search_space_block(task_spec)}

## TASK CONTEXT
{_task_context_block(description)}

## SEED CONFIGURATION
{{"current_seed": {json.dumps(_display_config(task_spec, current_seed), separators=(",", ":"), sort_keys=True)}}}

## PLANNER TASK
{planner_task_text.strip()}

## GENERATION RULES
{generation_rules}

## OUTPUT FORMAT (REQUIRED)
{_candidate_output_contract(task_spec)}"""
    return PromptBundle(
        role="worker",
        system=system,
        user=user,
        context={
            "task_spec": task_spec,
            "search_space": task_spec.search_space,
            "planner_task_name": planner_task_name,
            "planner_task_text": planner_task_text,
            "current_seed": current_seed,
        },
    )


__all__ = [
    "PromptBundle",
    "build_explorer_prompt",
    "build_planner_prompt",
    "build_worker_prompt",
    "summarize_description",
    "summarize_search_space",
]
