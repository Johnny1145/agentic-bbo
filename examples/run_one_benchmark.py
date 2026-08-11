"""Minimal editable benchmark run.

Change TASK_NAME, ALGORITHM_NAME, and the two kwargs dictionaries.
"""

from pathlib import Path

from bbo.benchmark import BenchmarkRunConfig, run_named_benchmark


TASK_NAME = "bbob_f01_d10"
ALGORITHM_NAME = "agentic_codex"

TASK_KWARGS = {
    "max_evaluations": 20,
}

ALGORITHM_KWARGS = {
    "tool_mode": "workspace_json",
    "prompt_profile": "native_tools",
    "enabled_tool_names": (
        "get_trial_history",
        "get_history_overview",
        "validate_candidate",
    ),
}


if __name__ == "__main__":
    result = run_named_benchmark(
        TASK_NAME,
        ALGORITHM_NAME,
        config=BenchmarkRunConfig(
            output_dir=Path("results") / TASK_NAME / ALGORITHM_NAME / "seed_0",
            seed=0,
        ),
        task_kwargs=TASK_KWARGS,
        algorithm_kwargs=ALGORITHM_KWARGS,
    )
    print(result["summary_json"] if "summary_json" in result else result["results_jsonl"])
