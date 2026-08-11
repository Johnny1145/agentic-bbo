"""Small object-based API for running one benchmark task with one algorithm."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..algorithms import create_algorithm
from ..core import Algorithm, ExperimentConfig, Experimenter, JsonlMetricLogger, Task
from ..tasks import create_task


@dataclass(frozen=True)
class BenchmarkRunConfig:
    """Execution-only settings, independent of task and algorithm configuration."""

    output_dir: Path
    seed: int = 0
    resume: bool = False
    fail_fast_on_sanity: bool = True


def run_benchmark(
    task: Task,
    algorithm: Algorithm,
    *,
    config: BenchmarkRunConfig,
) -> dict[str, Any]:
    """Run already-constructed task and algorithm objects and persist a summary."""

    run_dir = Path(config.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "trials.jsonl"
    logger = JsonlMetricLogger(results_path)
    summary = Experimenter(
        task=task,
        algorithm=algorithm,
        logger_backend=logger,
        config=ExperimentConfig(
            seed=config.seed,
            resume=config.resume,
            fail_fast_on_sanity=config.fail_fast_on_sanity,
        ),
    ).run()
    payload = {
        "task_name": summary.task_name,
        "algorithm_name": summary.algorithm_name,
        "seed": summary.seed,
        "n_completed": summary.n_completed,
        "best_primary_objective": summary.best_primary_objective,
        "total_eval_time": summary.total_eval_time,
        "stop_reason": summary.stop_reason,
        "run_dir": str(run_dir),
        "results_jsonl": str(results_path),
        "summary_json": str(run_dir / "summary.json"),
        "trial_count": len(logger.load_records()),
        "final_evaluation": (
            None
            if summary.final_evaluation is None
            else summary.final_evaluation.to_dict()
        ),
        "incumbents": [
            {
                "config": item.config,
                "score": item.score,
                "objectives": item.objectives,
                "trial_id": item.trial_id,
                "metadata": item.metadata,
            }
            for item in summary.incumbents
        ],
        "internal_artifacts": getattr(algorithm, "artifact_paths", {}),
        "role_model_routes": getattr(algorithm, "routing_table", {}),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def run_named_benchmark(
    task_name: str,
    algorithm_name: str,
    *,
    config: BenchmarkRunConfig,
    task_kwargs: dict[str, Any] | None = None,
    algorithm_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct a registry task and algorithm, then run them."""

    task = create_task(task_name, seed=config.seed, **dict(task_kwargs or {}))
    algorithm = create_algorithm(algorithm_name, **dict(algorithm_kwargs or {}))
    return run_benchmark(task, algorithm, config=config)


__all__ = ["BenchmarkRunConfig", "run_benchmark", "run_named_benchmark"]
