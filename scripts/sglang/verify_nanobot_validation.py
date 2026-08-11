#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any, Iterable, Sequence


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_DIR.parents[1]
CONFIG_DIR = SCRIPT_DIR / "configs"
FAMILIES = ("synthetic", "molecule", "bboplace")
MODES = ("no-skill", "skill")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate 5-eval SGLang Nanobot no-skill/skill run outputs.")
    parser.add_argument("run_root", type=Path, help="RUN_ROOT used by the nanobot validation shell scripts.")
    parser.add_argument("--family", action="append", choices=FAMILIES, help="Family to validate. Repeatable.")
    parser.add_argument("--mode", action="append", choices=MODES, help="Skill mode to validate. Repeatable.")
    parser.add_argument("--seeds", default="1", help="Comma/space-separated expected seeds. Default: 1.")
    parser.add_argument("--expected-evaluations", type=int, default=5)
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="Do not require every configured task to be present. Useful for smoke runs.",
    )
    args = parser.parse_args(argv)

    run_root = args.run_root.expanduser().resolve()
    families = tuple(args.family or FAMILIES)
    modes = tuple(args.mode or MODES)
    seeds = split_words(args.seeds)
    if not seeds:
        parser.error("--seeds must contain at least one seed")

    errors: list[str] = []
    checked = 0
    for family in families:
        expected_task_count = len(config_tasks(family))
        expected_count = expected_task_count * len(seeds)
        for mode in modes:
            mode_dir = run_root / family / mode
            result = validate_mode_dir(
                mode_dir=mode_dir,
                family=family,
                mode=mode,
                expected_count=expected_count,
                expected_evaluations=int(args.expected_evaluations),
                allow_subset=bool(args.allow_subset),
            )
            checked += result["checked"]
            errors.extend(result["errors"])

    if errors:
        print(f"FAIL: checked {checked} run summaries/records under {run_root}")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"PASS: checked {checked} run summaries/records under {run_root}")
    return 0


def validate_mode_dir(
    *,
    mode_dir: Path,
    family: str,
    mode: str,
    expected_count: int,
    expected_evaluations: int,
    allow_subset: bool,
) -> dict[str, Any]:
    errors: list[str] = []
    checked = 0
    label = f"{family}/{mode}"
    if not mode_dir.exists():
        errors.append(f"{label}: missing directory {mode_dir}")
        return {"checked": checked, "errors": errors}

    aggregate_counts: list[int] = []
    for path in sorted(mode_dir.rglob("benchmark_summary.json")):
        payload = read_json(path, errors)
        if payload is None:
            continue
        aggregate_counts.append(validate_benchmark_summary(path, payload, mode, expected_evaluations, errors))
        checked += len(payload.get("results") or [])

    for path in sorted(mode_dir.rglob("results.json")):
        payload = read_json(path, errors)
        if payload is None or not isinstance(payload.get("records"), list):
            continue
        aggregate_counts.append(validate_records_json(path, payload, family, expected_evaluations, errors))
        checked += len(payload.get("records") or [])

    run_summaries = [
        path
        for path in sorted(mode_dir.rglob("summary.json"))
        if path.name == "summary.json" and not path.parts[-2:] == ("summary", "summary.json")
    ]
    for path in run_summaries:
        payload = read_json(path, errors)
        if payload is None or not looks_like_run_summary(payload):
            continue
        validate_run_summary(path, payload, family, expected_evaluations, errors)

    observed_count = max(aggregate_counts, default=0)
    if observed_count == 0:
        observed_count = sum(1 for path in run_summaries if read_json(path, []) and looks_like_run_summary(read_json(path, []) or {}))

    if allow_subset:
        if observed_count <= 0:
            errors.append(f"{label}: no completed run records found")
    elif observed_count != expected_count:
        errors.append(f"{label}: expected {expected_count} run records, found {observed_count}")

    return {"checked": checked or observed_count, "errors": errors}


def validate_benchmark_summary(
    path: Path,
    payload: dict[str, Any],
    mode: str,
    expected_evaluations: int,
    errors: list[str],
) -> int:
    failures = payload.get("failures") or []
    if failures:
        errors.append(f"{path}: benchmark_summary failures={len(failures)}")
    planned = payload.get("planned_cases") or []
    results = payload.get("results") or []
    if planned and len(results) != len(planned):
        errors.append(f"{path}: results count {len(results)} does not match planned count {len(planned)}")
    for index, result in enumerate(results):
        metadata = result.get("benchmark_metadata") or {}
        result_mode = metadata.get("skill_mode")
        if result_mode and result_mode != mode:
            errors.append(f"{path}: result {index} has skill_mode={result_mode!r}, expected {mode!r}")
        validate_run_summary(path, result, "synthetic", expected_evaluations, errors, label_suffix=f" result[{index}]")
    return len(results)


def validate_records_json(
    path: Path,
    payload: dict[str, Any],
    family: str,
    expected_evaluations: int,
    errors: list[str],
) -> int:
    records = payload.get("records") or []
    planned = (payload.get("environment") or {}).get("planned_run_count")
    if planned is not None and len(records) != int(planned):
        errors.append(f"{path}: records count {len(records)} does not match planned_run_count {planned}")
    for index, record in enumerate(records):
        prefix = f"{path}: record[{index}]"
        if record.get("ok") is not True:
            errors.append(f"{prefix}: ok is not true")
        if str(record.get("status") or "") != "ok":
            errors.append(f"{prefix}: status={record.get('status')!r}, expected 'ok'")
        expect_int(record, "total_eval_budget", expected_evaluations, prefix, errors)
        expect_int(record, "n_completed", expected_evaluations, prefix, errors)
        expect_int(record, "successful_trials", expected_evaluations, prefix, errors)
        expect_int(record, "total_trials", expected_evaluations, prefix, errors)
        expect_int(record, "failed_trials", 0, prefix, errors)
        expect_int(record, "invalid_trials", 0, prefix, errors)
        if family == "molecule":
            expect_int(record, "valid_smiles_trials", expected_evaluations, prefix, errors)
            expect_int(record, "invalid_smiles_trials", 0, prefix, errors)
    return len(records)


def validate_run_summary(
    path: Path,
    payload: dict[str, Any],
    family: str,
    expected_evaluations: int,
    errors: list[str],
    *,
    label_suffix: str = "",
) -> None:
    prefix = f"{path}{label_suffix}"
    metrics = payload.get("logger_summary") or payload.get("metrics") or {}
    expect_int_any(payload, metrics, ("n_completed",), expected_evaluations, prefix, errors)
    expect_int_any(payload, metrics, ("trial_count", "total_trials"), expected_evaluations, prefix, errors)
    expect_int_any(payload, metrics, ("successful_trials",), expected_evaluations, prefix, errors)
    expect_int_any(payload, metrics, ("failed_trials",), 0, prefix, errors)
    expect_int_any(payload, metrics, ("invalid_trials",), 0, prefix, errors)
    budget = payload.get("budget")
    if isinstance(budget, dict):
        expect_int(budget, "total_evaluation_budget", expected_evaluations, prefix, errors)
        expect_int(budget, "n_initial_points", 0, prefix, errors)
    if family == "molecule":
        expect_int_any(payload, metrics, ("valid_smiles_trials",), expected_evaluations, prefix, errors)
        expect_int_any(payload, metrics, ("invalid_smiles_trials",), 0, prefix, errors)
    validate_trials_jsonl(path, payload.get("results_jsonl"), family, expected_evaluations, errors)


def validate_trials_jsonl(
    summary_path: Path,
    raw_path: Any,
    family: str,
    expected_evaluations: int,
    errors: list[str],
) -> None:
    if not raw_path:
        errors.append(f"{summary_path}: missing results_jsonl")
        return
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = (summary_path.parent / path).resolve()
    if not path.exists():
        errors.append(f"{summary_path}: results_jsonl does not exist: {path}")
        return
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                errors.append(f"{path}:{line_no}: invalid JSONL row: {exc}")
                continue
            if isinstance(item, dict):
                rows.append(item)
            else:
                errors.append(f"{path}:{line_no}: row is not an object")
    if len(rows) != expected_evaluations:
        errors.append(f"{path}: expected {expected_evaluations} rows, found {len(rows)}")
    for index, row in enumerate(rows):
        status = row.get("status")
        if status != "success":
            errors.append(f"{path}: row {index} status={status!r}, expected 'success'")
        metadata = row.get("metadata") or {}
        if family == "molecule" and metadata.get("valid_smiles") is False:
            errors.append(f"{path}: row {index} has valid_smiles=false")


def looks_like_run_summary(payload: dict[str, Any]) -> bool:
    return any(key in payload for key in ("logger_summary", "metrics", "results_jsonl", "trial_count", "budget"))


def expect_int_any(
    payload: dict[str, Any],
    metrics: dict[str, Any],
    keys: Iterable[str],
    expected: int,
    prefix: str,
    errors: list[str],
) -> None:
    for key in keys:
        if key in payload:
            expect_int(payload, key, expected, prefix, errors)
            return
        if key in metrics:
            expect_int(metrics, key, expected, prefix, errors)
            return
    errors.append(f"{prefix}: missing any of {tuple(keys)}")


def expect_int(payload: dict[str, Any], key: str, expected: int, prefix: str, errors: list[str]) -> None:
    try:
        value = int(payload.get(key))
    except (TypeError, ValueError):
        errors.append(f"{prefix}: {key}={payload.get(key)!r}, expected {expected}")
        return
    if value != expected:
        errors.append(f"{prefix}: {key}={value}, expected {expected}")


def read_json(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{path}: could not read JSON: {exc}")
        return None
    if not isinstance(payload, dict):
        errors.append(f"{path}: JSON payload is not an object")
        return None
    return payload


def config_tasks(family: str) -> list[str]:
    path = CONFIG_DIR / f"{family}.toml"
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    return [str(item) for item in config.get("tasks", [])]


def split_words(raw: str) -> tuple[str, ...]:
    return tuple(item for item in raw.replace(",", " ").split() if item)


if __name__ == "__main__":
    raise SystemExit(main())
