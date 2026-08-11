#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


LEAK_STRINGS = ("bbob_f01_d10", "bbob_problem_id", "bbob_function_id", "problem_key", "COCO/BBOB")
AGENT_VISIBLE_FILES = (
    "agent_workspace/task.md",
    "agent_workspace/history.jsonl",
    "agent_calls.jsonl",
    "agent_prompts.jsonl",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate restricted-prior exp1 Nanobot no-skill outputs.")
    parser.add_argument("run_root", type=Path, help="Run root passed to scripts/sglang/exp1/nanobot_no_skill.sh.")
    parser.add_argument("--expected-evaluations", type=int, default=20)
    parser.add_argument("--allow-subset", action="store_true", help="Allow fewer planned cases, but require at least one.")
    args = parser.parse_args(argv)

    run_root = args.run_root.expanduser().resolve()
    summary_path = run_root / "benchmark_summary.json"
    errors: list[str] = []
    if not summary_path.exists():
        errors.append(f"missing benchmark summary: {summary_path}")
        return finish(errors, checked=0, run_root=run_root)

    payload = read_json(summary_path, errors)
    if payload is None:
        return finish(errors, checked=0, run_root=run_root)
    failures = payload.get("failures") or []
    if failures:
        errors.append(f"{summary_path}: failures={len(failures)}")
    planned = payload.get("planned_cases") or []
    results = payload.get("results") or []
    if not results:
        errors.append(f"{summary_path}: no results")
    if planned and len(results) != len(planned):
        errors.append(f"{summary_path}: results count {len(results)} does not match planned count {len(planned)}")
    if not args.allow_subset and planned and len(results) != len(planned):
        errors.append(f"{summary_path}: incomplete planned case set")

    for index, result in enumerate(results):
        validate_result(summary_path, index, result, int(args.expected_evaluations), errors)
    return finish(errors, checked=len(results), run_root=run_root)


def read_json(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{path}: failed to read JSON: {exc}")
        return None
    if not isinstance(payload, dict):
        errors.append(f"{path}: expected JSON object")
        return None
    return payload


def validate_result(
    summary_path: Path,
    index: int,
    result: dict[str, Any],
    expected_evaluations: int,
    errors: list[str],
) -> None:
    prefix = f"{summary_path}: result[{index}]"
    metadata = result.get("benchmark_metadata") or {}
    if metadata.get("skill_mode") != "no-skill":
        errors.append(f"{prefix}: skill_mode={metadata.get('skill_mode')!r}, expected 'no-skill'")
    if metadata.get("exposure_policy") != "restricted-prior":
        errors.append(f"{prefix}: exposure_policy={metadata.get('exposure_policy')!r}, expected 'restricted-prior'")
    expect_int(result, "n_completed", expected_evaluations, prefix, errors)
    expect_int(result, "trial_count", expected_evaluations, prefix, errors)
    metrics = result.get("logger_summary") or {}
    expect_int(metrics, "successful_trials", expected_evaluations, prefix, errors)
    expect_int(metrics, "failed_trials", 0, prefix, errors)
    expect_int(metrics, "invalid_trials", 0, prefix, errors)
    usage = result.get("tool_usage_summary") or {}
    if usage.get("skill_read_counts"):
        errors.append(f"{prefix}: no-skill run unexpectedly read skills: {usage.get('skill_read_counts')}")
    if usage.get("accepted_skill_counts"):
        errors.append(f"{prefix}: no-skill run unexpectedly accepted skills: {usage.get('accepted_skill_counts')}")

    run_dir_raw = result.get("run_dir")
    if not run_dir_raw:
        errors.append(f"{prefix}: missing run_dir")
        return
    run_dir = Path(str(run_dir_raw))
    if not run_dir.exists():
        errors.append(f"{prefix}: run_dir does not exist: {run_dir}")
        return
    skill_files = sorted((run_dir / "agent_workspace" / "skills").glob("*/SKILL.md"))
    if skill_files:
        errors.append(f"{prefix}: no-skill agent workspace contains SKILL.md files: {skill_files[:3]}")
    validate_trials(run_dir / "trials.jsonl", expected_evaluations, errors)
    validate_no_leaks(run_dir, errors)


def validate_trials(path: Path, expected_evaluations: int, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"missing trials file: {path}")
        return
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != expected_evaluations:
        errors.append(f"{path}: expected {expected_evaluations} rows, found {len(rows)}")
    for line_no, line in enumerate(rows, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path}:{line_no}: invalid JSONL row: {exc}")
            continue
        if row.get("status") != "success":
            errors.append(f"{path}:{line_no}: status={row.get('status')!r}, expected 'success'")


def validate_no_leaks(run_dir: Path, errors: list[str]) -> None:
    for relative in AGENT_VISIBLE_FILES:
        path = run_dir / relative
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for leak in LEAK_STRINGS:
            if leak in text:
                errors.append(f"{path}: contains leaked string {leak!r}")


def expect_int(payload: dict[str, Any], key: str, expected: int, prefix: str, errors: list[str]) -> None:
    if key not in payload:
        errors.append(f"{prefix}: missing {key}")
        return
    try:
        value = int(payload[key])
    except Exception:  # noqa: BLE001
        errors.append(f"{prefix}: {key}={payload[key]!r} is not an int")
        return
    if value != expected:
        errors.append(f"{prefix}: {key}={value}, expected {expected}")


def finish(errors: list[str], *, checked: int, run_root: Path) -> int:
    if errors:
        print(f"FAIL: checked {checked} exp1 no-skill result(s) under {run_root}")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"PASS: checked {checked} exp1 no-skill result(s) under {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
