#!/usr/bin/env python3
"""Run the real default-configuration acceptance check for all 25 HPO tasks."""

from __future__ import annotations

import json
import math
import time

from bbo.core import TrialSuggestion
from bbo.tasks import HPO_TASK_IDS, create_task


def main() -> int:
    rows: list[dict[str, object]] = []
    failed = False
    started = time.perf_counter()
    for name in HPO_TASK_IDS:
        task = create_task(name, max_evaluations=1, seed=0)
        sanity = task.sanity_check()
        result = task.evaluate(TrialSuggestion(task.spec.search_space.defaults()))
        objectives_finite = all(math.isfinite(value) for value in result.objectives.values())
        ok = sanity.ok and result.success and objectives_finite
        failed = failed or not ok
        row = {
            "task": name,
            "ok": ok,
            "sanity_errors": [issue.message for issue in sanity.errors],
            "status": result.status.value,
            "objectives": result.objectives,
            "elapsed_seconds": result.elapsed_seconds,
            "error_type": result.error_type,
            "error_message": result.error_message,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
    summary = {
        "task_count": len(rows),
        "successful": sum(bool(row["ok"]) for row in rows),
        "failed": sum(not bool(row["ok"]) for row in rows),
        "elapsed_seconds": time.perf_counter() - started,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
