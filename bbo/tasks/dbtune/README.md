# `bbo.tasks.dbtune` — database knob task family

This package's active task surface is limited to the six HTTP sklearn-surrogate
database-knob benchmarks. A small `registry.py` documents the active ids, and
co-located assets live under clear subfolders.

## Layout

| Area | Role |
|------|------|
| `registry.py` | Re-exports catalog metadata for the active surrogate-service task ids. |
| `catalog.py` | Offline `*.joblib` benchmark specs (`SURROGATE_BENCHMARKS`). |
| `http_mariadb_specs.py` | Legacy real **MySQL 5.7 + sysbench** specs; these ids are not part of the active dbtune task set. |
| `http_mariadb_task.py` | Legacy task implementation: `HttpDatabaseKnobTask`; not exported or registered as an active task. |
| `offline_surrogate_task.py` | In-process sklearn surrogate: `SurrogateKnobTask`. |
| `http_surrogate_task.py` | Remote evaluator service (Python 3.7 Docker) for the six active surrogate tasks. |
| `cli_*.py` | Hooks for `bbo.tasks.registry` / `python -m bbo.run`; active dbtune registration is limited to `knob_http_surrogate_*`. |
| `assets/` | Shared `knobs_*.json` and downloaded `*.joblib` (large files are not committed; see `assets/README.md`). |
| `docker_mariadb/` | Image for the **live** MySQL 5.7 + sysbench evaluator (Flask API, legacy path name kept for compatibility). |
| `docker_surrogate/` | Image for **offline** sklearn inference via JSON (isolated old numpy/sklearn). |
| `gen_task_markdown.py` | One-off generator for `bbo/task_descriptions/knob_http_mariadb_sysbench_*/` packs. |

## Import surface

User code should use the stable active task ids from `bbo.tasks` / `bbo.tasks.registry`, e.g.
`create_task("knob_http_surrogate_sysbench_5")`. Only these six dbtune ids are registered:

```text
knob_http_surrogate_sysbench_5
knob_http_surrogate_sysbench_all
knob_http_surrogate_job_5
knob_http_surrogate_job_all
knob_http_surrogate_pg_5
knob_http_surrogate_pg_20
```

In-process `create_surrogate_knob_task("knob_surrogate_sysbench_5", ...)` remains available but is not registered on
`python -m bbo.run`. Legacy `knob_http_mariadb_sysbench_*` ids are not registered as active dbtune tasks. For a
**direct** import, prefer:

```python
from bbo.tasks.dbtune import create_surrogate_knob_task
```

## See also

- `bbo/tasks/scientific/` — same “family + registry + data/” pattern for non-database scientific benchmarks.
