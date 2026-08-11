# Constraints

The active feature set must match the validated `pg_5.joblib` checkpoint and consists of:

- `max_worker_processes`: integer;
- `shared_buffers`: integer;
- `wal_buffers`: integer;
- `checkpoint_completion_target`: real-valued;
- `backend_flush_after`: integer.

Each optimizer-facing coordinate lies in `[0, 1]` and is decoded according to `knobs_pg_top5.json`.

PostgreSQL settings can have special semantics:

- `wal_buffers = -1` enables automatic sizing;
- `max_worker_processes` is a process-capacity setting, not a direct request to launch that many workers;
- `checkpoint_completion_target` is a fraction and must remain within its valid range;
- integer values are quantized during decoding.

The optimizer may modify only the five active coordinates. It may not alter the workload, objective direction, checkpoint, physical bounds, or decoding process.

One surrogate prediction for one decoded configuration counts as one evaluation.
