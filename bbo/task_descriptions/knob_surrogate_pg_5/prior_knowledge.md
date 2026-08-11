# Domain Prior Knowledge

The active knobs cover concurrency, memory, write-ahead logging, and checkpoint behavior.

- `max_worker_processes` sets an upper bound on PostgreSQL background processes.
- `shared_buffers` controls the amount of shared memory used for database pages.
- `wal_buffers` controls shared memory used for WAL records before they are written.
- `checkpoint_completion_target` controls how checkpoint writes are spread across the checkpoint interval.
- `backend_flush_after` controls when previously issued backend writes are flushed to storage.

Useful search principles:

- Memory values should be interpreted jointly rather than as independent percentages.
- Automatic or sentinel settings such as `wal_buffers = -1` should be treated as distinct semantic choices.
- Concurrency capacity can help or hurt depending on CPU, memory, and workload pressure.
- Checkpoint settings trade short I/O bursts against more continuous background write activity.
- Test interactions between `shared_buffers` and `wal_buffers`.
- Test interactions between checkpoint pacing and flush behavior.
- Do not assume that increasing any one resource limit monotonically reduces latency.
- Use the PostgreSQL default configuration as an explicit reference point.
- Avoid normalized proposals that decode to the same physical integer settings.

Exact values reported as favorable in the source paper are intentionally omitted from the default prior.
