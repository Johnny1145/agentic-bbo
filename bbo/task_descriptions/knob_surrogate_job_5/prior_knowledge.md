# Domain Prior Knowledge

JOB is an analytical workload dominated by complex joins. Its latency can be affected by several interacting database subsystems:

- memory available for joins, sorts, and temporary data;
- buffer and cache behavior;
- query optimizer cost and statistics settings;
- parallelism and worker limits;
- storage and checkpoint activity;
- logging and background writes.

Useful search principles:

- The objective is tail latency, so a configuration that improves some queries can still be poor if it severely slows a difficult query.
- Memory knobs can interact with query complexity and concurrent operators.
- Query-planning and statistics knobs can produce discontinuous behavior when the optimizer changes an execution plan.
- Categorical optimizer settings should be explored as discrete alternatives.
- Do not infer a universal monotonic direction for memory, parallelism, or cost parameters.
- Test local changes around promising configurations, but preserve evaluations for plan-changing or cross-subsystem alternatives.
- Compare every result with the encoded database-default configuration.
- Avoid duplicate decoded configurations.

The exact active knob semantics should be inserted only after the JOB five-knob artifact has been validated against the checkpoint and source provenance. This file intentionally avoids repeating unverified knob names from the current repository JSON.
