# Constraints

The active search variables are exactly the ordered features exposed by the validated `pg_20.joblib` checkpoint.

Initialization must verify agreement among:

1. the checkpoint feature list;
2. the knob metadata JSON;
3. the intended PostgreSQL twenty-knob selection and its provenance artifact.

The bundled `knobs_pg_top20.json` file defines twenty variables, but the stored `important_rank` values are not simply ranks `1` through `20`. Treat the dimension as twenty and validate top-k provenance separately.

Each coordinate must lie in `[0, 1]` and is decoded to its physical PostgreSQL value.

The search space can contain:

- integer settings;
- real-valued settings;
- categorical settings;
- sentinel values with automatic behavior;
- settings that require restart or reload in a live PostgreSQL deployment.

Although this task is surrogate-only, the physical variable semantics must be preserved in metadata.

The optimizer may modify only the twenty active coordinates. It may not modify the workload, checkpoint, decoding, objective direction, or feature order.

One surrogate forward pass for one decoded configuration counts as one evaluation.
