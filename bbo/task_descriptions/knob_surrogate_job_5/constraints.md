# Constraints

The task uses exactly the ordered features exposed by `RF_JOB_5knob.joblib`.

Before the task is enabled, the following three objects must agree:

1. the checkpoint feature list;
2. the active entries in the knob metadata JSON;
3. the intended five-knob JOB selection derived from the source artifact.

If they disagree, initialization must fail.

The bundled `knobs_JOB_top5.json` file defines five variables, but the stored `important_rank` values are not `1` through `5`. Treat the filename as an asset name, not as sufficient proof of top-five rank semantics.

Each optimizer-facing coordinate must lie in `[0, 1]` and is decoded to its physical MySQL value.

The active set may contain integer, real, or categorical knobs. Categorical values are discrete, and integer values are quantized during decoding.

The optimizer may change only the five active knob coordinates. It may not alter the query workload, surrogate checkpoint, objective definition, decoding rules, or feature order.

One surrogate forward pass for one decoded configuration counts as one evaluation.
