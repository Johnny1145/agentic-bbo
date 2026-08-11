# Constraints

The search space contains exactly the ordered features exposed by the validated surrogate checkpoint.

Each submitted value must lie in `[0, 1]`. The evaluator decodes it using the physical type and range recorded in `knobs_SYSBENCH_top5.json`.

The active variables are expected to include:

- `tmp_table_size`: integer;
- `max_heap_table_size`: integer;
- `query_prealloc_size`: integer;
- `innodb_thread_concurrency`: integer;
- `innodb_doublewrite`: categorical.

The runtime checkpoint feature order is authoritative. Initialization must fail if this list does not match the checkpoint.

Important representation rules:

- integer knobs are quantized after decoding;
- categorical values are discrete and must not be interpreted as naturally ordered continuous quantities;
- two nearby normalized values can decode to the same integer or categorical value;
- all required variables must be supplied;
- unknown variables are invalid;
- values outside `[0, 1]` are invalid.

The optimizer may change only the active knob coordinates. It may not modify the surrogate, workload identity, checkpoint, knob bounds, objective definition, or decoding function.

The evaluation budget is defined by `TaskSpec.max_evaluations`. One call to the surrogate with one decoded configuration counts as one evaluation.
