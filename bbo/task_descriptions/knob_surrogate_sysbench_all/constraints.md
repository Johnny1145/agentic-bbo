# Constraints

The active search space is the complete ordered feature list exposed by the validated SYSBENCH full-space checkpoint.

Do not infer the dimension from the task name or JSON filename. At initialization, record:

- the paper-defined large-space size;
- the checkpoint feature count;
- the knob JSON entry count;
- the exact checkpoint feature order;
- any excluded or additional features.

Each optimizer-facing coordinate lies in `[0, 1]` and is decoded according to its physical knob metadata.

The physical space may contain:

- integer knobs;
- real-valued knobs;
- binary categorical knobs;
- multi-category knobs;
- values with special sentinel semantics.

Categorical encodings are not ordinal. Integer coordinates can contain large plateaus because many normalized values may decode to the same physical value.

The optimizer may modify only the active knob coordinates. Changing the workload, surrogate artifact, feature order, bounds, decoding logic, or objective direction is forbidden.

One surrogate prediction for one decoded configuration counts as one evaluation. Duplicate decoded configurations also consume budget unless the framework rejects them before evaluation.
