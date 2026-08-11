# Evaluation Protocol

- Primary metric: surrogate-predicted 95th-percentile JOB query latency.
- Objective direction: minimize.
- Evaluation implementation: `y = model.predict(decode(x))`, where `x` is the normalized suggestion vector and `decode` maps `[0, 1]^d` to physical MySQL knob values.
- One surrogate prediction for one decoded configuration counts as one evaluation.
- Report the actual checkpoint feature count; do not infer it from the task name.
