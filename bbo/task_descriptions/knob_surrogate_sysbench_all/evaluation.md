# Evaluation Protocol

- Primary metric: surrogate-predicted SYSBENCH throughput.
- Objective direction: maximize.
- Evaluation implementation: `y = model.predict(decode(x))`, where `x` is the normalized suggestion vector and `decode` maps `[0, 1]^d` to physical MySQL knob values.
- One surrogate prediction for one decoded configuration counts as one evaluation.
- Report the actual checkpoint feature count; do not infer it from the task name.
