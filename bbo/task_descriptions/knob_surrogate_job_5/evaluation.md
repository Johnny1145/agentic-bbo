# Evaluation Protocol

- Primary metric: surrogate-predicted 95th-percentile JOB query latency.
- Objective direction: minimize.
- Evaluation implementation: `y = model.predict(decode(x))`, where `x` is the normalized suggestion vector and `decode` maps `[0, 1]^d` to physical MySQL knob values.
- One surrogate prediction for one decoded configuration counts as one evaluation.
- Report the validated feature list and checkpoint checksum with results.
