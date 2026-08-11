# Evaluation Protocol

- Primary metric: surrogate-predicted 95th-percentile JOB query latency on PostgreSQL.
- Objective direction: minimize.
- Evaluation implementation: `y = model.predict(decode(x))`, where `x` is the normalized suggestion vector and `decode` maps `[0, 1]^d` to physical PostgreSQL settings.
- One surrogate prediction for one decoded configuration counts as one evaluation.
- Report decoded best configuration, checkpoint checksum, and unique decoded configurations.
