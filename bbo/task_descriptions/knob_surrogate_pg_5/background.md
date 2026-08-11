# Background

This benchmark represents PostgreSQL configuration tuning for the Join Order Benchmark.

The source study evaluates JOB on PostgreSQL 12.7. JOB is an analytical workload containing complex joins, and the optimization metric is the 95th-percentile query latency.

The source authors collect PostgreSQL measurements, select five important knobs with SHAP, and fit a random-forest surrogate over the resulting small configuration space.

This repository evaluates the released surrogate rather than a live PostgreSQL instance. The optimizer proposes normalized coordinates, which are decoded into physical PostgreSQL settings before surrogate prediction.

The objective is latency minimization. The task must not rename the response as throughput or maximize the raw surrogate output unless the checkpoint is explicitly documented as predicting a transformed negative-latency score.
