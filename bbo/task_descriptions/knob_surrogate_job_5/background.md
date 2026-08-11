# Background

This benchmark represents database configuration tuning for the Join Order Benchmark on MySQL.

The Join Order Benchmark uses an IMDb-derived analytical workload containing complex multi-table joins. Unlike SYSBENCH, this is an online analytical processing workload. Its performance is measured by query latency rather than transaction throughput.

The source study uses the 95th-percentile workload latency as the optimization metric. Lower values are better.

This repository does not execute the workload during optimization. It loads the released random-forest surrogate for a five-knob JOB configuration space and predicts latency from a decoded physical MySQL configuration.

The active knob set must be verified against the checkpoint feature list and the original JOB importance-ranking artifact. The bundled JSON contains five variables, but its stored `important_rank` values are not `1` through `5`; therefore, the task description must not claim that the JSON alone proves the source paper's SHAP top-five set until that consistency check passes.
