# Background

This benchmark represents medium-dimensional PostgreSQL configuration tuning for the Join Order Benchmark.

The source study evaluates JOB on PostgreSQL 12.7, uses SHAP-ranked PostgreSQL knob metadata to define selected configuration spaces, and fits random-forest surrogates over those spaces.

The workload is analytical and the optimization target is 95th-percentile query latency. Lower values are better.

The task uses an offline surrogate. It does not start PostgreSQL or execute JOB during an evaluation.

Before this task is enabled, the twenty active knob definitions must be regenerated or validated against both the released checkpoint feature list and the intended source selection. The bundled JSON contains twenty variables, but its stored `important_rank` values span beyond the first twenty ranks; therefore, it must not be described as the source paper's top-twenty ranked space without an explicit provenance explanation.
