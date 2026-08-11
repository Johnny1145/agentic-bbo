# Background

This benchmark represents high-dimensional database configuration tuning for a transactional MySQL SYSBENCH workload.

The source study considers MySQL 5.7 and defines a large configuration space containing all tunable knobs retained by the benchmark construction procedure. The paper describes a 197-knob large space. The released surrogate artifact may expose a different effective feature count if a feature was filtered during artifact preparation. Therefore, the runtime checkpoint feature list, actual dimension, and any difference from the paper must be recorded explicitly.

The workload is SYSBENCH in read-write mode, and the performance objective is transaction throughput.

This repository evaluates configurations with a released surrogate rather than a live MySQL instance. Each normalized proposal is decoded into a mixed physical configuration containing integer, real-valued, and categorical database knobs.

The benchmark tests whether an optimizer can operate under high dimensionality, heterogeneous variable types, uneven variable importance, and interactions between database subsystems.
