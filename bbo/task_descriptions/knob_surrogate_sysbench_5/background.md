# Background

This benchmark represents database configuration tuning for a transactional MySQL workload.

The source study evaluates MySQL 5.7 using SYSBENCH in read-write mode. The workload is an online transaction processing workload in which database configuration choices affect transaction throughput through memory allocation, concurrency control, logging, and storage behavior.

This repository does not start MySQL or execute SYSBENCH during an evaluation. Instead, it loads the released random-forest surrogate associated with the five-knob SYSBENCH configuration space. The surrogate predicts workload throughput from a decoded physical knob configuration.

The optimizer proposes one normalized coordinate for each active knob. The task converts these coordinates into physical MySQL values using the task-specific knob metadata and evaluates the resulting vector with the surrogate.

The surrogate is an approximation of measurements collected in the source experimental environment. A high predicted value means that the model expects higher throughput in that environment; it does not guarantee the same behavior on another MySQL deployment.

Source benchmark:

- Xinyi Zhang et al., "Facilitating Database Tuning with Hyper-Parameter Optimization: A Comprehensive Experimental Evaluation."
- Artifact repository: `PKU-DAIR/KnobsTuningEA`.
- Runtime checkpoint: `RF_SYSBENCH_5knob.joblib`.
