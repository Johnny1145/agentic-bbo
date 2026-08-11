# Goal

Minimize predicted 95th-percentile JOB latency on PostgreSQL over the validated twenty-knob configuration space.

The primary objective is:

```text
latency - minimize
```

The task ends when the evaluation budget is exhausted. The final result is the valid decoded configuration with the lowest predicted latency.

Report:

- best predicted latency;
- reduction relative to the encoded PostgreSQL default;
- unique decoded configurations;
- duplicate-decoding rate;
- number of knobs changed from default;
- subsystem coverage;
- validated feature list;
- checkpoint checksum.
