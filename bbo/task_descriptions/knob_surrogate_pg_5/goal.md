# Goal

Minimize predicted 95th-percentile JOB latency on PostgreSQL.

The primary objective is:

```text
latency - minimize
```

A valid proposal provides one normalized value for every active knob.

The final result is the valid decoded configuration with the lowest predicted latency found within the evaluation budget.

Report:

- best predicted latency;
- reduction relative to the encoded PostgreSQL default;
- number of evaluations;
- unique decoded configurations;
- checkpoint checksum;
- decoded best configuration.
