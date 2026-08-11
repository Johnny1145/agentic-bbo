# Goal

Minimize predicted 95th-percentile query latency for the MySQL Join Order Benchmark.

The primary objective is:

```text
latency - minimize
```

A valid proposal specifies one normalized coordinate for every verified active knob.

The final result is the valid decoded configuration with the lowest predicted latency found within the evaluation budget.

Report:

- best predicted latency;
- reduction relative to the encoded database-default configuration;
- number of evaluations;
- number of unique decoded configurations;
- checkpoint checksum;
- validated feature list.
