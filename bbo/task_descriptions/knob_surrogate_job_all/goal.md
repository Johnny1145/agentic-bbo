# Goal

Minimize predicted 95th-percentile query latency for JOB over the full active MySQL configuration space.

The primary objective is:

```text
latency - minimize
```

The best valid decoded configuration found before exhausting `TaskSpec.max_evaluations` is returned.

Report:

- best predicted latency;
- reduction relative to the encoded database-default configuration;
- unique decoded configurations;
- duplicate-decoding rate;
- number of changed knobs relative to default;
- coverage of memory, optimizer, concurrency, logging, and I/O knob groups;
- checkpoint feature count and checksum.
