# Goal

Maximize predicted SYSBENCH transaction throughput over the full active MySQL configuration space.

The primary objective is:

```text
throughput - maximize
```

The valid configuration schema is determined by the checkpoint feature order.

The task ends when the evaluation budget is exhausted. The final result is the valid decoded configuration with the highest predicted throughput.

In addition to the best objective value, report:

- improvement over the encoded database-default configuration;
- unique decoded configurations evaluated;
- duplicate-decoding rate;
- fraction of active variables changed from default;
- search coverage by knob subsystem;
- checkpoint feature count and checksum.
