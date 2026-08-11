# Goal

Maximize predicted SYSBENCH transaction throughput.

The primary objective is:

```text
throughput - maximize
```

A valid submission specifies one normalized value in `[0, 1]` for every active knob.

One evaluation consists of:

1. validating the submitted normalized configuration;
2. decoding it into a physical MySQL knob vector;
3. applying the released random-forest surrogate once;
4. returning the predicted throughput.

The best valid configuration found within the evaluation budget is the final result.

For reporting, include:

- best predicted throughput;
- improvement over the encoded database-default configuration;
- number of evaluations;
- number of duplicate decoded configurations;
- checkpoint checksum.
