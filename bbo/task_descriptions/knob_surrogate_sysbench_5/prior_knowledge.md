# Domain Prior Knowledge

SYSBENCH is a transactional workload. Throughput can be affected by interacting limits on memory use, concurrent execution, logging, and write durability.

The five active knobs have the following roles:

- `tmp_table_size` limits the size of internal in-memory temporary tables.
- `max_heap_table_size` limits the size of user-created MEMORY tables.
- `query_prealloc_size` controls a persistent buffer used during statement parsing and execution.
- `innodb_thread_concurrency` limits the number of threads allowed to execute concurrently inside InnoDB. The value `0` has special database semantics and should not automatically be treated as the smallest ordinary concurrency limit.
- `innodb_doublewrite` controls whether the InnoDB doublewrite mechanism is enabled. This knob creates a durability-versus-write-overhead trade-off and is categorical rather than continuous.

Useful search principles:

- Do not assume that increasing a memory or concurrency knob is always beneficial.
- Test interactions between temporary-table limits and concurrency rather than varying each knob only in isolation.
- Treat `innodb_doublewrite` as a discrete branch of the search.
- Preserve some evaluations for configurations that differ in more than one subsystem.
- Compare proposed configurations against the encoded database-default configuration.
- Avoid repeatedly proposing normalized points that decode to the same physical configuration.

This file intentionally does not provide the best values reported by the source experiment.
