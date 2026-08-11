# Domain Prior Knowledge

This is a 196-knob analytical-workload task. The useful prior is not that "database
knobs interact" in general, but that specific declared knobs form coupled mechanisms
that can be isolated with controlled contrasts.

## Mechanism groups visible in this search space

- Join, sort, and scan working memory: `join_buffer_size`, `sort_buffer_size`,
  `read_buffer_size`, and `read_rnd_buffer_size`. These are workload and plan
  dependent; increasing all of them together does not identify which mechanism
  changed latency.
- Internal temporary tables: `tmp_table_size` and `max_heap_table_size` act as a
  coupled limit. Testing only one while the other remains restrictive can hide an
  effect.
- InnoDB cache and I/O service: `innodb_buffer_pool_size`, `innodb_io_capacity`,
  `innodb_io_capacity_max`, `innodb_read_io_threads`, and
  `innodb_write_io_threads`. Capacity, background work, and concurrency should be
  tested as a subsystem rather than as independent monotone knobs.
- Join-plan and access-path decisions: `eq_range_index_dive_limit`,
  `max_seeks_for_key`, `optimizer_prune_level`, and `optimizer_search_depth` can alter
  plan selection. Such changes can create regime jumps rather than smooth local
  responses.
- Connection and execution concurrency: `innodb_thread_concurrency`,
  `thread_cache_size`, and `table_open_cache` can interact with the workload's
  parallel activity and resource use.

## Testable hypotheses

1. Controlled changes within one mechanism group will produce a clearer signal than
   changing dozens of unrelated knobs in the same candidate.
2. The paired temporary-table knobs will be more informative when moved together than
   when only one is changed.
3. Improvements caused by optimizer or access-path knobs may appear as discontinuous
   jumps; a smooth local model around one plan regime can therefore miss them.
4. A group that looks inactive at one anchor may become active at another anchor
   because memory, I/O, concurrency, and plan choices interact.

## Suggested search sequence

1. Select several diverse initial anchors, including but not limited to the current
   best. From each anchor, create paired contrasts that modify only one mechanism
   group while every other knob remains fixed.
2. Screen the memory, temporary-table, cache/I/O, plan-selection, and concurrency
   groups. Within a group, use a small coordinated set of changes; do not sweep all
   196 coordinates at once.
3. Promote groups that improve at more than one anchor, then test two-group
   interactions such as temporary-table memory with plan selection, or cache capacity
   with I/O service.
4. Search categorical and boundary/sentinel values explicitly and reserve at least a
   fifth of the optimization budget for regime changes away from the current basin.
5. Only perform repeated local refinement after a group or interaction has improved
   controlled contrasts. Track duplicates after physical decoding, not only in the
   normalized coordinates.

## Failure signals and adjustment

- If a one-knob change has no effect, first test whether it is capped or masked by a
  coupled knob before declaring it inactive.
- If many coordinated changes improve once but cannot be reproduced from another
  anchor, reduce the group size and identify the responsible contrast.
- If local proposals plateau, change a plan-selection category or another mechanism
  group rather than only shrinking the numerical step size.
- Do not infer that all JOB queries respond in the same direction from the scalar
  95th-percentile objective.

No surrogate-training feature ranking, historical result, or best-value pattern is
exposed.
