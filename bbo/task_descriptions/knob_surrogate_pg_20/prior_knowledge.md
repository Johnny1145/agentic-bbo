# Domain Prior Knowledge

The medium PostgreSQL configuration space combines settings related to:

- shared memory;
- worker and process concurrency;
- query execution memory;
- optimizer cost estimates;
- WAL buffering;
- checkpoint scheduling;
- background writing;
- statistics collection;
- storage and page flushing.

Useful expert heuristics:

- Group knobs by subsystem before interpreting observed improvements.
- Preserve automatic sentinel settings as distinct alternatives.
- Treat categorical settings as discrete, not ordinal.
- Test memory settings jointly because one limit can change the effective meaning of another.
- Test checkpoint timing and checkpoint pacing jointly.
- Treat optimizer cost settings as potentially discontinuous because they can change query plans.
- Prefer sparse, interpretable changes early in the run.
- Expand to cross-subsystem changes after identifying promising single-subsystem effects.
- Avoid over-localizing when several consecutive proposals fail to improve latency.
- Track decoded configurations to avoid integer and categorical duplicates.
- Do not assume that the twenty variables have equal importance.

The source paper reports interactions among some checkpoint-related knobs. The default prior acknowledges that such interactions can exist but does not disclose the paper's best-value region.
