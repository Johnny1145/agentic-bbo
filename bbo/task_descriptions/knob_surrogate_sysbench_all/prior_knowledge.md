# Domain Prior Knowledge

A full MySQL configuration space is high-dimensional and heterogeneous. The knobs control different subsystems, including:

- memory allocation;
- concurrency;
- query execution;
- logging and recovery;
- buffer management;
- storage and flushing;
- optimizer statistics;
- replication and binary logging.

Most candidate configurations should not be produced by changing every knob independently and by a large amount. Such proposals are difficult to interpret and can combine incompatible subsystem settings.

Useful expert heuristics:

- Start from the encoded database-default configuration or from a small set of diverse, valid anchors.
- Use observed evaluations to identify a smaller active subset before performing aggressive local refinement.
- Group variables by subsystem when analyzing changes.
- Treat categorical variables as discrete alternatives.
- Do not assume that normalized distance corresponds to equal physical or semantic distance.
- Prefer structured perturbations that change a small number of related knobs.
- Occasionally test cross-subsystem interactions after single-subsystem effects have been observed.
- Detect and avoid normalized proposals that decode to an already evaluated physical configuration.
- Maintain exploration because low-ranked or initially inactive knobs may matter through interactions.

No benchmark-specific optimum, SHAP ranking, or direction-of-effect from the source response surface is disclosed here.
