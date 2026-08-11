# Domain Prior Knowledge

This is a GuacaMol median-molecule benchmark. GuacaMol describes median-molecule tasks as conflicting tasks in which similarity to several molecules is maximized simultaneously.

For this task, the benchmark-defined target molecules are camphor and menthol. The scoring function uses ECFP4 similarities and a geometric mean, so the scalar score is low if similarity to either target is very weak.

No additional structural editing rule is specified by the benchmark.
