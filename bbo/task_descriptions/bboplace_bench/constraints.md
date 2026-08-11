# Constraints

- The raw candidate vector x must stay within the coordinate bounds exposed by the task.
- The raw vector is a proposal, not necessarily a legal final placement.
- The evaluator applies MGO decoding to obtain a legal placement:
  - candidate grid locations that overlap existing macros are excluded;
  - grid locations outside the chip canvas are excluded;
  - each macro is moved to a legal grid before HPWL is computed.
- The optimizer should not assume gradients, convexity, smoothness, or a known optimum.
- The number of evaluator queries is limited by the task budget.
