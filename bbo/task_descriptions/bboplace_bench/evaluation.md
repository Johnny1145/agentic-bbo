# Evaluation Protocol

For the packaged task, the evaluator returns one scalar y for each submitted candidate x.

- y is the HPWL of the decoded macro placement.
- Lower y is better.
- One candidate submission counts as one evaluation.
- The current adapter exposes only this HPWL-style macro-placement objective.
- Full GP-HPWL or downstream PPA metrics are not part of this task unless a different evaluator is used.
