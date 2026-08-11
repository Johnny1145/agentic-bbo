# Domain Prior Knowledge

This prior turns the declared MLP-SGD training protocol into hypotheses that can be
tested using only the five-fold black-box objective. It does not provide a preferred
configuration.

## Mechanism

- `learning_rate_init`, `momentum`, and `power_t` jointly control the size and decay of
  SGD updates. With inverse scaling, a larger `power_t` makes the learning rate decay
  faster. High initial step size and high momentum can therefore behave very
  differently from either setting alone.
- The 40-iteration cap and early stopping make learning speed part of the objective:
  a configuration that would work after long training can still be poor here.
- `batch_size` changes both the number and noise of updates per epoch, so it should be
  tested together with the learning-dynamics parameters rather than treated as an
  independent systems knob.
- `hidden_layer_sizes` and `alpha` form a capacity--regularization trade-off. The
  useful capacity can only be judged after a workable learning-dynamics region has
  been found.
- `validation_fraction` changes how much data is used for fitting and how much is used
  by early stopping; `tol` changes how much improvement is required to continue.

## Testable hypotheses

1. Under the short training cap, crossed changes to `learning_rate_init`, `momentum`,
   and `power_t` will explain more of the early objective variation than isolated
   changes to network width.
2. Once a stable learning-dynamics region is found, changing width and `alpha`
   together will be more informative than always increasing width.
3. The best `batch_size`, `tol`, and `validation_fraction` settings will depend on the
   selected learning-dynamics region; their effects should not be extrapolated from a
   single anchor.

## Suggested search sequence

1. Use the initial observations to choose several distinct anchors, not only the
   current best. Spend roughly the first third of new evaluations on crossed
   low/middle/high probes of `learning_rate_init`, `momentum`, and `power_t` around
   those anchors.
2. Keep two or more promising learning-dynamics combinations and spend the next third
   testing paired changes of `hidden_layer_sizes` and `alpha`.
3. Use the final third for `batch_size`, `tol`, and `validation_fraction`, followed by
   recombinations of changes that improved different anchors.
4. Compare every proposal using the five-fold objective only, respect the declared
   log/logit transforms, and reject duplicate decoded configurations before spending
   an evaluation.

## Failure signals and adjustment

- If a neighborhood is highly sensitive and neighboring settings repeatedly become
  much worse, reduce effective step pressure by testing a lower initial learning
  rate, lower momentum, or faster decay instead of refining the same point.
- If all learning-dynamics probes remain similarly mediocre, stop micro-tuning them
  and test capacity, regularization, and batch size from more than one anchor.
- If local recombinations stop improving, return to a deliberately different
  learning-dynamics regime rather than assuming that smaller local steps will help.

No best-known configuration, task result, held-out outcome, or test-set-derived prior
is provided.
