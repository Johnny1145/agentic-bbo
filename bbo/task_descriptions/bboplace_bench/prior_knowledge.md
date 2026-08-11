# Domain Prior Knowledge

The optimizer proposes 64 coordinates for 32 macros, but the evaluator first decodes
them into a legal grid placement. The prior therefore concerns the decoder, not an
assumption that HPWL is smooth in the raw coordinates.

## Mechanism

- Coordinate `x[i]` and `x[32+i]` are the horizontal and vertical proposal for the
  same macro. They should normally be treated as a pair.
- Macros with larger connected module area are placed earlier by MGO. Earlier choices
  remove legal grid locations from later macros, so some macro pairs can have much
  larger downstream effects than others.
- For each macro, the decoder first minimizes incremental HPWL over legal grids. The
  proposed coordinate only breaks ties by distance. A raw-coordinate move can
  therefore leave the decoded layout unchanged, while crossing a decoder boundary can
  cause a discrete jump.
- HPWL is a wirelength proxy, not final routed PPA; only the returned HPWL should be
  optimized in this experiment.

## Testable hypotheses

1. Paired x/y moves for one macro will be more interpretable than changing unrelated
   x coordinates without their y partners.
2. A minority of macro pairs will cause larger objective changes because of placement
   order and connectivity, so screening pairs before full-dimensional refinement will
   use budget more effectively.
3. Very small raw-coordinate perturbations will often fall on decoder plateaus;
   multi-scale moves should reveal boundaries that fine local search misses.

## Suggested search sequence

1. From several initial anchors, screen macro pairs or small blocks of pairs. Change
   both coordinates of a selected macro and keep all unselected pairs fixed.
2. For each screened pair, use at least two move scales: one local move and one large
   enough to seek a different decoded grid region. Do not assume the smallest move is
   the most informative.
3. Rank pairs by repeatable objective response across anchors, then spend most local
   refinement on the responsive pairs while preserving the other coordinates of a
   good anchor.
4. Periodically propose a block move involving several responsive pairs, because an
   early macro can occupy a location needed by a later one. Keep some independent
   anchors so that one placement order outcome does not dominate the search.

## Failure signals and adjustment

- Repeatedly identical objectives after small local moves are evidence of a possible
  decoder plateau, not proof that the macro is irrelevant. Increase the move scale or
  change both coordinates together.
- If moving many macros at once gives an improvement that cannot be localized, return
  to smaller blocks and perform controlled ablations around that candidate.
- If refinement of responsive pairs stalls, restart from another anchor or perturb an
  earlier interacting block rather than spending the remaining budget inside one
  decoded basin.

No decoded placement, connectivity ranking, best coordinates, or prior-run result is
provided.
