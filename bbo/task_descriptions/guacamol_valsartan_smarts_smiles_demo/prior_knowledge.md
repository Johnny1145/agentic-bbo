# Domain Prior Knowledge

This task-aware prior operationalizes only the P1 scoring semantics: SMARTS-pattern
presence plus Gaussian preferences for logP, TPSA, and Bertz complexity, combined by a
geometric mean.

## Mechanism

- The pattern-presence term is a feasibility gate for the geometric mean. A molecule
  missing the required substructure cannot make up for that absence by matching the
  three continuous property targets.
- After the motif is present, the remaining problem is balanced rather than
  single-property optimization: the weakest of logP, TPSA, or complexity can dominate
  the loss in the product.
- In qualitative chemical terms, logP responds to hydrophobic/polar balance, TPSA to
  polar atoms and functional groups, and Bertz complexity to molecular size,
  branching, rings, and structural diversity. One edit can move several of these at
  once, so proposed directions must be verified rather than assumed.

## Testable hypotheses

1. Preserving a successful motif-containing parent will be more useful than repeatedly
   rediscovering motif feasibility from unrelated molecules.
2. Matched one-edit variants outside the required motif will make the scalar response
   easier to attribute than simultaneous scaffold and property changes.
3. Once feasibility is satisfied, improving the most mismatched property while keeping
   the other two near their current values will be more effective than pushing one
   property past its target.

## Suggested search sequence

1. Partition initial candidates into motif-feasible and motif-missing lineages using
   the declared objective behavior and structural reasoning. Preserve multiple valid,
   distinct parents rather than a single best molecule.
2. If no promising feasible parent exists, prioritize deliberate motif-preserving or
   motif-constructing proposals before fine property tuning. Do not spend most of the
   budget polishing logP, TPSA, or complexity in motif-missing molecules.
3. Once a feasible parent is found, freeze the required substructure and generate
   sibling molecules with one edit outside it. Choose edits intended to correct one
   property mismatch, then use the scalar response to accept or reject that direction.
4. Alternate among the three property mechanisms and retain several lineages so that
   correcting one property does not irreversibly damage another. Recheck feasibility
   after every structural edit.

## Failure signals and adjustment

- Scores that remain at the floor despite property-oriented edits indicate that motif
  feasibility should be revisited before further property tuning.
- If a supposedly helpful edit repeatedly worsens the score, treat its direction as
  unvalidated: reverse it or test it on another feasible parent.
- If one lineage plateaus, switch to a structurally distinct feasible parent rather
  than making larger edits that risk destroying the motif.

The exact SMARTS pattern, component values, property-modifier widths, best molecule,
and prior-run results are not supplied.
