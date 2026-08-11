# Domain Prior Knowledge

This task-aware prior uses only the scoring semantics declared in P1: ECFP4 similarity
to amlodipine and a total-ring-count modifier centered at three, combined by a
geometric mean.

## Mechanism

- A geometric mean is bottleneck-sensitive: a candidate cannot compensate for a very
  weak ring-count term using similarity alone, or for very low similarity using ring
  count alone.
- Ring count changes discretely. A local edit that preserves ring topology can probe
  similarity while approximately holding the ring-count term fixed; adding, deleting,
  opening, or closing a ring changes the other component as well.
- ECFP4 is a local substructure fingerprint. Small scaffold-preserving edits are more
  likely to retain fingerprint overlap than a complete scaffold replacement, although
  every claim must still be tested with the scalar objective.

## Testable hypotheses

1. Among valid candidates with the same total ring count, one-edit molecular variants
   will provide a cleaner signal about similarity than unrelated SMILES rewrites.
2. Moving a candidate toward three total rings can improve the combined score even if
   it sacrifices some similarity, but repeated ring edits after reaching three are
   likely to spend budget on the wrong component.
3. Lineages that preserve a recognizable scaffold while varying substituents will be
   more sample-efficient than repeatedly generating unrelated molecules.

## Suggested search sequence

1. Organize the initial valid molecules into a few parent lineages. Prefer parents
   that are both high-scoring and structurally distinct; do not keep only one current
   best SMILES.
2. First identify lineages that plausibly have three rings. Around those parents,
   preserve ring topology and make one local edit at a time, such as a substituent,
   short side-chain, or heteroatom change.
3. For parents with a different ring count, test a small number of explicit ring
   edits. Once a three-ring child improves, return to topology-preserving local edits.
4. Compare sibling molecules that differ by one edit, keep successful edits attached
   to their parent lineage, and occasionally recombine compatible changes. Always
   submit syntactically valid SMILES.

## Failure signals and adjustment

- A run of invalid SMILES means the edit operator is too large or syntax-unsafe;
  revert to a valid parent and use a single smaller edit.
- If topology-preserving variants are valid but uniformly poor, switch to another
  parent scaffold instead of making ever smaller changes to the same lineage.
- If ring-changing variants improve once and then plateau, freeze the three-ring
  topology and redirect the remaining search toward similarity-preserving edits.

No reference SMILES, best molecule, component-level observation, or prior-run result
is supplied.
