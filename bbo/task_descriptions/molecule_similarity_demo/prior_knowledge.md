# Domain Prior Knowledge

Tanimoto similarity measures overlap between molecular fingerprints.
Here, the fingerprint is ECFP4, which encodes local circular atom environments up to radius 2 bonds around each atom.

A higher Tanimoto score means the candidate and target share more encoded structural features.
This can be useful for rediscovery and analogue-search settings, where the aim is to find molecules structurally close to a known active compound.

SELFIES token sequences are robust to many edits, but not every fixed-length token configuration decodes to a useful RDKit molecule.
Short valid sequences can represent simple molecules; longer sequences expand the search space quickly.
Including tokens seen in the Celecoxib target can help recover substructures related to the target fingerprint.

The score should not be interpreted as biological activity.
High structural similarity to Celecoxib does not guarantee target binding, selectivity, toxicity, synthetic accessibility, or clinical usefulness.
