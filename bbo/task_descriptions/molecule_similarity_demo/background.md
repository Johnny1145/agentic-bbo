# Background

`molecule_similarity_demo` is a generative molecule similarity task.
The staged ZINC archive at `examples/Molecule/zinc.txt.gz` is used to seed a SELFIES token vocabulary and a valid default molecule, not as a closed candidate pool.

The task decodes each fixed-length SELFIES token sequence to SMILES, then scores the molecule by structural similarity to a fixed target molecule, Celecoxib.
It uses RDKit to parse the decoded SMILES, compute an ECFP4 Morgan fingerprint, and compare the candidate fingerprint with the target fingerprint using Tanimoto similarity.

This is not a docking, molecular-dynamics, quantum-chemistry, or wet-lab assay task.
It is a deterministic cheminformatics scoring task intended to test whether an optimizer can generate molecules that share structural features with a known target.
