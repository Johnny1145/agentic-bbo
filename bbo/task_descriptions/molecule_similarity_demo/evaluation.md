# Evaluation Protocol

- Data source: staged copy of `examples/Molecule/zinc.txt.gz`; the evaluator reads the archive member `zinc.txt` to seed the SELFIES vocabulary and default molecule.
- For each evaluation, concatenate SELFIES tokens until `__EOS__`, ignore `__PAD__`, decode to SMILES, then parse the decoded SMILES with RDKit.
- Compute an ECFP4 Morgan fingerprint for the candidate and target molecule.
- Compute Tanimoto similarity between the two fingerprints.
- Report `similarity_loss = 1.0 - tanimoto_similarity` as the primary objective.
- Log raw `tanimoto_similarity`, decoded `selfies`, decoded `smiles`, token list, validity flags, target metadata, and the fingerprint type.
