# Goal

Implementation source: `bbo/tasks/scientific/guacamol_smiles.py`, `_score_mol()`.

Maximize `perindopril_mpo_score`.

Optimized objective: minimize `perindopril_mpo_loss = 1 - perindopril_mpo_score`.

Fingerprint type recorded in the task definition: `ECFP4`.

The implemented score uses `_geometric_mean()` over Perindopril ECFP4 similarity and a two-aromatic-ring count modifier.
