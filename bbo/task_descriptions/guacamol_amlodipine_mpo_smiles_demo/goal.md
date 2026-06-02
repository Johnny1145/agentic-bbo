# Goal

Implementation source: `bbo/tasks/scientific/guacamol_smiles.py`, `_score_mol()`.

Maximize `amlodipine_mpo_score`.

Optimized objective: minimize `amlodipine_mpo_loss = 1 - amlodipine_mpo_score`.

Fingerprint type recorded in the task definition: `ECFP4`.

The implemented score uses `_geometric_mean()` over Amlodipine ECFP4 similarity and a three-ring count modifier.
