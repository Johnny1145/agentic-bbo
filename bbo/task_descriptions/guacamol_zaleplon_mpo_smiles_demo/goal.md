# Goal

Implementation source: `bbo/tasks/scientific/guacamol_smiles.py`, `_score_mol()`.

Maximize `zaleplon_mpo_score`.

Optimized objective: minimize `zaleplon_mpo_loss = 1 - zaleplon_mpo_score`.

Fingerprint type recorded in the task definition: `ECFP4`.

The implemented score uses `_geometric_mean()` over Zaleplon ECFP4 similarity and formula modifiers.
