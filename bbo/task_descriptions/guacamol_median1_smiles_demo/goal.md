# Goal

Implementation source: `bbo/tasks/scientific/guacamol_smiles.py`, `_score_mol()`.

Maximize `median1_score`.

Optimized objective: minimize `median1_loss = 1 - median1_score`.

Fingerprint type recorded in the task definition: `ECFP4`.

The implemented score uses `_geometric_mean()`.
