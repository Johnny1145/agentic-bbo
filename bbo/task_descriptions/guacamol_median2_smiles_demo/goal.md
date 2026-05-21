# Goal

Implementation source: `bbo/tasks/scientific/guacamol_smiles.py`, `_score_mol()`.

Maximize `median2_score`.

Optimized objective: minimize `median2_loss = 1 - median2_score`.

Fingerprint type recorded in the task definition: `ECFP6`.

The implemented score uses `_geometric_mean()` over ECFP6 similarity to tadalafil and sildenafil.
