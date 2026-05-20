# Goal

Implementation source: `bbo/tasks/scientific/guacamol_smiles.py`, `_score_mol()`.

Maximize `fexofenadine_mpo_score`.

Optimized objective: minimize `fexofenadine_mpo_loss = 1 - fexofenadine_mpo_score`.

Fingerprint type recorded in the task definition: `AP`.

The implemented score uses `_geometric_mean()`.
