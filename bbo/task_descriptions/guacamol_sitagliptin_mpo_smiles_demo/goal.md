# Goal

Implementation source: `bbo/tasks/scientific/guacamol_smiles.py`, `_score_mol()`.

Maximize `sitagliptin_mpo_score`.

Optimized objective: minimize `sitagliptin_mpo_loss = 1 - sitagliptin_mpo_score`.

Fingerprint type recorded in the task definition: `ECFP4`.

The implemented score uses `_geometric_mean()` over Sitagliptin ECFP4 dissimilarity, logP, TPSA, and formula modifiers.
