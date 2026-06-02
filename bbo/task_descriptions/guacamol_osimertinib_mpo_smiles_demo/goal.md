# Goal

Implementation source: `bbo/tasks/scientific/guacamol_smiles.py`, `_score_mol()`.

Maximize `osimertinib_mpo_score`.

Optimized objective: minimize `osimertinib_mpo_loss = 1 - osimertinib_mpo_score`.

Fingerprint type recorded in the task definition: `FCFP4, ECFP6`.

The implemented score uses `_geometric_mean()` over Osimertinib FCFP4 similarity, ECFP6 dissimilarity, TPSA, and logP modifiers.
