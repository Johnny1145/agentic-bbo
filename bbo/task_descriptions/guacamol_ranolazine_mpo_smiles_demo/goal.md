# Goal

Implementation source: `bbo/tasks/scientific/guacamol_smiles.py`, `_score_mol()`.

Maximize `ranolazine_mpo_score`.

Optimized objective: minimize `ranolazine_mpo_loss = 1 - ranolazine_mpo_score`.

Fingerprint type recorded in the task definition: `AP`.

The implemented score uses `_geometric_mean()` over Ranolazine AP similarity, logP, fluorine count, and TPSA modifiers.
