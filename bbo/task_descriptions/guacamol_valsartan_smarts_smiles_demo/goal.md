# Goal

Implementation source: `bbo/tasks/scientific/guacamol_smiles.py`, `_score_mol()`.

Maximize `valsartan_smarts_score`.

Optimized objective: minimize `valsartan_smarts_loss = 1 - valsartan_smarts_score`.

Fingerprint type recorded in the task definition: `none`.

The implemented score uses `_geometric_mean()` over the Valsartan SMARTS match, logP, TPSA, and Bertz complexity modifiers.
