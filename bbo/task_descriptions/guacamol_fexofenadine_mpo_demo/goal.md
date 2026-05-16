# Goal

Minimize `fexofenadine_mpo_loss`, where:

- `fexofenadine_mpo_loss = 1.0 - fexofenadine_mpo_score`
- `fexofenadine_mpo_score` is the geometric mean of GuacaMol-style Fexofenadine similarity, TPSA, and logP components

Lower loss means the decoded molecule better satisfies the combined MPO objective.
