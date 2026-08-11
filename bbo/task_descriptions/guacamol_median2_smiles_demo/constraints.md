# Constraints

A submission must provide one SMILES string as the value of `smiles`.

The SMILES string must be parseable as a valid molecule by RDKit. Invalid or empty SMILES strings receive the minimum score.

The benchmark returns a scalar score in [0, 1]. In the local loss-minimization interface, the optimized loss is `1 - guacamol_score`; therefore, minimizing loss is equivalent to maximizing the GuacaMol score.

Do not assume access to hidden scoring components unless the experiment is explicitly configured as a task-aware or grey-box setting.
