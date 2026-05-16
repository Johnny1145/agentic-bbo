# Background

`guacamol_qed_selfies_demo` is a GuacaMol-derived goal-directed molecule optimization task.
It uses the local BO tutorial ZINC archive only to seed a SELFIES token vocabulary and a valid default molecule.

The task decodes each fixed-length SELFIES token sequence to SMILES and scores the molecule with RDKit QED, matching the objective used by GuacaMol's `qed_benchmark`.
This is a deterministic cheminformatics scoring task, not a docking, wet-lab, or distribution-learning benchmark.
