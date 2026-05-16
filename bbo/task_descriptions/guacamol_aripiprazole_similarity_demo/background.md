# Background

`guacamol_aripiprazole_similarity_demo` is a GuacaMol-derived similarity task.
It decodes SELFIES token configurations to SMILES and scores them by FCFP4 Tanimoto similarity to Aripiprazole with GuacaMol's thresholded similarity modifier.

Unlike rediscovery tasks, this objective rewards high similarity but does not require exact reconstruction of the target molecule.
