# Background

`guacamol_fexofenadine_mpo_demo` is a GuacaMol-derived multi-property optimization task.
It decodes SELFIES token configurations to SMILES and combines three components: atom-pair similarity to Fexofenadine, high TPSA, and low logP.

The implementation follows the structure of GuacaMol's `hard_fexofenadine` benchmark while exposing a fixed categorical SELFIES search space.
