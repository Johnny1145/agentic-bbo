# Constraints

Implementation source: `bbo/tasks/scientific/guacamol_smiles.py`.

Input parameter: `StringParam("smiles", default="", min_length=0, max_length=512)`.

Invalid or empty SMILES receive score `0.0` and loss `1.0`.

The schema default role is `schema_only_not_initial_population`.
