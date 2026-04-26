# Constraints

- Submissions must provide every fixed-length `selfies_token_XX` categorical parameter declared by the task.
- Token values must come from the SELFIES vocabulary exposed in the task search space.
- The ZINC archive seeds the SELFIES vocabulary and default molecule; valid decoded molecules may be outside the original archive.
- The evaluator is deterministic and does not use the task seed when computing similarity.
- Unless overridden, the task stops after 40 evaluations.
- Empty, invalid, or undecodable SELFIES sequences receive `tanimoto_similarity = 0.0` and therefore `similarity_loss = 1.0`.
