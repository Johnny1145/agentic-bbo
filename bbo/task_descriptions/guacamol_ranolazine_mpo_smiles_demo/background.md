# Background

This task is a GuacaMol goal-directed molecular design benchmark over valid molecular structures.

A candidate is represented as a SMILES string. The evaluator maps each valid molecule to a scalar GuacaMol score in [0, 1]. The benchmark objective is to find molecules with high scalar scores under the task-specific scoring function.

GuacaMol describes these goal-directed tasks as molecular design problems in which models generate high-scoring molecules for a predefined scoring function. Some tasks combine several molecular properties into a multi-property objective (MPO), but the benchmark still returns one scalar molecule score.

This description records benchmark-level information only. It does not include agent-discovered search strategies or unsupported medicinal-chemistry heuristics.
