# Literature-aligned task descriptions for the 10 PABLO GuacaMol molecular tasks

This document gives a clean, literature-aligned version of the task descriptions for the 10 GuacaMol molecular design tasks used by PABLO.

The goal is to remove agent-generated search advice and keep only information that can be traced to the GuacaMol benchmark definition or to the PABLO experimental setting.

## Final consistency check

The descriptions below are consistent with the papers under the following interpretation.

1. **PABLO main GuacaMol results are single-objective black-box optimization.**
   PABLO optimizes a scalar objective value over structured discrete inputs. In the molecular setting, candidates are SMILES strings and the oracle returns one scalar score or loss.

2. **GuacaMol MPO tasks are scalarized multi-property objectives, not Pareto multi-objective optimization.**
   GuacaMol defines a molecule score that combines several molecular properties such as similarity, logP, TPSA, ring count, SMARTS pattern matching, or formula matching. The optimizer still maximizes one scalar score.

3. **For strict PABLO main-setting reproduction, do not expose task semantics.**
   PABLO's primary Figure 3 / Table 1 comparisons are reported without task descriptions or literature tools. In that setting, the agent should only see evaluated molecules and numeric scores.

4. **For Task Awareness / grey-box agentic runs, expose only benchmark-defined semantics.**
   It is acceptable to expose the target molecule names, fingerprint types, property terms, and aggregation rules because these are GuacaMol task definitions. It is not appropriate to add advice such as “increase hydrophobicity,” “replace this group,” “shorter SELFIES are better,” or “try modifying a scaffold,” unless a separate literature-retrieval setting is explicitly enabled and the retrieved source supports it.

## Recommended common files

The following common files can be reused across all 10 SMILES-based GuacaMol task directories.

### `background.md`

```md
# Background

This task is a GuacaMol goal-directed molecular design benchmark over valid molecular structures.

A candidate is represented as a SMILES string. The evaluator maps each valid molecule to a scalar GuacaMol score in [0, 1]. The benchmark objective is to find molecules with high scalar scores under the task-specific scoring function.

GuacaMol describes these goal-directed tasks as molecular design problems in which models generate high-scoring molecules for a predefined scoring function. Some tasks combine several molecular properties into a multi-property objective (MPO), but the benchmark still returns one scalar molecule score.

This description records benchmark-level information only. It does not include agent-discovered search strategies or unsupported medicinal-chemistry heuristics.
```

### `constraints.md`

```md
# Constraints

A submission must provide one SMILES string as the value of `smiles`.

The SMILES string must be parseable as a valid molecule by RDKit. Invalid or empty SMILES strings receive the minimum score.

The benchmark returns a scalar score in [0, 1]. In the local loss-minimization interface, the optimized loss is `1 - guacamol_score`; therefore, minimizing loss is equivalent to maximizing the GuacaMol score.

Do not assume access to hidden scoring components unless the experiment is explicitly configured as a task-aware or grey-box setting.
```

### `prior_knowledge.md` base template

```md
# Domain Prior Knowledge

This file records only benchmark-level information from GuacaMol and the local scoring implementation.

The optimizer receives a scalar score. Some tasks combine multiple molecular properties, but these properties are aggregated into one molecule score. This is therefore a single-objective black-box optimization task from the optimizer's perspective.

No additional medicinal-chemistry modification rules, scaffold-editing strategies, or agent-discovered heuristics are included here.
```

---

## Per-task task-aware descriptions

The sections below provide task-specific `goal.md` and `prior_knowledge.md` contents. They are appropriate for a **Task Awareness / grey-box** variant. For strict black-box PABLO reproduction, use only the common background and constraints and do not expose the task-specific scoring components.

---

## 1. `guacamol_median1_smiles_demo` / PABLO abbreviation: `med1`

### `goal.md`

```md
# Goal

Maximize the GuacaMol Median Molecules 1 score.

This benchmark scores a molecule by its similarity to two target molecules: camphor and menthol. The score combines ECFP4 Tanimoto similarity to camphor and ECFP4 Tanimoto similarity to menthol using a geometric mean.

In the local loss-minimization interface, optimize `median1_loss = 1 - median1_score`.
```

### `prior_knowledge.md`

```md
# Domain Prior Knowledge

This is a GuacaMol median-molecule benchmark. GuacaMol describes median-molecule tasks as conflicting tasks in which similarity to several molecules is maximized simultaneously.

For this task, the benchmark-defined target molecules are camphor and menthol. The scoring function uses ECFP4 similarities and a geometric mean, so the scalar score is low if similarity to either target is very weak.

No additional structural editing rule is specified by the benchmark.
```

---

## 2. `guacamol_median2_smiles_demo` / PABLO abbreviation: `med2`

### `goal.md`

```md
# Goal

Maximize the GuacaMol Median Molecules 2 score.

This benchmark scores a molecule by its similarity to two target molecules: tadalafil and sildenafil. The score combines ECFP6 Tanimoto similarity to tadalafil and ECFP6 Tanimoto similarity to sildenafil using a geometric mean.

In the local loss-minimization interface, optimize `median2_loss = 1 - median2_score`.
```

### `prior_knowledge.md`

```md
# Domain Prior Knowledge

This is a GuacaMol median-molecule benchmark. The benchmark aims to evaluate whether a model can generate molecules that lie between two target structures under the specified similarity measure.

For this task, the benchmark-defined target molecules are tadalafil and sildenafil. The scoring function uses ECFP6 similarities and a geometric mean.

No additional structural editing rule is specified by the benchmark.
```

---

## 3. `guacamol_osimertinib_mpo_smiles_demo` / PABLO abbreviation: `osmb`

### `goal.md`

```md
# Goal

Maximize the GuacaMol Osimertinib MPO score.

This benchmark combines the following benchmark-defined scoring components into one scalar molecule score:

- similarity to osimertinib using FCFP4, with a thresholded modifier at 0.8;
- similarity to osimertinib using ECFP6, with a MinGaussian modifier centered at 0.85;
- TPSA, with a MaxGaussian modifier centered at 100;
- logP, with a MinGaussian modifier centered at 1.

The components are aggregated by a geometric mean. In the local loss-minimization interface, optimize `osimertinib_mpo_loss = 1 - osimertinib_mpo_score`.
```

### `prior_knowledge.md`

```md
# Domain Prior Knowledge

This is one of GuacaMol's drug-related multi-property objective benchmarks. GuacaMol describes these tasks as objectives related to known drug molecules that fine-tune structural or physicochemical properties.

The benchmark-defined reference molecule is osimertinib. The score is a scalarized combination of similarity, TPSA, and logP terms.

No additional medicinal-chemistry rule is specified by the benchmark.
```

---

## 4. `guacamol_fexofenadine_mpo_smiles_demo` / PABLO abbreviation: `fexo`

### `goal.md`

```md
# Goal

Maximize the GuacaMol Fexofenadine MPO score.

This benchmark combines the following benchmark-defined scoring components into one scalar molecule score:

- similarity to fexofenadine using atom-pair fingerprints (AP), with a thresholded modifier at 0.8;
- TPSA, with a MaxGaussian modifier centered at 90;
- logP, with a MinGaussian modifier centered at 4.

The components are aggregated by a geometric mean. In the local loss-minimization interface, optimize `fexofenadine_mpo_loss = 1 - fexofenadine_mpo_score`.
```

### `prior_knowledge.md`

```md
# Domain Prior Knowledge

This is one of GuacaMol's drug-related multi-property objective benchmarks. The benchmark-defined reference molecule is fexofenadine.

The score is a scalarized combination of atom-pair similarity, TPSA, and logP terms. The geometric mean aggregation means that the final score depends on all listed components.

Do not add search instructions such as how to chemically modify fexofenadine unless an external literature source explicitly supports them.
```

---

## 5. `guacamol_ranolazine_mpo_smiles_demo` / PABLO abbreviation: `rano`

### `goal.md`

```md
# Goal

Maximize the GuacaMol Ranolazine MPO score.

This benchmark combines the following benchmark-defined scoring components into one scalar molecule score:

- similarity to ranolazine using atom-pair fingerprints (AP), with a thresholded modifier at 0.7;
- logP, with a MaxGaussian modifier centered at 7;
- TPSA, with a MaxGaussian modifier centered at 95;
- number of fluorine atoms, with a Gaussian modifier centered at 1.

The components are aggregated by a geometric mean. In the local loss-minimization interface, optimize `ranolazine_mpo_loss = 1 - ranolazine_mpo_score`.
```

### `prior_knowledge.md`

```md
# Domain Prior Knowledge

This is one of GuacaMol's drug-related multi-property objective benchmarks. The benchmark-defined reference molecule is ranolazine.

GuacaMol notes that the Ranolazine MPO benchmark uses a start population containing one molecule, ranolazine, in the original benchmark description. The scoring function combines ranolazine similarity with logP, TPSA, and fluorine-count terms.

No additional structural editing rule is specified by the benchmark.
```

---

## 6. `guacamol_perindopril_mpo_smiles_demo` / PABLO abbreviation: `pdop`

### `goal.md`

```md
# Goal

Maximize the GuacaMol Perindopril MPO score.

This benchmark combines the following benchmark-defined scoring components into one scalar molecule score:

- similarity to perindopril using ECFP4;
- number of aromatic rings, with a Gaussian modifier centered at 2.

The components are aggregated by a geometric mean. In the local loss-minimization interface, optimize `perindopril_mpo_loss = 1 - perindopril_mpo_score`.
```

### `prior_knowledge.md`

```md
# Domain Prior Knowledge

This is one of GuacaMol's drug-related multi-property objective benchmarks. The benchmark-defined reference molecule is perindopril.

The score is a scalarized combination of perindopril similarity and an aromatic-ring-count term. The task description should not add molecule-editing advice beyond these benchmark-defined components.
```

---

## 7. `guacamol_amlodipine_mpo_smiles_demo` / PABLO abbreviation: `adip`

### `goal.md`

```md
# Goal

Maximize the GuacaMol Amlodipine MPO score.

This benchmark combines the following benchmark-defined scoring components into one scalar molecule score:

- similarity to amlodipine using ECFP4;
- total number of rings, with a Gaussian modifier centered at 3.

The components are aggregated by a geometric mean. In the local loss-minimization interface, optimize `amlodipine_mpo_loss = 1 - amlodipine_mpo_score`.
```

### `prior_knowledge.md`

```md
# Domain Prior Knowledge

This is one of GuacaMol's drug-related multi-property objective benchmarks. The benchmark-defined reference molecule is amlodipine.

The score is a scalarized combination of amlodipine similarity and a total-ring-count term. PABLO uses the Amlodipine task as an example of Task Awareness with the natural-language objective “maximize similarity to amlodipine while having exactly 3 total rings.” This is a task-aware setting, not the strict black-box setting.
```

---

## 8. `guacamol_sitagliptin_mpo_smiles_demo` / PABLO abbreviation: `siga`

### `goal.md`

```md
# Goal

Maximize the GuacaMol Sitagliptin MPO score.

This benchmark combines the following benchmark-defined scoring components into one scalar molecule score:

- ECFP4 similarity to sitagliptin, with a Gaussian modifier centered at 0, encouraging dissimilarity to sitagliptin;
- logP, with a Gaussian modifier centered at 2.0165;
- TPSA, with a Gaussian modifier centered at 77.04;
- molecular formula matching for C16H15F6N5O.

The components are aggregated by a geometric mean. In the local loss-minimization interface, optimize `sitagliptin_mpo_loss = 1 - sitagliptin_mpo_score`.
```

### `prior_knowledge.md`

```md
# Domain Prior Knowledge

GuacaMol describes this benchmark as requiring models to generate molecules that are as dissimilar to sitagliptin as possible while keeping some of its properties.

The benchmark-defined property terms are sitagliptin dissimilarity, logP, TPSA, and the molecular formula C16H15F6N5O. These terms are aggregated into one scalar score.

No additional structural editing rule is specified by the benchmark.
```

---

## 9. `guacamol_valsartan_smarts_smiles_demo` / PABLO abbreviation: `valt`

### `goal.md`

```md
# Goal

Maximize the GuacaMol Valsartan SMARTS score.

This benchmark combines the following benchmark-defined scoring components into one scalar molecule score:

- presence of a valsartan-related SMARTS pattern;
- logP, with a Gaussian modifier centered at 2.0165;
- TPSA, with a Gaussian modifier centered at 77.04;
- Bertz complexity, with a Gaussian modifier centered at 896.38.

The components are aggregated by a geometric mean. In the local loss-minimization interface, optimize `valsartan_smarts_loss = 1 - valsartan_smarts_score`.
```

### `prior_knowledge.md`

```md
# Domain Prior Knowledge

This is a GuacaMol SMARTS-pattern benchmark. GuacaMol describes it as targeting molecules containing a SMARTS pattern related to valsartan while having physicochemical properties corresponding to the sitagliptin molecule.

The benchmark-defined terms are SMARTS-pattern presence, logP, TPSA, and Bertz complexity. These terms are aggregated into one scalar score.

Do not add unsupported instructions about how to build or modify the SMARTS-containing molecule.
```

---

## 10. `guacamol_zaleplon_mpo_smiles_demo` / PABLO abbreviation: `zale`

### `goal.md`

```md
# Goal

Maximize the GuacaMol Zaleplon MPO score.

This benchmark combines the following benchmark-defined scoring components into one scalar molecule score:

- similarity to zaleplon using ECFP4;
- molecular formula matching for C19H17N3O2.

The components are aggregated by a geometric mean. In the local loss-minimization interface, optimize `zaleplon_mpo_loss = 1 - zaleplon_mpo_score`.
```

### `prior_knowledge.md`

```md
# Domain Prior Knowledge

GuacaMol describes this benchmark as requiring generative models to find molecules that are similar to zaleplon but have a different molecular formula condition.

The benchmark-defined terms are ECFP4 similarity to zaleplon and formula matching for C19H17N3O2. These terms are aggregated into one scalar score.

No additional structural editing rule is specified by the benchmark.
```

---

## Practical recommendation for the repository

Use two levels of task context depending on the experiment.

### Strict black-box PABLO reproduction

Use only the common `background.md` and `constraints.md`. Keep `prior_knowledge.md` minimal:

```md
# Domain Prior Knowledge

No task-specific prior is exposed in this strict black-box setting. The optimizer observes only candidate SMILES strings and their scalar scores.
```

### Task-aware / grey-box agentic benchmark

Use the task-specific `goal.md` and `prior_knowledge.md` above. This exposes the benchmark-defined objective semantics but avoids adding agent-generated search traces.
