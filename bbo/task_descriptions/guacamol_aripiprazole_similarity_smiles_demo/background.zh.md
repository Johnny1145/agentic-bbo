# 背景

实现来源：`bbo/tasks/scientific/guacamol_smiles.py`，`GUACAMOL_SMILES_TASK_DEFINITIONS["guacamol_aripiprazole_similarity_smiles_demo"]`。

源 benchmark：`guacamol.standard_benchmarks.similarity(name='Aripiprazole', fp_type='FCFP4', threshold=0.75)`。

表示方式：名为 `smiles` 的直接 SMILES 字符串参数。

目标 SMILES 来源：从 `bbo/tasks/scientific/guacamol_selfies.py` 导入的 `ARIPIPRAZOLE_SMILES`。
