# 背景

`guacamol_qed_selfies_demo` 是一个基于 GuacaMol goal-directed 任务的分子优化任务。
它只使用本地 BO tutorial ZINC 归档来初始化 SELFIES token 词表和默认有效分子。

任务会把固定长度的 SELFIES token 序列解码为 SMILES，并使用 RDKit QED 打分，对应 GuacaMol 的 `qed_benchmark` 目标。
这是一个确定性的化学信息学打分任务，不是 docking、湿实验或分布学习 benchmark。
