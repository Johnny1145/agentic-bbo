# 约束

提交必须提供一个 SMILES 字符串作为 `smiles` 的值。

SMILES 字符串必须能被 RDKit 解析为有效分子。无效或空 SMILES 字符串会得到最低分。

benchmark 返回 [0, 1] 范围内的标量 score。在本地 loss-minimization 接口中，优化的 loss 是 `1 - guacamol_score`；因此，最小化 loss 等价于最大化 GuacaMol score。

除非实验被显式配置为 task-aware 或 grey-box 设置，否则不要假设可以访问隐藏的 scoring components。
