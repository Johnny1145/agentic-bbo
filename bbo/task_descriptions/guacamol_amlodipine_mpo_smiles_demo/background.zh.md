# 背景

该任务是一个 GuacaMol goal-directed 分子设计 benchmark，候选对象是有效分子结构。

候选分子表示为一个 SMILES 字符串。evaluator 会将每个有效分子映射为 [0, 1] 范围内的标量 GuacaMol score。benchmark 目标是在任务特定 scoring function 下找到高标量分数的分子。

GuacaMol 将这些 goal-directed 任务描述为：模型需要针对预定义 scoring function 生成高分分子。有些任务会把若干分子性质组合成 multi-property objective (MPO)，但 benchmark 仍然只返回一个标量分子分数。

本描述只记录 benchmark 层面的信息，不包含 agent 发现的搜索策略或无来源支持的药物化学启发式建议。
