# 领域先验

这是一个 GuacaMol median-molecule benchmark。GuacaMol 将 median-molecule 任务描述为冲突任务，需要同时最大化与多个分子的相似度。

对该任务，benchmark 定义的目标分子是 camphor 和 menthol。scoring function 使用 ECFP4 相似度和几何平均；因此，如果与任一目标的相似度很弱，标量 score 都会较低。

benchmark 不指定额外的结构编辑规则。
