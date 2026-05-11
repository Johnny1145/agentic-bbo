# 背景

`guacamol_median1_demo` 是一个基于 GuacaMol 的 median molecule 任务。
任务会把 SELFIES token 配置解码为 SMILES，并按候选分子与 camphor 和 menthol 的 ECFP4 Tanimoto 相似度几何平均打分。

目标不是精确重构任一分子，而是生成同时共享两者结构特征的中间分子。
