# 背景

`guacamol_fexofenadine_mpo_demo` 是一个基于 GuacaMol 的多性质优化任务。
任务会把 SELFIES token 配置解码为 SMILES，并组合三个评分分量：与 Fexofenadine 的 atom-pair 相似度、高 TPSA、低 logP。

该实现遵循 GuacaMol `hard_fexofenadine` benchmark 的结构，同时暴露固定类别型 SELFIES 搜索空间。
