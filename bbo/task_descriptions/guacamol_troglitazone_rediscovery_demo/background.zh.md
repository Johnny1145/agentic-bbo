# 背景

`guacamol_troglitazone_rediscovery_demo` 是一个基于 GuacaMol 的 rediscovery 任务。
任务会把固定长度的 SELFIES token 配置解码为 SMILES，并按候选分子与 Troglitazone 的 ECFP4 Tanimoto 相似度打分。

该任务完全本地、确定性运行；ZINC 归档只用于初始化 SELFIES 词表。
