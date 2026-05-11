# 背景

`guacamol_aripiprazole_similarity_demo` 是一个基于 GuacaMol 的 similarity 任务。
任务会把 SELFIES token 配置解码为 SMILES，并按候选分子与 Aripiprazole 的 FCFP4 Tanimoto 相似度打分，同时使用 GuacaMol 的阈值化相似度修饰。

不同于 rediscovery 任务，该目标奖励高相似度，但不要求精确重构目标分子。
