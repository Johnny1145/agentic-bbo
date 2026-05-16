# 目标

最小化 `aripiprazole_similarity_loss`，其中：

- `aripiprazole_similarity_loss = 1.0 - aripiprazole_similarity_score`
- `aripiprazole_similarity_score` 是与 Aripiprazole 的 FCFP4 Tanimoto 相似度，并在 GuacaMol 阈值 `0.75` 处截断

任何达到或超过阈值的解码分子都会获得最高分。
