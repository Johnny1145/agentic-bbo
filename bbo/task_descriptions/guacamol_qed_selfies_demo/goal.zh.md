# 目标

最小化 `guacamol_qed_loss`，其中：

- `guacamol_qed_loss = 1.0 - guacamol_qed_score`
- `guacamol_qed_score` 是解码分子的 RDKit QED

loss 越低，表示生成分子的 QED 越高。
