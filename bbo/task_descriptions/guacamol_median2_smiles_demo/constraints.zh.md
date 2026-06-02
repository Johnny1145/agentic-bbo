# 约束

实现来源：`bbo/tasks/scientific/guacamol_smiles.py`。

输入参数：`StringParam("smiles", default="", min_length=0, max_length=512)`。

无效或空 SMILES 的 score 为 `0.0`，loss 为 `1.0`。

schema default 的角色是 `schema_only_not_initial_population`。
