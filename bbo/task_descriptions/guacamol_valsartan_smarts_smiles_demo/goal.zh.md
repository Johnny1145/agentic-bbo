# 目标

最大化 GuacaMol Valsartan SMARTS score。

该 benchmark 将以下 benchmark 定义的 scoring components 合并为一个标量分子 score：

- 存在与 valsartan 相关的 SMARTS pattern；
- logP，并使用中心为 2.0165 的 Gaussian modifier；
- TPSA，并使用中心为 77.04 的 Gaussian modifier；
- Bertz complexity，并使用中心为 896.38 的 Gaussian modifier。

components 使用几何平均聚合。在本地 loss-minimization 接口中，优化 `valsartan_smarts_loss = 1 - valsartan_smarts_score`。
