# 编码清单

更新时间：2026-04-19 CST

本清单固定 `pfns4bo` 的输入编码规则，执行 agent 不应在远程临时改写。

## 1. numeric task 路线

适用：

- `branin_demo`
- `her_demo`
- `hea_demo`
- `bh_demo`

规则：

- 使用 task 原始 numeric search space
- 按 PFNs4BO 连续接口要求做归一化
- 保持 seed 控制下的确定性

## 2. OER pool-based 路线

适用：

- `oer_demo`

候选来源：

- 从 task 可行搜索空间中构造候选池
- pool size 默认 `256`

编码规则：

- `Metal_1`
- `Metal_2`
- `Metal_3`

必须 one-hot。

下列数值和整数特征必须做 min-max 归一化：

- `Metal_1_Proportion`
- `Metal_2_Proportion`
- `Metal_3_Proportion`
- `Hydrothermal Temp degree`
- `Hydrothermal Time min`
- `Annealing Temp degree`
- `Annealing Time min`
- `Proton Concentration M`
- `Catalyst_Loading mg cm -2`

列顺序必须稳定。

## 3. Molecule pool-based 路线

适用：

- `molecule_qed_demo`

候选来源：

- 原始 SMILES 列表
- pool size 默认 `256`

固定描述符集：

- `MolWt`
- `MolLogP`
- `TPSA`
- `NumHAcceptors`
- `NumHDonors`
- `NumRotatableBonds`
- `RingCount`
- `FractionCSP3`

规则：

- 使用 RDKit 提取描述符
- 对每个 descriptor 做 dataset-level min-max 归一化
- 非法 SMILES 不允许静默伪造 descriptor
- 若 descriptor 提取失败，必须记录 blocker

## 4. 可复现性要求

必须保证：

- 候选池抽样受 seed 控制
- 编码结果稳定
- 同一历史 replay 后，后续候选顺序稳定
- tie-break 受 seed 控制

## 5. 失败时的记录要求

若 OER 或 molecule 的 pool-based 路线失败，必须在 `exp_result.md` 记录：

- 候选池来源
- 编码失败的具体阶段
- 是否为 RDKit / 模型 / 归一化 / 设备问题
- 下一步建议
