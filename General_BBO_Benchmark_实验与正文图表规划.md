# General BBO Benchmark：实验与正文图表规划

> 🎯 **核心问题：**不依赖额外领域先验和现成 BBO 工具的通用 Model + Harness，能否直接在连续向量、混合配置、工程参数和分子字符串上完成有效的序贯黑箱优化？

正文只保留三组实验：**Bare Agent 主实验、Prior 实验、Tool 实验**。三组实验分别回答通用 Agent 本身有没有 BBO 能力、语义与领域知识能否改善搜索、分析和优化工具能否修复 Bare Agent 的失败。

# 1. 主实验：Bare General Agent

## 1.1 具体做法

主方法固定为同一个 **Model + Harness + Evaluator Interface**。Agent 能看到任务描述、变量范围、候选格式和全部历史结果，可以使用 harness 原生的 Python、bash 和文件读写；不额外提供 GP、TPE、CMA-ES 等封装工具，不提供领域文档，不使用任务专用 prompt。Agent 如果自行写分析代码或实现搜索逻辑，仍算 Bare Agent。

## 1.2 Baselines

| Baseline | 使用范围 | 作用 |
|-|-|-|
| **Random Search** | 全部任务 | 最低能力基线 |
| **Sobol** | HPO、DB、EDA 的标准化数值空间 | 低差异覆盖基线 |
| **GP-EI** | HPO、DB、EDA 中可编码为连续/混合向量的任务 | 经典 surrogate-based BO |
| **TPE** | HPO、DB、EDA | 混合变量空间强基线 |
| **CMA-ES / TuRBO** | DB、EDA 及连续子空间 | 进化搜索与高维局部 BO 基线 |
| **Gitbo** | DB、EDA 及连续子空间 | 代表性 learned based BO |
| **Graph-GA** | Molecule | 分子领域强基线 |
| **OPRO-style Pure LLM** | 全部可表达为文本候选的任务 | 区分纯 LLM 提点与完整 Agent harness |
| **Bare General Agent** | 全部任务 | 本文主方法 |

同一任务内使用完全相同的初始化点和 evaluation budget。无法自然处理某种表示的方法标记为 **N/A**，不记为零分。

### 完整实验矩阵与当前运行状态

| 任务族 | 要跑的任务 | 任务设置 / 每个方法 run 数 | Random Search  <br/>原有 | Optuna TPE  <br/>原有 | pycma  <br/>原有 | GP-EI  <br/>原有 | Sobol Search  <br/>新增 | BoTorch TuRBO  <br/>新增 | GitBO  <br/>待提供实现 | Graph-GA  <br/>分子原有 | LLAMBO  <br/>原有 LLM | OPRO  <br/>原有 LLM | Codex  <br/>Harness | Claude Code  <br/>Harness | Nanobot  <br/>Harness |
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
| **BBOPlace** | `adaptec1`  <br/>`adaptec2`  <br/>`adaptec3`  <br/>`adaptec4`  <br/>`bigblue1`  <br/>`bigblue2`  <br/>`bigblue3`  <br/>`bigblue4`  <br/>`superblue1`  <br/>`superblue3`  <br/>`superblue4`  <br/>`superblue5`  <br/>`superblue7`  <br/>`superblue10`  <br/>`superblue16`  <br/>`superblue18` | 16 个 placement tasks；`n_macro=32`；seeds `1,2`；50 init + 200 opt；**32 runs / 方法** | ✅ 32/32 | ✅ 32/32 | ✅ 32/32（popsize=10 历史）  <br/>⏳ 0/32（popsize=2 重跑；待 OPRO strict 结束） | ✅ 32/32 | ⏳ 0/32（待 OPRO strict 结束） | ⏳ 0/32（待 OPRO strict 结束） | ⏸️ 待实现与适配 | — | — | 🟡 0/32 完成；12/32 已启动，1,849/8,000 trials；4 个首轮中断，首轮仍运行 | ✅ 历史 no-tool：32/32 完整；8,000/8,000 trials | ⚠️ 历史 aligned 尝试：0/32；32 失败 | ⚠️ 历史 no-tool：14/32 完整、18 部分；6,493/8,000 trials |
| **分子 / GuacaMol** | `guacamol_amlodipine_mpo_smiles_demo`  <br/>`guacamol_fexofenadine_mpo_smiles_demo`  <br/>`guacamol_median1_smiles_demo`  <br/>`guacamol_median2_smiles_demo`  <br/>`guacamol_osimertinib_mpo_smiles_demo`  <br/>`guacamol_perindopril_mpo_smiles_demo`  <br/>`guacamol_ranolazine_mpo_smiles_demo`  <br/>`guacamol_sitagliptin_mpo_smiles_demo`  <br/>`guacamol_valsartan_smarts_smiles_demo`  <br/>`guacamol_zaleplon_mpo_smiles_demo` | 10 个 direct-SMILES tasks；seeds `1,2`；50 init + 200 opt；**20 runs / 方法** | — | — | — | — | — | — | ⏸️ 待确认是否支持 SMILES | ✅ 20/20 | — | ⚠️ 0/20 完成；1,428/5,000 trials；20/20 首轮候选生成中断，待自动续跑 | ⚠️ 历史 no-tool：1/20 完整、19 部分；2,465/5,000 trials | ⚠️ 历史 aligned 尝试：1/20；19 失败 | ⚠️ 历史 no-tool：0/20 完整、20 部分；1,371/5,000 trials |
| **DBTune** | `knob_http_surrogate_job_5`  <br/>`knob_http_surrogate_job_all`  <br/>`knob_http_surrogate_pg_5`  <br/>`knob_http_surrogate_pg_20`  <br/>`knob_http_surrogate_sysbench_5`  <br/>`knob_http_surrogate_sysbench_all` | 6 个 surrogate tasks；seeds `1,2`；50 init + 200 opt；**12 runs / 方法** | ✅ 12/12 | ✅ 12/12 | ✅ 12/12（popsize=10 历史）  <br/>⏳ 0/12（popsize=2 重跑；待 OPRO strict 结束） | ✅ 12/12 | ⏳ 0/12（待 OPRO strict 结束） | ⏳ 0/12（待 OPRO strict 结束） | ⏸️ 待实现与适配 | — | — | ✅ 12/12；3,000/3,000 trials；0 失败 | ⚠️ 历史 no-tool：8/12 完整、4 部分；2,769/3,000 trials | ⚠️ 历史 aligned 尝试：7/12；5 失败 | ⚠️ 历史 no-tool：9/12 完整、3 部分；2,693/3,000 trials |
| **Synthetic / COCO-BBOB** | `bbob_f01_d10`–`bbob_f24_d10` | 24 个官方 BBOB 10D tasks；seeds `0,1` → instances `1,2`；20 Sobol init + 100 opt；**48 runs / 方法** | ⏳ 0/48（待 OPRO strict 结束） | ⏳ 0/48（待 OPRO strict 结束） | ⏳ 0/48（popsize=2；待 OPRO strict 结束） | ⏳ 0/48（待 OPRO strict 结束） | ⏳ 0/48（待 OPRO strict 结束） | ⏳ 0/48（待 OPRO strict 结束） | ⏸️ 待实现与适配 | — | — | ✅ 48/48 | ✅ 对齐口径 seeds `0,1`：48/48，5,760/5,760 trials；0 失败  <br/>另存 seed `2` 历史：7 完整 + 2 部分，不计入两-seed矩阵 | ✅ 48/48；0 失败 | 🟡 6/48 完整；11/48 已启动；1,106/5,760 trials；1 个可续跑中断；仍在运行 |
| **HPO / LLAMBO-Bayesmark** | `hpo_bayesmark_{breast,wine,iris,digits,diabetes}_{random_forest,svm,decision_tree,mlp_sgd,adaboost}` | 25 个 dataset-model tasks；seeds `0,1`；5 个固定 LLAMBO init + 25 opt；**50 runs / 方法** | ⏳ 0/50（待 OPRO strict 结束） | ⏳ 0/50（待 OPRO strict 结束） | ⏳ 0/50（popsize=2；待 OPRO strict 结束） | ⏳ 0/50（待 OPRO strict 结束） | ⏳ 0/50（待 OPRO strict 结束） | ⏳ 0/50（待 OPRO strict 结束） | ⏸️ 待实现与适配 | — | — 不跑（按要求） | ✅ 50/50；1,500/1,500 trials；0 失败 | — 未安排 | — 未安排 | — 未安排 |

> 💡 **Harness 状态说明：**表中 Harness 的“部分”“中断”或“失败”，主要指优化已经进行了一段时间，但 Agent 随后反复生成与历史完全重复的候选，或持续输出不满足任务格式/约束的非法解。达到预设的连续无效尝试阈值后，Harness 主动停止该 run，因此这些状态不应简单理解为基础设施故障，也不代表从第一轮就无法运行。统计时应保留停止前已完成的有效 trials，并单独报告完整 run 数、部分 run 数、duplicate rate、invalid-candidate rate 和实际完成的 trials。

## 1.3 主结果保留什么

- **Best-so-far curves：**展示整个搜索过程，而不是只看终点。
- **Final Best：**统一预算结束时的最优结果。
- **Mean Rank 与 Win Rate：**在每个任务内排名后跨任务汇总。
- **Coverage：**能够直接运行的 benchmark 任务比例。
- **可靠性：**invalid、duplicate 和 evaluator failure rate。

# 2. Prior 实验：任务信息是否改善搜索

## 2.1 具体做法

固定模型、harness、工具权限、初始化点和预算，只改变 Agent 看到的任务信息。使用以下 6 个任务，每个条件跑 5 seeds：

- HPO：digits_mlp_sgd
- DB：job_all
- EDA：superblue16
- Molecule：amlodipine_mpo、valsartan_smarts
- Synthetic control：BBOB f15

| 条件 | Agent 能看到的信息 |
|-|-|
| **P0 Minimal / Anonymous** | 匿名变量、类型、范围、候选格式、目标方向 |
| **P1 Semantic** | P0 + 真实变量名、单位、参数类别和目标的现实含义 |
| **P2 Domain Prior** | P1 + 领域文档、经验规则、常见配置、变量关系和合法性知识 |

## 2.2 Baselines 与输出

**Baseline：**P0；主要比较 P1−P0 和 P2−P0。正文只画 Early Best（25% budget）、Final Best 和完整 best-so-far curves。BBOB f15 是无语义 control：如果 P1/P2 在它上面同样显著提升，需要检查提示长度、随机性或额外信息量是否成为混杂因素。

# 3. Tool 实验：工具能否修复 Bare Agent

## 3.1 具体做法

固定模型、harness、任务描述、初始化点和预算，只改变可用工具。使用以下 6 个任务，每个条件跑 5 seeds：

- 低维连续：BBOB f01
- 高维 / 多峰：BBOB f15
- 混合 HPO：digits_mlp_sgd
- DB：job_all
- EDA：superblue16
- Molecule：amlodipine_mpo

| 条件 | 提供的工具 | 对应 baseline |
|-|-|-|
| **T0 Bare Agent** | 不提供额外工具 | 主实验 Bare Agent |
| **T1 Analysis Tools** | best-so-far、重复检测、候选距离、搜索覆盖、变量趋势、停滞检测 | T1−T0 检验“稳定分析历史”本身的价值 |
| **T2 Single Optimizer** | 连续任务用 GP-EI；混合空间用 TPE；高维连续用 CMA-ES/TuRBO；Molecule 用 Graph-GA mutation | T2−T1 检验单一匹配 backend 是否已经足够 |
| **T3 Tool Portfolio** | Random/Sobol、local perturbation、GP-EI、TPE、CMA-ES/TuRBO、Graph-GA mutation、validator；Agent 自主选择、修改或拒绝候选 | T3−T2 检验自主工具调度的额外价值 |

正文输出 best-so-far curves、Final Best、token / wall-clock / tool calls，以及 duplicate、invalid、search coverage、longest stagnation 和 recovery probability。T3 只有在性能超过 T2、且工具调用随任务或搜索阶段发生有意义切换时，才能支持“tool orchestration”结论。

# 4. 正文图表

正文使用 **2 张表 + 5 张图**。Benchmark 组成由 Table 1 承担，Figure 1 不再重复画“任务 → Agent → 候选表示”的普通流程图。

## Figure 1：Why General Agentic BBO?

> 💡 **新定位：**Figure 1 是 motivation + problem formulation 图，不是 benchmark 数据流图。它要让读者一眼看到：现实 BBO 同时存在表示、信息和反馈异质性；固定优化器只能覆盖局部；General Agentic BBO 的价值是用同一个决策循环适配这些差异，并把 Bare、Prior、Tool 三种能力拆开评价。

![](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=M2ZjNDFhNjllNDg1MWZiYzIwNzgxYTg0ZWY0N2EyZTlfMWIzMTBmZWEyZTY1MzEyZDE0MjVjZGM2OGIwZmI5NTZfSUQ6NzY3MDUyMjkzNzAxNTMzOTk5NF8xNzg2MDA3ODIxOjE3ODYwMTE0MjFfVjM)

类似模仿这种改一改

## Figure 2：Main Optimization Curves

画 4–5 个代表任务的 best-so-far curves：digits_mlp_sgd、job_all、superblue16、amlodipine_mpo，再加一个 Bare Agent 明显失败的任务。每个 panel 最多保留 Random、最强传统 baseline、最强 LLM baseline、Bare Agent 和必要的领域方法。横轴为 evaluation，纵轴用该任务原始 objective；阴影为 seed 区间。

**必须回答：**Bare Agent 是 early-stage 强、后期追赶还是提前停滞；最终差距来自整个搜索过程还是少数关键 evaluation。

## Figure 3：Mean Rank over Budget

在 0%、25%、50%、75%、100% normalized budget 上计算每个任务当前 best-so-far rank，再跨任务平均。优先分 HPO、DB、EDA、Molecule 四个小 panel，避免 N/A 造成不公平；正文主表另外报告 Coverage。

**必须回答：**Bare Agent 的跨任务排名是否随预算持续改善，哪些方法早期领先但后期停滞。

## Figure 4：Prior Effects

Panel A：3–4 个任务的 P0/P1/P2 best-so-far curves。Panel B：全部 6 个任务的 paired forest plot，每行一个任务，同时画 Early improvement 与 Final improvement，相对 P0 的提升统一成“正数更好”，带置信区间和零线。

**必须回答：**变量语义与完整领域知识分别带来多少收益；收益是集中在 cold start，还是能保留到最终；哪些任务出现负迁移。

## Figure 5：Tool Effects

Panel A：3–4 个任务的 T0–T3 best-so-far curves。Panel B：相对 T0 的 Final Best forest plot。Panel C：行为 heatmap，列为 duplicate、invalid、coverage、stagnation、recovery，所有方向统一成“正数表示改善”。

**必须回答：**Analysis、Single Optimizer 和 Portfolio 分别解决什么问题；Portfolio 是否真的通过自主调度获得额外收益，而非只增加工具和计算。

## Table 1：Benchmark Summary

列出 Domain、#Tasks、Representation、Variable Types、Constraints、Evaluator 和 Budget。重点突出 mixed vector、configuration、structured parameters、SMILES 四种候选表示；不再用 Figure 1 重复介绍任务数量。

## Table 2：Main Results

列出 Method、四个任务族的 Mean Rank、Overall Mean Rank、Win Rate、Worst-Family Rank 和 Coverage。完整 per-task Final Best、seed 分布、所有曲线、BBOB 诊断、工具调用时间线和成本结果放附录。
