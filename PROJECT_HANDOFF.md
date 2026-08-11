# Agentic BBO v4：运行、实验与接管摘要

最后更新：2026-08-07 AWST（UTC+8）

这份文档是后续会话的首要接管入口。它记录当前项目的权威目录、用户已经确认的实验口径、任务与算法、正式 workflow、结果位置、运行状态、审计方式和已知问题。

后续会话开始时应先阅读：

1. 本文件 PROJECT_HANDOFF.md；
2. EXPERIMENT_MATRIX.md（每个任务族 × 方法的完成度）；
3. 对应正式 workflow 下的 setting.json（机器可读、不可凭记忆改动）。

## 1. 权威项目与机器

当前唯一应该继续开发和运行的目录：

    /home/trx/agentic-bbo-multiharness-benchmark-v4

远端正式实验机：

    ssh -p 223 trx@10.26.17.10
    remote repo: /home/trx/agentic-bbo-multiharness-benchmark-v4

本机和远端使用相同的 v4 绝对路径。近期 OPRO、对齐数值 baseline、GIT-BO 和 HPO 三 harness 的正式结果以远端目录为准。

模型服务：

- SGLang：127.0.0.1:18301/v1
- served model：qwen3.5-9b
- 模型：Qwen3.5-9B，tensor parallel 8，context length 65536
- Nanobot 使用 Chat Completions。
- Codex 使用 Responses；runtime 为每次调用创建 loopback compatibility proxy，再转到 SGLang Chat Completions。
- Claude Code 使用 Messages；runtime 同样使用每次调用的 compatibility proxy。
- 远端用户级可执行文件位于 /home/trx/.npm-global/bin 和 /home/trx/.local/bin。后台脚本已显式补 PATH。

重要目录状态：

- /home/trx/agentic-bbo-nanobot-benchmark-v2 是保留的历史 v2。
- 系统中还存在若干相似命名目录（无版本、v3、下划线 v4），它们不是当前权威项目，也不是 v4 的符号链接。
- 不要因为名称相似而同步、运行或删除这些目录。
- 任何目录删除都必须先向用户列出精确目标并取得确认。
- 当前 v4 不是 Git worktree，没有 Git commit/status 可作为回滚依据；修改前后要依赖明确的文件 diff、测试和备份。

## 2. 用户已经确认、后续不得静默改变的口径

1. 正式比较统一使用两个 seed，不使用 5 seeds。
2. BBOB 使用 seeds 0,1，对应 COCO instances 1,2。
3. BBOPlace、DBTune、GuacaMol 的历史矩阵使用 seeds 1,2。
4. HPO 使用 seeds 0,1。
5. pycma 的正式新结果使用 popsize=2；popsize=10 结果只作为历史保留，不能混进 popsize=2 矩阵。
6. GP-EI 与 BoTorch TuRBO 的正式内部候选数统一为 2048，batch size 为 1。
7. TuRBO 必须使用已 vendored、带来源信息的 BoTorch 官方 TuRBO-1 tutorial core，不能自行另写一个“类似 TuRBO”的算法。
8. BBOB 初始化固定为 20 个 task-owned scrambled-Sobol 点，然后 100 个 optimizer 点。
9. HPO 初始化必须严格使用 LLAMBO release 的 5 个固定配置，然后 25 个 optimizer 点；不能用 harness 或 baseline 自己另采 5 个随机点。
10. BBOPlace、DBTune、GuacaMol 的 strict OPRO/aligned 条件重放数值 baseline 的完整初始化 observation（config、objective、metrics），不重新评估。
11. Harness 条件是 no-tool：不注入 benchmark BBO tools 或 BBO skills，但保留 Codex、Claude Code、Nanobot 各自原生工具和 agent loop。
12. 正式 harness 禁止随机 fallback。无效输出按配置重试，仍失败则诚实记为失败。
13. 用户明确要求：中断或没跑完的实验应诚实记录，不要为了补齐表格自动重跑。
14. 新 workflow 可以具有 resume 能力以防同一次任务进程中断，但不能未经用户确认启动额外 rerun pass。
15. LLAMBO baseline 当前不安排正式矩阵；HPO 的任务与初始化来自 LLAMBO，但“LLAMBO 算法”本身按要求不跑。
16. GIT-BO 不跑 direct-SMILES 分子任务；其余数值任务均纳入。
17. v4 的 synthetic 正式任务只保留官方 COCO-BBOB，不使用旧的 BoTorch toy synthetic 函数。

## 3. 当前正式任务集

| 任务族 | 任务数 | Seeds | 每 run 预算 | 暴露方式 | 主要实现 |
|---|---:|---|---|---|---|
| BBOPlace | 16 | 1,2 | 50 init + 200 opt = 250 | full-prior | HTTP evaluator，n_macro=32，64D 坐标 |
| GuacaMol direct-SMILES | 10 | 1,2 | 50 init + 200 opt = 250 | full-prior | 字符串 SMILES，Graph-GA/LLM harness |
| DBTune | 6 | 1,2 | 50 init + 200 opt = 250 | full-prior | HTTP sklearn surrogate |
| COCO-BBOB | 24 | 0,1 | 20 init + 100 opt = 120 | restricted-prior | official coco-experiment==2.8.2，10D |
| LLAMBO-Bayesmark HPO | 25 | 0,1 | 5 init + 25 opt = 30 | full-prior | 本地固定数据资产 + scikit-learn 5-fold CV |

### 3.1 BBOPlace

16 个 benchmark：

    adaptec1, adaptec2, adaptec3, adaptec4
    bigblue1, bigblue2, bigblue3, bigblue4
    superblue1, superblue3, superblue4, superblue5
    superblue7, superblue10, superblue16, superblue18

统一 n_macro=32。服务端口池由 workflow/40 的 runner 管理，常用端口包括 8270、8280–8287、8070。

已知数据问题：在当前 n_macro=32 选择规则下，superblue3 和 superblue10 的 HPWL 是精确常数 0，superblue18 近似常数。保留原始 run 以维持矩阵对齐，但聚合算法性能时要排除或单独标注这些退化任务。

### 3.2 GuacaMol direct-SMILES

10 个任务：

    amlodipine_mpo, fexofenadine_mpo, median1, median2,
    osimertinib_mpo, perindopril_mpo, ranolazine_mpo,
    sitagliptin_mpo, valsartan_smarts, zaleplon_mpo

完整 task id 都以 guacamol_ 开头并以 _smiles_demo 结尾，详见 EXPERIMENT_MATRIX.md。

这些任务不是有界数值向量，因此数值 baseline 和 GIT-BO 不跑。正式传统 baseline 是 Graph-GA；OPRO 和三个 harness 可以直接提出 SMILES。

### 3.3 DBTune

6 个正式 HTTP surrogate 任务：

    knob_http_surrogate_job_5
    knob_http_surrogate_job_all
    knob_http_surrogate_pg_5
    knob_http_surrogate_pg_20
    knob_http_surrogate_sysbench_5
    knob_http_surrogate_sysbench_all

默认服务为 127.0.0.1:8090。job_all 和 sysbench_all 是 196D；LLM prompt 使用更小的 4 个 top + 4 个 rank-uniform 历史子集，避免超过 65536 context。

### 3.4 COCO-BBOB

任务为 bbob_f01_d10 到 bbob_f24_d10：

- official COCO bbob suite；
- dimension=10，bounds=[-5,5]；
- seed 0 → instance 1，seed 1 → instance 2；
- objective 为 loss，最小化；
- optimizer-facing task id/描述匿名化为 restricted_task_xxx，不能向 agent 泄漏函数名称；
- 同一 seed 的 20 个 scrambled-Sobol 初始化配置在 24 个函数和所有算法间一致。

正式 setting：

    workflow/50_bbob24_d10_i123_init20_opt100_20260802/setting.json

注意该目录名仍含 i123，但正式 setting_version 已是 i12，seed 2 历史保留但不计入当前两-seed矩阵。

### 3.5 LLAMBO-Bayesmark HPO

25 个任务是 5 datasets × 5 models：

    datasets: breast, wine, iris, digits, diabetes
    models: random_forest, svm, decision_tree, mlp_sgd, adaboost
    task id: hpo_bayesmark_<dataset>_<model>

- breast/wine/iris/digits 为 classification，最大化 5-fold CV accuracy。
- diabetes 为 regression，最小化 5-fold CV MSE。
- 数据划分固定为 published_llambo_80_20_random_state_0。
- 同一 model + seed 的 5 个 LLAMBO release 初始化配置在 5 个 dataset 和所有算法间一致。
- task-owned 初始化只支持 seeds 0..9；正式矩阵只用 0,1。
- MLP 按论文使用 8D 空间；release 里缺失的 tol 和 validation_fraction 用 release 评估时实际生效的默认值补齐。

权威说明：

    docs/hpo_bayesmark.md
    bbo/tasks/hpo/
    bbo/tasks/hpo/assets/llambo_release_initial_samples.json

## 4. 算法与关键超参数

| 方法 | 正式实现与设置 | 适用范围 |
|---|---|---|
| random_search | seeded independent uniform；先消费共享初始化；batch=1 | 所有数值任务 |
| optuna_tpe | Optuna TPESampler；startup_trials=5；batch=1 | 所有数值任务 |
| pycma | pycma.CMAEvolutionStrategy；sigma_fraction=0.18；popsize=2；用共享初始化中的 best config warm start | 所有数值任务 |
| gp_ei | BoTorch SingleTaskGP + ExpectedImprovement；raw_samples=2048；10 acquisition restarts；xi=0；CPU；batch=1 | 所有数值任务 |
| sobol_search | torch.quasirandom.SobolEngine；scramble=true；batch=1 | 所有数值任务 |
| botorch_turbo | pinned official BoTorch TuRBO-1 tutorial core；Thompson sampling；n_candidates=2048；CPU；batch=1 | 所有数值任务 |
| git_bo | official TabPFN v2 differentiable-input adapter；1 estimator；r=min(10,d)；beta=2.33；2048 gradient points + 2048 candidates | 除 direct-SMILES 外所有任务 |
| graph_ga | 分子图遗传算法 | GuacaMol |
| OPRO | Qwen3.5-9B；thinking；temperature=0.8；1 candidate；4 generation/parse rounds；no fallback | BBOB、HPO、BBOPlace、DBTune、GuacaMol |
| Codex harness | native Codex Responses protocol；no injected BBO tools/skills；native tools retained；no fallback | 所有 task family |
| Claude Code harness | native Messages protocol；其余同上 | 所有 task family |
| Nanobot harness | Chat Completions protocol；其余同上 | 所有 task family |

### 4.1 GP-EI / TuRBO 对齐

正式 workflow 明确覆盖 CLI 默认，GP-EI raw_samples 和 TuRBO n_candidates 都固定为 2048。通用代码也支持 task-resolved 规则 min(5000,max(2048,200*d))，但当前正式 BBOB/HPO setting 以及 workflow/54 都写死 2048；不要在同一正式矩阵中混用另一个候选预算。

GP-EI：

- 连续/整数参数先经 continuous converter 编码；混合类别空间使用 one-hot。
- 输入特征按已有训练数据做标准化，同时把 bounds 转换到同一标准化坐标。
- objective 由 BoTorch Standardize(m=1) 处理。
- 最小化 objective 在拟合/采集时转成最大化符号。
- optimize_acqf 使用 q=1、num_restarts=10、raw_samples=2048、maxiter=200。
- 候选 decode/coerce 回原始 task search space 后才送 evaluator。

GP-EI 实现 caveat：CLI、setting 和 constructor 会记录 alpha=1e-6 与 n_restarts_optimizer=0，但当前 bbo/algorithms/model_based/gp_ei.py 的 SingleTaskGP/mll 构造没有消费这两个字段。已完成矩阵彼此仍使用同一代码和 setting，但报告算法时不能声称 alpha 已作为 observation noise/jitter 生效。若未来修复接线，必须使用新 setting_version 和新结果目录，不能与现有结果混合。

TuRBO：

- 使用 UnitCubeSearchSpaceConverter 映射到 [0,1]^d。
- 尊重 task metadata 中声明的 linear/log/logit parameter transforms。
- 生成候选后 decode、clip、整数 round/coerce 回原始物理空间。
- 官方 TuRBO-1 state：初始 trust-region length 0.8，min 0.0078125，max 1.6；BBOB d10/q1 下 success tolerance=10、failure tolerance=10。

Sobol 与 GIT-BO 同样使用 unit-cube converter 和 task-declared transforms；GIT-BO 的 suggestion_metadata 会记录 transform、候选池、Fisher eigenvalues、rank、UCB 与依赖 commit。

### 4.2 GIT-BO fallback 的含义

GIT-BO 遇到 startup、退化目标、非有限数值或离散候选耗尽时，会显式记录为 Sobol fallback，不会伪装成 GIT-BO acquisition。正式结果中必须分别统计 acquisition 和 fallback。

权威文档：

    docs/GIT_BO_ADAPTER.md
    bbo/algorithms/model_based/git_bo.py

## 5. 初始化与公平比较

所有可比较方法必须共享相同 task/seed 初始化前缀。

| 任务族 | 初始化 |
|---|---|
| BBOB | task-owned 20 个 scrambled-Sobol configs；同 seed 跨函数、baseline、harness 一致 |
| HPO | task-owned 5 个 LLAMBO release configs；同 model+seed 跨 dataset、baseline、harness 一致 |
| BBOPlace/DBTune | strict aligned 运行重放 random_search baseline 的 50 个完整 observations |
| GuacaMol | strict aligned 运行重放 Graph-GA 的 50 个完整 observations |

对 HPO native harness 已验证：

- GeneralAgent 在 ask() 时优先读取 task.spec.metadata.benchmark_protocol.initialization；
- 前 5 次不会调用模型，也不会生成自己的 random init；
- 第 6 条开始才进入 optimization 和真实 agent call；
- 2026-08-06 远端 Codex、Claude Code、Nanobot 首个 run 的前 5 configs 均与 LLAMBO metadata 完全相等。

## 6. 正式 workflow 索引

### workflow/40：核心 multifamily runner

目录：

    workflow/40_nanobot_no_tool_multifamily_20260724/

关键入口：

    run_harness_no_tool_multifamily.py
    run_baseline_multifamily.py
    run_opro_multifamily.py
    shared_initialization.py

它是所有新 workflow 复用的核心 runner。当前 runner 已支持 synthetic/BBOB、DBTune、molecule、BBOPlace 和 HPO。

注意：workflow/40/README.md 中仍有“96 runs”“旧 synthetic 数量”等历史描述。当前计数必须以本 handoff、EXPERIMENT_MATRIX.md 和 workflow/50–56 的 setting.json 为准。

### workflow/50：BBOB 三 harness + 六数值 baseline

    workflow/50_bbob24_d10_i123_init20_opt100_20260802/

- BBOB f01–f24，10D，seeds 0,1，20+100。
- run_codex.sh、run_claude_code.sh、run_nanobot.sh。
- audit_harness_matrix.py。
- 本机保有三 harness 的正式结果目录。

### workflow/51：远端 BBOB OPRO

    workflow/51_bbob24_d10_i12_opro_remote_20260804/
    outputs/opro_qwen35_9b_bbob24_i12_v1

48/48 已完成。不要再启动 run_until_complete.sh。

### workflow/52：远端 HPO OPRO

    workflow/52_hpo25_seed01_opro_remote_20260804/
    outputs/opro_qwen35_9b_hpo25_seed01_v2

50/50、1500/1500 trials 已完成。v1 和 smoke 目录仅为历史。

### workflow/53：远端 BBOPlace/DBTune/GuacaMol strict OPRO

    workflow/53_remaining64_strict_opro_remote_20260805/
    outputs/opro_qwen35_9b_remaining64_strict_v1

用户决定保留真实中断状态，不再自动续跑：

- DBTune：12/12 完成；
- BBOPlace：9/32 完成，23 partial；
- GuacaMol：0/20 完成，20 partial。

不要为了“补齐 strict”重启该 workflow。

### workflow/54：远端对齐数值 baseline

    workflow/54_aligned_numerical_baselines_remote_20260805/

结果根：

    outputs/bboplace_sobol_turbo_pycma2_seed12_v2
    outputs/dbtune_sobol_turbo_pycma2_seed12_v2
    outputs/bbob24_all6_pycma2_seed01_v2
    outputs/hpo25_all6_pycma2_seed01_v2

当前 EXPERIMENT_MATRIX 显示对应正式数值 baseline 已完成。setting.json 内的 “GitBO implementation pending” 是创建 workflow/54 时的历史文字；GIT-BO 后来在 workflow/55 单独完成，不能因此认为仍未实现。

审计：

    .venv/bin/python workflow/54_aligned_numerical_baselines_remote_20260805/audit_baseline_matrix.py --matrix hpo --root <root>

matrix 可选 bboplace、dbtune、bbob、hpo。

### workflow/55：远端 GIT-BO

    workflow/55_git_bo_remote_20260806/

结果根：

    outputs/bboplace_git_bo_seed12_v1
    outputs/dbtune_git_bo_seed12_v1
    outputs/bbob24_git_bo_seed01_v1
    outputs/hpo25_git_bo_seed01_v1

一次性 run_all.sh，不含自动 rerun。

当前结果：

- DBTune：12/12 完整；
- BBOB：48/48 完整；
- HPO：50/50 完整；
- BBOPlace：31/32 完整；superblue7 seed 2 只有 50/250 初始化 trials，按用户要求不重跑。

### workflow/56：远端 HPO 三 harness（已结束）

    workflow/56_hpo25_seed01_three_harness_remote_20260806/

结果根：

    outputs/codex_qwen35_9b_hpo25_seed01_v1
    outputs/claude_code_qwen35_9b_hpo25_seed01_v1
    outputs/nanobot_qwen35_9b_hpo25_seed01_v1

每个 harness：

- 25 tasks × 2 seeds = 50 runs；
- 5 fixed init + 25 optimizer = 30 trials/run；
- 1500 trials、1250 agent optimization calls；
- jobs=4；
- 无自动 matrix rerun。

2026-08-07 最终审计：

| Harness | started | complete | trials | agent_calls | recorded failures | PID |
|---|---:|---:|---:|---:|---:|---:|
| Codex | 50/50 | 49/50 | 1489/1500 | 1743 | 1 | exited |
| Claude Code | 50/50 | 47/50 | 1473/1500 | 1535 | 3 | exited |
| Nanobot | 50/50 | 50/50 | 1500/1500 | 1369 | 0 | exited |

三条正式进程均已退出。四个不完整 run 按用户要求保留且不自动重跑：Codex `digits_mlp_sgd` seed 0 为 19/30；Claude Code `diabetes_mlp_sgd` seed 1 为 20/30、`diabetes_svm` seed 1 为 28/30、`wine_decision_tree` seed 0 为 15/30。若需复核可在远端运行：

    cd /home/trx/agentic-bbo-multiharness-benchmark-v4
    workflow/56_hpo25_seed01_three_harness_remote_20260806/status.py

查看日志：

    tail -n 50 workflow/56_hpo25_seed01_three_harness_remote_20260806/logs/codex.log
    tail -n 50 workflow/56_hpo25_seed01_three_harness_remote_20260806/logs/claude_code.log
    tail -n 50 workflow/56_hpo25_seed01_three_harness_remote_20260806/logs/nanobot.log

不要再次执行 run_all.sh，除非用户明确要求重新启动。

### workflow/57：本机 Codex 先验消融（已结束，部分不完整）

    workflow/57_prior_ablation_codex_20260806/
    outputs/prior_ablation_v1

- 6 tasks × 2 seeds × P0/P1/P2 = 36 planned runs；
- 10/36 完整，26/36 recorded failure；
- 共同预算分析有 10 个含 optimization step 的 block，BBOPlace 两个 block 仅有初始化；
- 不自动重跑；正式结论和复现 CSV 见 `reports/20260807_main_prior_conclusions/REPORT.md`。

### workflow/58：工具消融（仅准备，未开跑）

缺少 `approval.json`，无正式结果，不纳入结论。

## 7. 当前实验完成度摘要

完整细表只维护在 EXPERIMENT_MATRIX.md。这里给出接管所需的高层结论。

### 数值 baseline

- BBOPlace：Random、TPE、GP-EI、Sobol、TuRBO、pycma(popsize=2) 全部 32/32；GIT-BO 31/32 + 1 partial。
- DBTune：上述六数值 baseline 和 GIT-BO 全部 12/12。
- BBOB：上述六数值 baseline 和 GIT-BO 全部 48/48。
- HPO：上述六数值 baseline 和 GIT-BO 全部 50/50。
- GuacaMol：Graph-GA 20/20；数值 baseline 不适用。

### OPRO

- BBOB 48/48。
- HPO 50/50。
- DBTune 12/12。
- BBOPlace 9/32 complete、23 partial，已终止保留。
- GuacaMol 0/20 complete、20 partial，已终止保留。

### Native harness

- BBOB：Codex、Claude Code、Nanobot 均为 48/48 完整。
- BBOPlace、DBTune、GuacaMol 的历史 no-tool/aligned 结果状态详见 EXPERIMENT_MATRIX；其中部分矩阵不完整，不应写成全部成功。
- HPO：Codex 49/50 完整 + 1 partial；Claude Code 47/50 完整 + 3 partial；Nanobot 50/50 完整；均已退出且不自动重跑。

## 8. 标准结果结构与读取规则

矩阵结果根通常包含：

    setting.json
    planned_tasks.json
    benchmark_summary.json
    _plans/<plan_id>.json

单 run 至少包含：

    run_setting.json
    trials.jsonl
    summary.json

Harness run 还会包含：

    agent_calls.jsonl
    agent_prompts.jsonl
    agent_state/
    agent_workspace/
    llm_logs/
    reasoning_traces/

常见 canonical 路径：

数值 baseline：

    <root>/<exposure_policy>/baseline/<task>/<algorithm>/seed_<seed>/trials.jsonl

native harness：

    <root>/<harness>/<exposure_policy>/no-tool/<task>/<harness>/seed_<seed>/trials.jsonl

读取规则：

- trials.jsonl 才是真实 benchmark evaluation，一行一 trial。
- agent_calls.jsonl 是 agent 调用，不等于 evaluation 数。
- suggestion_metadata.phase 或 metadata.phase 区分 initialization 与 optimization。
- summary.json 是单 run 汇总。
- benchmark_summary.json 是 matrix 计划、成功与失败汇总；运行中会逐步更新。
- run_setting.json 是 resume 的不可变设置。runner 会拒绝把不同模型、endpoint、预算、初始化或 harness 混到同一个 run 目录。
- 判断“完成”不能只看 summary 是否存在；应同时验证 trial 数、全部 status=success、init/opt phase 数和 setting_version。

## 9. 审计与测试

本机最近验证：

    .venv/bin/python -m pytest -q tests/test_native_harness_workflow.py tests/test_hpo_bayesmark.py tests/test_opro_multifamily_workflow.py

结果：34 passed，1 skipped。

远端同步后验证：

    .venv/bin/python -m pytest -q tests/test_native_harness_workflow.py tests/test_hpo_bayesmark.py

结果：20 passed，1 skipped。

HPO 三 harness dry-run 计划应为：

    planned runs: 150
    planned evaluations: 4500
    planned agent optimization steps: 3750

GIT-BO compatibility smoke：

    workflow/55_git_bo_remote_20260806/validate_task_compatibility.py

正式审计脚本：

    workflow/50_bbob24_d10_i123_init20_opt100_20260802/audit_harness_matrix.py
    workflow/51_bbob24_d10_i12_opro_remote_20260804/audit_opro_matrix.py
    workflow/52_hpo25_seed01_opro_remote_20260804/audit_opro_matrix.py
    workflow/53_remaining64_strict_opro_remote_20260805/audit_opro_matrix.py
    workflow/54_aligned_numerical_baselines_remote_20260805/audit_baseline_matrix.py
    workflow/55_git_bo_remote_20260806/audit_git_bo_matrix.py

审计脚本返回非零不一定表示代码错误；对用户明确保留的 partial 矩阵，这是诚实反映“不完整”。

## 10. 后续会话的标准接管步骤

1. 确认 cwd 是 /home/trx/agentic-bbo-multiharness-benchmark-v4。
2. 阅读本文件和 EXPERIMENT_MATRIX.md。
3. 若要复核 HPO harness，SSH 到远端执行 workflow/56/status.py；当前三个 PID 均已退出。
4. 不要根据旧 snapshot 判断状态；以 50 runs × 30 trials 的最终审计为准。
5. EXPERIMENT_MATRIX.md 已记录四个 partial/failed run；不要自动重跑。
6. 若要启动新实验，先给用户列出任务、seed、初始化、optimizer budget、候选数、并发、结果根和是否 resume，得到确认后再启动。
7. 任何正式 setting 变化都新建 workflow/结果目录和新 setting_version，不覆盖旧结果。
8. 不要清理相似目录或旧 outputs，除非用户明确确认精确删除清单。

## 11. 最重要的 source-of-truth 文件

    PROJECT_HANDOFF.md
    EXPERIMENT_MATRIX.md
    reports/20260807_main_prior_conclusions/REPORT.md
    reports/20260807_main_prior_conclusions/analysis_summary.json
    docs/hpo_bayesmark.md
    docs/GIT_BO_ADAPTER.md
    bbo/algorithms/benchmark_protocol.py
    bbo/core/conversion.py
    bbo/algorithms/model_based/gp_ei.py
    bbo/algorithms/model_based/botorch_turbo.py
    bbo/algorithms/model_based/git_bo.py
    workflow/40_nanobot_no_tool_multifamily_20260724/run_harness_no_tool_multifamily.py
    workflow/40_nanobot_no_tool_multifamily_20260724/run_baseline_multifamily.py
    workflow/40_nanobot_no_tool_multifamily_20260724/run_opro_multifamily.py
    workflow/50_bbob24_d10_i123_init20_opt100_20260802/setting.json
    workflow/52_hpo25_seed01_opro_remote_20260804/setting.json
    workflow/54_aligned_numerical_baselines_remote_20260805/setting.json
    workflow/55_git_bo_remote_20260806/setting.json
    workflow/56_hpo25_seed01_three_harness_remote_20260806/setting.json
