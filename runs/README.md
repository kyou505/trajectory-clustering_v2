# 实验结果

保留 ConDTC 基线与速率三元组历史结果，新增双分支运行目录，按方法、实验配置和运行时间命名。

```text
runs/
├── baseline/
│   └── condtc_porto_full_baseline_pre30/<时间>_seed42/
├── speed_only/
│   └── spdtc_porto_full_speed_only_pre30/<时间>_seed42/
└── dual_branch/
    └── spdtc_porto_full_dual_branch_pre30/<时间>_seed42/
```

`speed_only` 对应代码内部的 `tempo_only`。同一批次两组使用相同时间标识，分别完成预训练与联合聚类。

## 已有结果

2026-10-26 整理后只保留 full/pre30 设置下的 baseline 和速率三元组，共两次完整运行：

| 设置 | 样本数 / 类别数 | 预训练 / 联合轮数 | 批次 |
|---|---|---|---|
| Porto full | 36,000 / 36 | 30 / 6 | `20261025_155146_890675_seed42` |

这些记录来自旧版 schema v1，速率组使用旧的几何沿程扰动。当前 schema v3 使用 `warp_axis: sample_index`，尚无正式运行结果；不能把旧指标当成当前代码的实验结果。配置、指标和日志内容原样保留，历史文件中的旧路径没有改写。

移出项目的内容包括 small 实验记录、motion_only、combined、路径读出层、路径三元组及其参考输入。两个旧路径实验中的 tempo_only 是额外的联合阶段复跑，四项最终指标均与上述 full 速率组一致；此处只保留包含完整预训练过程的那次运行。

## Git 提交范围

参考 con-dtc，保留 `config.yaml`、组级 `metrics.json` 和主日志 `train.log`；另保留 `completed.json` / `failed.json`、`environment.json`、`suite_identity.json`，用于确认完成状态与运行来源。

权重、优化器状态、逐轮历史、诊断 CSV、表示快照、analysis 下的重复指标和额外控制台日志默认忽略。保留的两次运行中，这些文件仍在本地，Git 忽略规则不会删除它们。

## 新运行与补跑

新结果沿用上述结构，full 默认配置的目录名与 configs 中的 YAML 文件名一致。单独运行一组后，可将其时间目录传给 `--suite-dir`，补跑另一组；入口会核对代码、数据和公共配置。每组目录各自保存 `suite_identity.json`，已完成的同版本运行会跳过。

旧版运行不支持用当前代码补跑，应开始新的批次。

双分支使用 schema v4，从头预训练并独立运行，不与原两组共用套件。初始化及最终的表示保存在 `cluster/representations/`，默认同其他大型训练产物一样不收录到 Git。该目录出现不代表已完成有效性实验，完成状态以实际运行记录为准。
