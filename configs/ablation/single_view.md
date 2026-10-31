# 双分支的单视图聚类消融

目的：检查聚类阶段的删点视图、时间平移视图及其跨视图目标是否有帮助。

对照为 `configs/architecture/spdtc_porto_full_dual_branch_pre30.yaml`。本配置只新增 `cluster.view_mode: single`，路径/节奏两个独立编码器、归一化融合、距离尺度、学习率与损失权重均保持一致。

| 环节 | 原双视图配置 | 本单视图配置 |
|---|---|---|
| 预训练 | MSTM + 节奏三元组，30 轮 | 相同 |
| 中心初始化 | 原始轨迹表示上的 KMeans | 相同 |
| 每轮目标 | 两个增强视图的全量 Q1、Q2，分别锐化为 P1、P2 | 原始轨迹的全量 Q，锐化为 P |
| 聚类项 | 0.5 × [KL(P2∥Q1) + KL(P1∥Q2)] | KL(P∥Q) |
| 联合目标 | MSTM + 0.5 × 聚类项 + 1.0 × 节奏三元组 | 相同权重 |
| 联合轮数 | 6 | 6 |

单视图下，P 由每轮开始时 eval 模式的模型计算，簇频率来自全部样本；本轮 batch 按样本 ID 读取固定且不反传梯度的 P。训练时编码器仍为 train 模式，保留原有 dropout。训练 loader 保留原始输入供 DEC 使用，另附 MSTM 遮盖字段，不构造聚类的 view1/view2。

本消融移除的是聚类双视图增强。MSTM 遮盖、节奏三元组中的正样本删点与负样本局部时间变形仍保留；双编码器也仍保留。没有把增强幅度设为零来模拟单视图。

在 SP-DTC 仓库根目录运行（服务器需 CUDA）：

```bash
python -m src.experiment.run \
  --config configs/ablation/spdtc_porto_full_dual_branch_single_view_pre30.yaml
```

完整数据为 `portoLocalTempo4Full_v1`，默认 seed 42，重新执行 30 轮预训练和 6 轮联合优化。结果写入：

```text
runs/dual_branch/spdtc_porto_full_dual_branch_single_view_pre30/<时间>_seed42/
```

新目录不覆盖已有双视图结果。训练日志的 `view_mode` 为 `single`，轮初分配统计写入 `epoch_start_assignments_original`。保留第 0/6 轮双分支表示快照以及最终 `metrics.json`。

比较时同时查看最终 UACC/NMI/ARI、路径纯度、同路径内节奏区分，以及第 0 轮与第 6 轮的变化。先检查两组第 0 轮快照是否一致或接近：同一预训练设置和种子不等于跨环境逐位一致。本入口会重新预训练，不自动复用旧检查点。一次 seed 42 实验只能提供初步证据；本消融同时移除增强输入和跨视图约束，不能仅凭结果进一步归因于某一种增强。
