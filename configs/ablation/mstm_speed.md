# 完整 MSTM 分支与速度三元组分支

对照对象为刚完成的单视图双分支配置 `spdtc_porto_full_dual_branch_single_view_pre30.yaml`。新配置只增加 `dual_branch.task_routing: mstm_speed`，使用同一份完整数据、seed 42、30 轮预训练和 6 轮单视图 DEC 联合训练。

## 分工

| 部分 | 原分工（split_masked） | 本组（mstm_speed） |
|---|---|---|
| 第一支输入 | 位置 token + 序列编码 | 位置 token + 时间 token + 序列编码 |
| 第一支预训练 | 掩码位置预测 | 完整 ConDTC MSTM：掩码位置预测 + 掩码时间预测 |
| 第二支输入 | 位置 token + 时间 token + 序列编码 | 相同 |
| 第二支预训练 | 掩码时间预测 + 速度三元组 | 仅速度三元组 |
| 联合阶段 | 原辅助任务 + 单视图 DEC | 新辅助任务分工 + 单视图 DEC |

两支的位置/时间嵌入和 Transformer 参数均独立。第一支使用与仓库中 ConDTC 基线相同的 `TrajectoryEncoder` 结构，以及共享的 `Linear + GELU` 预测变换；位置头和时间头分别与第一支的位置嵌入、时间嵌入绑定权重。MSTM 只读取遮盖字段，不执行第二支。第二支由原轨迹及其速度正负样本学习，不增加预测头或掩码任务。

两支都从头训练，不复制已有模型或使用外部教师。这里“第二支仅速度三元组”指辅助任务分工；联合阶段的 DEC 仍更新第二支。

```text
L_pretrain = L_location(branch1) + 0.1 L_time(branch1) + L_speed(branch2)
L_joint = L_pretrain + 0.5 KL(P || Q)
```

| 损失 | 第一支 MSTM 编码器 | 第二支速度编码器 | 簇中心 |
|---|---|---|---|
| 位置预测 | 更新 | 不更新 | 不更新 |
| 时间预测 | 更新 | 不更新 | 不更新 |
| 速度三元组 | 不更新 | 更新 | 不更新 |
| 单视图 DEC | 更新 | 更新 | 更新 |

表格表示各项损失的直接梯度来源，并不声称联合训练中两支完全解耦。预训练 checkpoint 仍按验证集 MSTM 损失选取，不改为按三元组或真实标签选择；因此也应查看被选轮次的速度任务验证指标。

## 融合与对照范围

两支输出仍各为 256 维，有效点平均池化、L2 归一化后等权拼接成 512 维向量；`distance_scale=100`，KMeans 初始化 36 个簇，其余损失权重与学习率不变。聚类采用原始单视图，保留 MSTM 遮盖及速度三元组的正负样本构造。

配置沿用 `path_weight=0.5` 作为第一支融合权重。快照仍使用 `path`、`rhythm` 字段以兼容分析，但额外记录 `branch_roles: {path: mstm, rhythm: speed}`。本组的 `path` 是完整时空表示，不能解读为时间不变的纯路径表示。

这是一组架构与任务分工的联合对照：第一支新增时间嵌入，时间预测迁移至第一支，两个预测变换改为 ConDTC 的共享变换。参数量相对原双分支增加 `time_vocab_size * d_model - (d_model * d_model + d_model)`，默认 1444/256 时增加 303,872 个参数。因此结果不能单独归因于移动时间损失，也不能先验保证路径纯度更高。

## 服务器运行

在 SP-DTC 仓库根目录执行：

```bash
python -m src.experiment.run \
  --config configs/ablation/spdtc_porto_full_dual_branch_mstm_speed_single_view_pre30.yaml
```

结果写入独立目录：

```text
runs/dual_branch/spdtc_porto_full_dual_branch_mstm_speed_single_view_pre30/<时间>_seed42/
```

原分工与新分工的预训练权重不兼容，加载时会校验 `task_routing` 并拒绝混用。该入口完整重新预训练，不复用刚完成实验的权重。

对比时使用同一服务器软件环境，检查最终 UACC/NMI/ARI、路径纯度、每条路径内的节奏 ARI，以及第 0/6 轮快照。第 0 轮起点变化本身也是预训练分工改变的结果；本实验不是仅改变聚类阶段的消融。
