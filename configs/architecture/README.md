# 路径与节奏双分支实验

本组改变表示结构及聚类距离。两条分支同时预训练，不依赖外部 baseline 教师，也不延后开启速率任务。

本页描述缺省 `task_routing: split_masked`。第一支完整 MSTM、第二支速度三元组的独立对照见 [任务分工消融](../ablation/mstm_speed.md)。

| 部分 | 输入 | 参数与任务 |
|---|---|---|
| path | 位置 token + 正弦序列编码 | 独立嵌入及 Transformer；预测被遮盖的位置 |
| rhythm | 位置 token + 时间 token + 正弦序列编码 | 独立嵌入及 Transformer；预测被遮盖的时间，学习速率三元组 |
| clustering | 两分支分别有效点平均池化 | 分别归一化，加权拼接，在同一坐标系下执行 KMeans 与 DEC |

默认每支维度 256、2 层、2 个注意力头，拼接维度 512。两支的位置嵌入也不共享。MSTM 的两个预测变换独立，预测头只与各自分支内的嵌入共享权重。两个分支在掩码任务中均使用遮盖后的输入，没有从原始位置/时间字段绕过遮盖。

速率正负视图沿用当前 `sample_index` 构造方式。其损失只调用节奏分支，在关闭 dropout、保留梯度的模式下前向。

## 目标和梯度

```text
L_pretrain = L_location(path) + 0.1 L_time(rhythm) + L_speed(rhythm)
L_joint = L_pretrain + 0.5 L_cross_KL(joint)
```

| 损失 | path 参数 | rhythm 参数 | 聚类中心 |
|---|---|---|---|
| 位置预测 | 更新 | 不更新 | 不更新 |
| 时间预测 | 不更新 | 更新 | 不更新 |
| 速率三元组 | 不更新 | 更新 | 不更新 |
| 联合跨视图 KL | 更新 | 更新 | 更新 |

结构上阻止的是速率损失直接改动路径分支。联合 KL 仍耦合两支，位置预测也不保证轨迹级路径分离，所以该设计不能保证路径纯度为 100%。没有加入正交约束、额外空间三元组或坐标回归目标。

## 联合距离

令 `u_s = normalize(z_s)`，`u_r = normalize(z_r)`，实际聚类向量为：

```text
z_joint = [sqrt(c * lambda) * u_s, sqrt(c * (1-lambda)) * u_r]
```

因此样本间平方距离严格等于 `c * [lambda * D_path + (1-lambda) * D_rhythm]`。
每个簇中心是同一 512 维空间中的一个向量，其前后两块对应路径和节奏中心；KMeans 初始化后作为 DEC 参数更新，不重新归一化中心。簇数仍为 36，不先指定 9 个路径簇或强制每条路径有 4 个节奏簇。

配置 `path_weight=0.5` 设定两支权重；`distance_scale=100` 缩放平方距离，相当于将拼接向量放大 10 倍。共同缩放不改变固定表示上的 KMeans 最优分组，但会影响 Student-t 软分配、梯度和后续训练。这个起始尺度用于避免单位向量距离下的分配过于平坦，尚未证明是最优设置。不要把调整尺度后的结果归因于纯粹的结构变化。

## 运行

在 SP-DTC 根目录、已有依赖的 CUDA 环境执行：

```bash
python -m src.experiment.run --preset full --groups dual_branch
# 等价的完整配置入口：
python -m src.experiment.run --config configs/architecture/spdtc_porto_full_dual_branch_pre30.yaml
```

使用 `--preset small --groups dual_branch` 可切到已有 2,400 条小数据集，默认 15 轮预训练、6 轮联合训练。full 默认 30 + 6 轮，seed 42；支持原入口的 `--seed`、`--data-root` 和 `--pretrain-epochs`。

不指定 groups 时仍只运行原有两组。新组需单独调用入口；其架构配置不能和原有两组共用套件。三组均从头预训练，按固定验证 MSTM 损失选择预训练 checkpoint，按最后一轮联合训练报告结果，标签不参与训练与 checkpoint 选择。

## 结果与比较边界

结果写到 `runs/dual_branch/spdtc_porto_full_dual_branch_pre30/<时间>_seed42/`。
除原有日志和最终指标外，默认保存 `cluster/representations/epoch_000.pt` 与 `epoch_006.pt`：

- 稳定样本 ID、评估标签、未归一化 path/rhythm 池化向量、q、加权空间中的簇中心及配置。
- 对应 JSON 记录分支维度/平均范数、分配归一化熵、最大概率均值和实际使用的簇数。
- 可以比较同一路径的节奏分离、跨路径混淆以及这些现象从初始化到联合训练结束的变化；路径/节奏标签需与原数据元信息对齐，不在训练里推断路径编号。

日志中的 `epoch_start_assignments_view1` 是每轮更新前的全量视图 1 分配统计。保存表示开关可设为 false；不影响目标、checkpoint 选择和最终指标。

默认双分支比原单分支有更多参数，MSTM/聚类前向也更贵。初步对照应重新运行当前 schema v3 的 tempo_only，使用同一数据与种子；已有旧版速率结果使用不同扰动规则，不能作为严格对照。若效果提升，后续还需控制参数量、归一化及距离尺度，才能区分各项设计的贡献。

本地功能验证：

```bash
python -m pytest -q tests/test_dual_branch.py
```

测试覆盖分支梯度隔离、时间不变性、PAD/掩码处理、联合距离等价性、KMeans/DEC 坐标一致性、检查点加载、原模型初始化兼容性，以及小型数据上一轮预训练和一轮联合训练。它们验证实现，不能代替完整数据的有效性实验。
