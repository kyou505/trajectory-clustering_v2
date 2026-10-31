# 实验配置

命名使用 `方法_数据集与规模_实验变体_预训练轮数.yaml`，沿用 con-dtc 的实验配置形式：

| 配置 | 用途 | 内部组名 |
|---|---|---|
| `baseline/condtc_porto_full_baseline_pre30.yaml` | ConDTC 基线 | `baseline` |
| `ablation/spdtc_porto_full_speed_only_pre30.yaml` | 基线 + 速率三元组 | `tempo_only` |
| `architecture/spdtc_porto_full_dual_branch_pre30.yaml` | 独立路径/节奏分支 + 联合距离 | `dual_branch` |
| `ablation/spdtc_porto_full_dual_branch_single_view_pre30.yaml` | 双分支移除聚类双视图增强，改用原始单视图 DEC | `dual_branch` |

原有两组共享编码器、MSTM 和双视图 DEC，只改变 `speed_pattern.enabled`。速率组的三元组同时用于预训练和联合优化，权重为 1、margin 为 0.2。双分支沿用这些任务权重、数据与预算，改变编码结构、任务路由和聚类距离；详见 [双分支说明](architecture/README.md)。

单视图消融只增加 `cluster.view_mode: single`，其余设置与双分支默认配置一致。缺省或 `dual` 仍使用原双视图训练。运行方法和对照范围见 [单视图消融说明](ablation/single_view.md)。

## 配置与规模

单分支 schema 为 v3，双分支为 v4，后者显式记录分支权重与距离尺度。`speed_pattern.warp_axis: sample_index` 表示按采样点序号定位局部时间扰动。旧配置采用不同的扰动规则，不自动转换为当前方法；请使用当前配置开始新实验。

| preset | 样本数 | 簇数 | 默认预训练 / 联合轮数 |
|---|---:|---:|---:|
| small | 2,400 | 12 | 15 / 6 |
| scale | 12,000 | 12 | 30 / 6 |
| full | 36,000 | 36 | 30 / 6 |

`porto_full` 为 9 条基础路径 × 4 种速度模式的构造数据。`pre30` 表示预训练预算；内部 `tempo_only` 沿用已有命名。

```bash
# 两组完整流程。
python -m src.experiment.run --preset full

# 只运行速率组，或显式使用配置文件。
python -m src.experiment.run --preset full --groups tempo_only
python -m src.experiment.run --config configs/ablation/spdtc_porto_full_speed_only_pre30.yaml

# 双分支单独运行，避免把结构改动当成仅开关速率损失的消融。
python -m src.experiment.run --config configs/architecture/spdtc_porto_full_dual_branch_pre30.yaml
```

`--config` 使用指定文件的数据集与轮数；`--preset small/scale` 覆盖数据集、簇数和默认预训练轮数。可用 `--pretrain-epochs` 和 `--seed` 显式覆盖对应参数。

训练从当前数据开始预训练，无需外部权重或历史结果摘要。结果写入 `runs/<baseline、speed_only 或 dual_branch>/<实验配置名>/<时间>_seed<种子>/`，与本目录的配置命名对应。同一批次的原有两组使用相同时间标识；`speed_only` 对应内部组名 `tempo_only`。双分支必须单独调用入口，不能与单分支配置共享 `--suite-dir`。

`--suite-dir` 指向该批次任意一组的时间目录，可在另一方法目录下补跑缺失组；数据、代码和公共配置必须一致。旧版结果不能用当前代码接续。

数据生成资料位于 `src/data/generation/resources/`。历史结果和新运行目录说明见 [runs/README.md](../runs/README.md)。
