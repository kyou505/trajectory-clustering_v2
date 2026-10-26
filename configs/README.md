# 实验配置

命名使用 `方法_数据集与规模_实验变体_预训练轮数.yaml`，沿用 con-dtc 的实验配置形式。当前只保留两份配置：

| 配置 | 用途 | 内部组名 |
|---|---|---|
| `baseline/condtc_porto_full_baseline_pre30.yaml` | ConDTC 基线 | `baseline` |
| `ablation/spdtc_porto_full_speed_only_pre30.yaml` | 基线 + 速率三元组 | `tempo_only` |

两组共享编码器、MSTM 和双视图 DEC，只改变 `speed_pattern.enabled`。速率组的三元组同时用于预训练和联合优化，权重为 1、margin 为 0.2。

## 配置与规模

当前 schema 为 v3，`speed_pattern.warp_axis: sample_index` 表示按采样点序号定位局部时间扰动。旧配置采用不同的扰动规则，不自动转换为当前方法；请使用这两份配置开始新实验。

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
```

`--config` 使用指定文件的数据集与轮数；`--preset small/scale` 覆盖数据集、簇数和默认预训练轮数。可用 `--pretrain-epochs` 和 `--seed` 显式覆盖对应参数。

训练从当前数据开始预训练，无需外部权重或历史结果摘要。结果写入 `runs/<baseline 或 speed_only>/<实验配置名>/<时间>_seed<种子>/`，与本目录的配置命名对应。同一批次的两组使用相同时间标识；`speed_only` 对应内部组名 `tempo_only`。

`--suite-dir` 指向该批次任意一组的时间目录，可在另一方法目录下补跑缺失组；数据、代码和公共配置必须一致。旧版结果不能用当前代码接续。

数据生成资料位于 `src/data/generation/resources/`。结果目录只保留 baseline 和速率三元组，说明见 [runs/README.md](../runs/README.md)。
