# SP-DTC：速度模式感知深度轨迹聚类

保留 ConDTC 基线和速率三元组两组，并新增独立路径/节奏双分支实验。输入仍为位置 token、分钟时间 token 和序列位置编码，现有数据不变。

| 组别 | 预训练 | 联合优化 |
|---|---|---|
| `baseline` | MSTM | MSTM + 双视图 DEC |
| `tempo_only` | MSTM + 速率三元组 | MSTM + 双视图 DEC + 速率三元组 |
| `dual_branch` | 路径分支预测位置；节奏分支预测时间并学习速率三元组 | 两分支加权距离上的双视图 DEC + 原预训练任务 |

双分支有独立编码参数，速率梯度只进入节奏分支，联合 DEC 更新两支。设计、参数含义与诊断输出见 [双分支说明](configs/architecture/README.md)。此组是待验证的候选结构，不代表已经改善聚类质量。

## 代码结构

```text
SP-DTC/
├── configs/                # baseline / ablation / architecture
├── data/                   # 已生成的数据
├── runs/                   # baseline / speed_only / dual_branch
└── src/
    ├── data/
    │   ├── data_process.py # token、掩码与样本索引
    │   ├── data_loader.py  # 训练/验证划分和批处理
    │   ├── augmentation.py # DEC 双视图增强
    │   ├── mstm.py         # 时空掩码任务
    │   ├── speed_pattern.py # 速率三元组的正负视图
    │   └── generation/    # 数据生成及 resources/ 基础路径资料
    ├── models/            # 编码器、MSTM、DEC 与速率三元组损失
    ├── training/          # 预训练、联合优化、共用工具与最终指标
    └── experiment/        # 配置与统一运行入口
```

## 速率三元组

正样本轻微删除内部点并整体平移时间 1–2 分钟，保留起终点。负样本保留位置序列、点数、首尾时刻和总用时，用平滑权重重新分配局部时间间隔。分钟量化后的累计时间位移 RMS 达到阈值才参与损失。

扰动中心按有效采样点序号归一化到 `[0, 1]`，配置字段为 `warp_axis: sample_index`。这一序号只表示采样顺序，不表示等物理距离；构造视图只读取 token、掩码与样本索引。

```text
L_speed = mean_valid max(||z̄ - z̄_positive||² - ||z̄ - z̄_negative||² + margin, 0)
L_pretrain = L_location + 0.1 L_time + λ_speed L_speed
L_joint = L_location + 0.1 L_time + 0.5 L_cross_KL + λ_speed L_speed
L_cross_KL = 0.5 [KL(P2 || Q1) + KL(P1 || Q2)]
```

单分支中，`z̄` 是 L2 归一化的池化表示，DEC 使用未归一化的 `z`。基线的 λ 为 0，速率组为 1，margin 为 0.2。双分支中三元组作用于节奏分支的池化表示，DEC 使用分别归一化、加权且缩放后的两支拼接向量。辅助编码关闭 dropout，同时保留梯度。基线继承的实例/簇投影头保持参数和初始化顺序以兼容旧检查点，当前目标函数不使用其输出。

## 数据与服务器运行

| preset | 数据集 | 样本数 | 簇数 | 默认预训练 / 联合轮数 |
|---|---|---:|---:|---:|
| small | portoLocalTempo4Small_v1 | 2,400 | 12 | 15 / 6 |
| scale | portoLocalTempo4Scale_v1 | 12,000 | 12 | 30 / 6 |
| full | portoLocalTempo4Full_v1 | 36,000 | 36 | 30 / 6 |

类别定义仍为基础路径 × 四种速度模式。现有数据保持原样；预训练按样本划分 80/10/10。当前数据为构造数据，对真实行为差异的适用性需要另外验证。

```bash
# 完整运行两组，各自预训练并联合优化。
python -m src.experiment.run --preset full

# 单独运行一组。
python -m src.experiment.run --preset full --groups baseline
python -m src.experiment.run --preset full --groups tempo_only

# 新双分支独立运行；默认入口仍只运行原有两组。
python -m src.experiment.run --preset full --groups dual_branch
```

默认要求 CUDA。相对 `data_root` 从 SP-DTC 根目录解析，服务器可通过 `--data-root` 指定数据位置。配置说明见 [configs/README.md](configs/README.md)。

入口直接加载数据，数据加载器负责 token、长度与词表的基本检查。运行记录保存实际使用的代码和数据哈希，用于识别版本及补跑同一套件。

结果按 `runs/<baseline、speed_only 或 dual_branch>/<实验配置名>/<时间>_seed<种子>/` 存放，同一批次的原有两组使用相同时间标识，双分支单独建立运行。`speed_only` 对应内部组名 `tempo_only`。输出包括预训练日志、检查点、联合训练日志和最终 `metrics.json`（UACC/NMI/RI/ARI）。训练完成后，在原始轨迹上评估一次最终分配；标签只用于评估。

单分支训练配置为 schema v3，双分支为 schema v4；两者都明确记录按采样点序号扰动。结果目录保留 full/pre30 下的两组旧版结果，其中速率组按几何沿程进度扰动，与当前增强规则不同；它们保留原始配置，不作为当前版本的新实验结果。三组都从头预训练，无需外部教师或预训练输入。

Git 只收录配置、最终指标、主训练日志和少量来源记录；权重、逐轮快照及诊断明细保留在本地。目录和提交范围见 [runs/README.md](runs/README.md)。

复制到服务器时，携带 `src/`、`configs/`、`data/`、`requirements.txt` 和本说明即可。

## 数据生成

```bash
# 按需重新生成；生成工具拒绝覆盖已有目录。
python -m src.data.generation.build_small --output-dir data/regenerated/portoLocalTempo4Small_v1
python -m src.data.generation.build_fullscale \
  --pilot-dir data/regenerated/portoLocalTempo4Small_v1 \
  --output-root data/regenerated
```

数据生成所需资料位于 `src/data/generation/resources/`。SP-DTC 使用独立 Git，现有 `con-dtc` 不参与本项目运行。

## NGSIM 连续轨迹实验

US-101 的真实数据单独按整条车辆观测轨迹整理，保留相对时间、相对位置和速度；无需位置词表。

```bash
python -m src.data.generation.build_ngsim
```

默认读取 `data/NGSIM/us101.csv`，输出到 `data/NGSIM/us101_processed/`，包含 H5、轨迹索引和质量报告。
数据结构和处理规则见 [NGSIM 预处理说明](src/data/generation/NGSIM.md)。该连续数值 H5 使用独立入口，不能直接传给 Porto 的 token 训练入口。

```bash
# 本地只检查数据和前向计算，不训练。
python -m src.experiment.run_ngsim --check-only

# 在服务器上预训练，并导出完整轨迹表示；默认要求 CUDA。
python -m src.experiment.run_ngsim --config configs/baseline/spdtc_us101_single_masked_pre30.yaml
```

预训练完成后，使用独立入口对完整轨迹表示和统计特征分别进行 HDBSCAN：

```bash
python -m src.experiment.cluster_ngsim \
  --run runs/ngsim/spdtc_us101_single_masked_pre30/20261101T094030_429486Z_seed42
```

将 `--run` 替换为自己的预训练运行目录；默认不指定簇数，不重新训练。
数据 H5 需单独传输，权重、表示和分析结果保留在本地，不提交 Git。
划分、掩码任务及完整命令见 [US-101 实验说明](configs/baseline/ngsim.md)。
