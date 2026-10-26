# SP-DTC：速度模式感知深度轨迹聚类

当前只保留 ConDTC 基线和速率三元组两组。输入为位置 token、分钟时间 token 和序列位置编码，两组共享编码器与聚类结构。

| 组别 | 预训练 | 联合优化 |
|---|---|---|
| `baseline` | MSTM | MSTM + 双视图 DEC |
| `tempo_only` | MSTM + 速率三元组 | MSTM + 双视图 DEC + 速率三元组 |

## 代码结构

```text
SP-DTC/
├── configs/                # baseline / ablation，各一份配置
├── data/                   # 已生成的数据
├── runs/                   # baseline / speed_only，两组实验结果
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

`z̄` 是 L2 归一化的池化表示，DEC 使用未归一化的 `z`。基线的 λ 为 0，速率组为 1，margin 为 0.2。辅助编码关闭 dropout，同时保留梯度。基线继承的实例/簇投影头保持参数和初始化顺序以兼容旧检查点，当前目标函数不使用其输出。

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
```

默认要求 CUDA。相对 `data_root` 从 SP-DTC 根目录解析，服务器可通过 `--data-root` 指定数据位置。配置说明见 [configs/README.md](configs/README.md)。

入口直接加载数据，数据加载器负责 token、长度与词表的基本检查。运行记录保存实际使用的代码和数据哈希，用于识别版本及补跑同一套件。

结果按 `runs/<baseline 或 speed_only>/<实验配置名>/<时间>_seed<种子>/` 存放，同一批次的两组使用相同时间标识。`speed_only` 对应内部组名 `tempo_only`。输出包括预训练日志、检查点、联合训练日志和最终 `metrics.json`（UACC/NMI/RI/ARI）。训练完成后，在原始轨迹上评估一次最终分配；标签只用于评估。

当前训练配置为 schema v3，明确记录按采样点序号扰动。结果目录只保留 full/pre30 下的两组旧版结果，其中速率组按几何沿程进度扰动，与当前增强规则不同；它们保留原始配置，不作为当前版本的新实验结果。当前两组从头预训练，无需外部预训练输入。

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
