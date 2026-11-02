# US-101：连续特征掩码预训练与聚类

这是独立的真实数据探索基线，以整条有效车辆观测轨迹为单位，
不使用 Porto 词表、速度三元组或预设行为标签。
流程为：**连续数据 → 掩码预训练 → 整轨迹表示 → HDBSCAN**。

## 运行

从 SP-DTC 根目录执行，服务器复用已有 CUDA PyTorch 环境：

```bash
python -m pip install -r requirements.txt

# 已有连续 H5 时跳过此步；默认读取 data/NGSIM/us101.csv。
python -m src.data.generation.build_ngsim

# CPU 前向检查，不创建优化器或训练模型。
python -m src.experiment.run_ngsim --check-only

# 正式预训练，默认要求 CUDA。
python -m src.experiment.run_ngsim \
  --config configs/baseline/spdtc_us101_single_masked_pre30.yaml

# 对已导出的表示聚类，不重新训练；替换为自己的运行目录。
python -m src.experiment.cluster_ngsim \
  --run runs/ngsim/spdtc_us101_single_masked_pre30/20261101T094030_429486Z_seed42
```

服务器训练只需代码、配置、依赖和 `trajectories.h5`，无需原始 CSV。
H5 不提交 Git，需单独传输；预训练和聚类入口均可用 `--data /absolute/path/trajectories.h5`。
预训练写入 `runs/ngsim/<配置名>/<UTC时间>_seed42/`；前向检查和聚类写入 `artifacts/`。
可用 `--output-dir` 指定新目录，已有目录拒绝覆盖。当前不提供断点续训。

## 数据与模型

- 原始轨迹的整理、质量标记和 H5 字段见 [预处理说明](../../src/data/generation/NGSIM.md)。
- 输入为 `[relative_x_m, relative_y_m, speed_m_s]`，默认采用原始记录速度。
  `speed_source: smoothed` 可显式选用预处理时另存的平滑速度。
- 每条轨迹的位置和时间相对其起点；保留全部采样点、真实时长和速度尺度。
  三个通道仅按训练集有效点的均值、标准差缩放；时间以相对秒数加入正弦/余弦编码。
- 单分支 Transformer：256 维、2 层、2 个注意力头、1024 维前馈层。
  批次补齐由真实长度生成 PAD 掩码，合法零值不会被当作 PAD。
- 每条轨迹约 20% 的点按约 2 秒连续段遮盖，三个物理通道同时遮盖，首尾点保留。
  时间已知，被遮盖的物理数值在投影前清除。训练掩码逐轮变化，验证掩码固定。
- 只在遮盖点计算 Smooth L1，先逐轨迹平均，再对样本平均。
  AdamW 学习率 0.00015，最多 30 轮；验证标准化 MSE 连续 8 轮不改善则早停。
- 同一验证掩码下保留线性插值对照，报告标准化 MSE 和物理单位 RMSE。
  这是重建评价，不代表行为聚类准确率。

## 划分

第一轮只使用汽车（`vehicle_class=2`），车型不输入模型。
训练批次为 `1118846978900`、`1118847842000`，验证批次为 `1118848757000`。
批次由 `Global_Time - 100 * Frame_ID` 推导，并非官方批次名称。
隔离时钟重叠区间及前后 5 秒涉及的整条轨迹，不在边界处截短轨迹。
位置—速度失配标记默认保留，并报告其重建误差。

当前 H5 的训练/验证轨迹数为 3,497 / 1,697；725 条汽车轨迹因时钟重叠被隔离，
182 条非汽车轨迹不在范围内。验证集用于检查点选择和早停，**不是独立测试集**。

## 聚类

最优检查点对全部选入的完整、未遮盖轨迹导出原始 256 维均值池化表示。
HDBSCAN 同时提供两个独立结果：

1. **编码器表示**：不额外标准化或 L2 归一化。
2. **统计特征对照**：时长、纵向距离、均速、速度标准差、起止均速差、低速占比、
   横向净位移和横向范围，共 8 维。起止均速取首尾各 1 秒，低速阈值为 0.5 m/s。

统计特征先标准化，两组输入再分别用 PCA 保留 99% 方差、不白化；
所有变换只在原训练集拟合。以上设置沿用已有探索实验。
两组均使用欧氏距离、`eom` 提取，默认 `min_cluster_size=50`、`min_samples=10`，
不预设 K。可通过 `--min-cluster-size`、`--min-samples` 显式改变参数。

输出 `embedding_assignments.csv`、`statistics_assignments.csv` 和各自的
`*_cluster_summary.csv`，保留逐轨迹 ID、簇号、成员强度及统计量的中位数和四分位数。
`feature_spaces.npz` 保存变换参数和聚类输入，`summary.json` 保存参数、簇规模和噪声占比，
`manifest.json` 保存来源文件指纹和软件版本。

全部导出的训练与验证轨迹共同参与本轮探索；簇号没有预先赋予行为含义。
`-1` 表示未入簇，成员强度不是行为正确概率，簇数和噪声占比也不是准确率。
当前入口仅保留固定参数聚类，不自动运行参数扫描、抽样稳定性或行为解释诊断。

## 文件管理

训练保留配置、划分、标准化参数、数据指纹、训练历史、最佳检查点、验证误差和表示文件。
Git 只收录小型运行摘要和主日志；H5、逐轨迹 CSV、权重、表示和 `artifacts/` 保留在本地。
历史分析报告仍在 `artifacts/ngsim_clustering/` 和 `artifacts/ngsim_processes/`，不由训练入口生成。
