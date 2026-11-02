# US-101 连续轨迹数据

此入口整理真实观测轨迹，不生成类别、词表或时间扰动样本。默认输入为
`data/NGSIM/us101.csv`，默认输出为 `data/NGSIM/us101_processed/`。

```bash
python -m src.data.generation.build_ngsim
# 修改输出目录可保留不同处理版本；已有目录不会被覆盖。
python -m src.data.generation.build_ngsim \
  --source data/NGSIM/us101.csv \
  --output-dir data/NGSIM/us101_processed_v2
```

使用 `requirements.txt` 中的 numpy、pandas、scipy、h5py；无需 GPU。

## 处理边界

1. 读取带千位逗号的数值，按所有原始字段去除完全重复记录。
2. 用 `Global_Time - 100 * Frame_ID` 推导录像批次，与车辆 ID 组合成轨迹键。
   此字段是本导出文件的时钟分组，不是官方批次名称。批次时钟存在重叠，未自动修复。
3. 按批次、车辆、时间排序。主分析单位为整条观测轨迹，不切 20 秒窗口。
4. 同车同刻冲突、时间或帧号不连续、声明帧数不符、非有限输入、负速度、车型不一致等
   结构问题会隔离整条轨迹，并写出原因；不任意删点后重新拼接。
5. 使用 `0.3048` 将英尺换算为米。每条轨迹分别减去首个位置和时间，保留真实秒数。
6. 保留全部车型；`vehicle_class=2` 可用于第一轮汽车分析。车型不等于行为类别。

**主输入采用原始记录速度，仅转换单位。** 另存 11 点、二阶 Savitzky–Golay 平滑速度：
10 Hz 下窗口跨度为 1 秒，在整条连续轨迹内计算；不足 11 点时复制原速度并标记。
平滑产生的负速度截为零，并记录数量。改变 `--smoothing-points` 可以生成平滑对照。

位置差分速度与记录速度之差超过 5 m/s、纵向倒退和原始加速度边界值等只用于标记，
不是已经验证的剔除规则。可用 `--mismatch-threshold` 改变位置—速度差异的诊断阈值。
当前主输入没有加入原始加速度，没有坐标平滑、插值、下采样或标准化。

## 文件

| 文件 | 内容 |
|---|---|
| `trajectories.h5` | 连续数值输入、变长索引、来源信息与诊断字段 |
| `trajectory_index.csv` | 一条轨迹一行：ID、长度、时长、批次、车型、质量标记 |
| `rejected_trajectories.csv` | 结构不合格轨迹及原因，零条时仍保存表头 |
| `processing_config.json` | 输入及生成脚本指纹、单位转换、平滑与阈值等设置 |
| `quality_report.json` | 去重、保留、隔离、分车型规模、时钟重叠及写回检查 |
| `checksums.json` | 本次生成的各文件 SHA-256 |

## H5 结构

所有轨迹的点按顺序连续保存，避免为最长轨迹提前补齐整个数据集。
设总点数为 N、轨迹数为 M：

```text
features                       float32 [N, 3]
  列：relative_x_m, relative_y_m, speed_m_s（原始记录速度）
time                           float64 [N]，每条轨迹从 0 开始的秒数
speed_smoothed                 float32 [N]，可选的平滑速度对照
offsets                        int64 [M + 1]
tracks/
  trajectory_id                string [M]，us101:<推导批次>:<车辆编号>
  length                       int64 [M]
  recording_id, vehicle_id, vehicle_class, duration_s, ...
  origin_x_m, origin_y_m, start_time_ms, end_time_ms, ...
  position_speed_mismatch_pairs, smoothing_negative_clipped_points, ...
observations/
  global_time_ms               int64 [N]，原始时间戳
  frame_id                    int32 [N]
  lane_id                     int16 [N]
  raw_acceleration_m_s2        float32 [N]，仅供核查
```

第 i 条轨迹使用 `offsets[i]:offsets[i+1]` 截取：

```python
import h5py

with h5py.File('data/NGSIM/us101_processed/trajectories.h5', 'r') as data:
    i = 0
    start, end = data['offsets'][i:i + 2]
    time = data['time'][start:end]          # (T,)，真实相对秒数
    features = data['features'][start:end]  # (T, 3)
    trajectory_id = data['tracks/trajectory_id'].asstr()[i]
    # 如需平滑对照，应明确替换速度列，不要误以为主输入已平滑。
    # features[:, 2] = data['speed_smoothed'][start:end]
```

时间作为时间编码的输入，与三维物理特征分开保存。
来源 ID、车型、车道、原始加速度和质量标记不是默认模型输入。
文件不含行为标签、簇数、训练/验证划分、标准化参数或预生成掩码。
后续先确定数据划分，仅用训练数据拟合标准化参数；按批次补齐，并按长度生成有效掩码。
合法的零速度、相对起点为零都不能当作 PAD。

**这份 H5 不兼容现有 token 数据加载器。** 连续输入层、掩码重建头、变长批处理
及表示提取使用独立的 `python -m src.experiment.run_ngsim` 入口，详见
[US-101 单分支预训练说明](../../../configs/baseline/ngsim.md)。原来的 Porto 训练入口保持原样。
