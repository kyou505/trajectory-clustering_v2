"""速度模式视图：正例保留局部模式，负例重分配局部用时并保持路径与总时长。"""

import numpy as np
import torch

VIEW_KEYS = ("location_ids", "time_ids", "attention_mask", "pooling_mask")


def select_view(sample, keep, times=None):
    """按 keep 选出的有效点重组序列，可用 times 替换这些点的时间 token。

    序列格式与原样本一致：[CLS, p_1, ..., p_k, PAD, ..., PAD, SEP]。
    """
    # 有效轨迹点（不含 CLS、SEP、PAD）的下标
    positions = torch.where(sample["pooling_mask"])[0]
    ids = sample["location_ids"][positions].numpy()
    t = sample["time_ids"][positions].numpy() if times is None else np.asarray(times)

    loc = torch.zeros_like(sample["location_ids"])
    clock = torch.zeros_like(sample["time_ids"])
    # 首位 CLS(1)、末位 SEP(2)，中间依次填入保留点，其余保持 PAD(0)
    loc[0] = clock[0] = 1
    loc[-1] = clock[-1] = 2
    loc[1 : len(keep) + 1] = torch.tensor(ids[keep])
    clock[1 : len(keep) + 1] = torch.tensor(t[keep])

    # 只对真实轨迹点做池化
    pool = torch.zeros_like(sample["pooling_mask"])
    pool[1 : len(keep) + 1] = True
    return {
        "location_ids": loc,
        "time_ids": clock,
        "pooling_mask": pool,
        "attention_mask": loc.ne(0),
    }


def make_views(sample, config, *, seed, epoch, stage):
    """返回原始、正例、负例视图及形变是否足够明显的标记。

    正例仅删除内部采样点并整体平移时钟；负例改变局部用时分配。
    按有效采样点的归一化序号定位扰动，不读取坐标、路径距离或类别标签。
    """
    # 每条样本使用独立随机数，不改变训练的 shuffle、mask 和 dropout 序列。
    rng = np.random.default_rng(
        seed + 1_000_003 * epoch + int(sample["index"]) + 91_000_019 * stage
    )
    positions = torch.where(sample["pooling_mask"])[0]
    # 时间 token 减去 4 个特殊 token 偏移，还原为一天内的分钟数
    t = sample["time_ids"][positions].numpy().astype(float) - 4
    # 用分钟 token 展开跨午夜时间，不读取生成器的连续时间。
    t = np.r_[t[0], t[0] + np.cumsum(np.mod(np.diff(t), 1440))]
    n = len(t)
    if config["warp_axis"] != "sample_index":
        raise ValueError("Speed-pattern warping requires the sample_index axis")
    # 扰动中心按序列顺序定位，不假设采样点之间的物理距离相等。
    progress = np.linspace(0.0, 1.0, n)

    # ---- 正例：随机删除部分内部点（保留首尾），并把所有时间整体平移 1~2 分钟 ----
    interior = np.arange(1, n - 1)
    count = min(max(1, int(n * config["positive_dropout"])), max(0, n - 2))
    dropped = rng.choice(interior, count, replace=False) if count else []
    keep = np.setdiff1d(np.arange(n), dropped)
    offset = int(rng.integers(1, 3))
    positive = select_view(sample, keep, ((t + offset) % 1440 + 4).astype(np.int64))

    # ---- 负例：路径与总时长不变，只重新分配各段用时 ----
    duration = t[-1] - t[0]
    negative_times = t.copy()
    accepted, rms, changed = False, 0.0, 0.0
    if duration > 0 and progress[-1] > 0 and n >= 3:
        # 每一段（相邻两点之间）的中点进度，用于计算该段的形变权重
        midpoint = (progress[:-1] + progress[1:]) / 2
        # 多次尝试，直到形变幅度超过阈值
        for _ in range(config["attempts"]):
            center = rng.uniform(*config["center"])
            width = rng.uniform(*config["width"])
            amplitude = rng.uniform(*config["amplitude"]) * rng.choice([-1, 1])
            # 随机位置上的平滑权重调整各段用时，再归一化以保持总时长。
            bump = np.exp(-0.5 * ((midpoint - center) / width) ** 2)
            weights = np.exp(amplitude * bump)
            dt = np.diff(t) * weights
            candidate = np.r_[t[0], t[0] + duration * np.cumsum(dt) / dt.sum()]
            # 形变后仍量化到分钟，保持各组实验的输入精度一致。
            candidate = np.floor(candidate + 1e-8)
            candidate[0], candidate[-1] = t[0], t[-1]
            # 形变强度：相对总时长的均方根时间偏移
            score = float(np.sqrt(np.mean(((candidate - t) / duration) ** 2)))
            # 以模型实际可见的分钟时间判断形变；不足阈值的负例不参与损失。
            if score >= config["minimum_rms_fraction"]:
                negative_times, rms, accepted = candidate, score, True
                # 时间 token 发生变化的点所占比例
                changed = float(np.mean(candidate != t))
                break
    # 负例保留全部点，只替换时间；未通过阈值时等同于原始时间
    negative = select_view(
        sample, np.arange(n), (negative_times % 1440 + 4).astype(np.int64)
    )
    anchor = {key: sample[key].clone() for key in VIEW_KEYS}
    return {
        "anchor": anchor,
        "positive": positive,
        "negative": negative,
        "accepted": accepted,
        "rms_fraction": rms,
        "changed_fraction": changed,
    }


def batch_views(batch, config, *, seed, epoch, stage):
    """对一个 batch 逐条生成三种视图，并堆叠回 batch 形式的张量。"""
    rows = []
    for i in range(len(batch["index"])):
        sample = {key: batch[key][i].cpu() for key in (*VIEW_KEYS, "index")}
        rows.append(make_views(sample, config, seed=seed, epoch=epoch, stage=stage))
    result = {
        name: {key: torch.stack([row[name][key] for row in rows]) for key in VIEW_KEYS}
        for name in ("anchor", "positive", "negative")
    }
    # 逐样本的形变统计，供负例筛选和日志记录使用
    for key in ("accepted", "rms_fraction", "changed_fraction"):
        result[key] = torch.tensor([row[key] for row in rows])
    return result
