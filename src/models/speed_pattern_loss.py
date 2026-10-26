"""直接约束用于聚类的池化表示，不增加独立对比投影头。"""

from contextlib import contextmanager

import torch
from torch.nn import functional as F


@contextmanager
def stable_auxiliary_forward(model):
    """辅助前向关闭 dropout 但保留梯度，结束后恢复每个子模块的原有模式。"""
    # 逐个记录子模块的 training 状态，而不是只记录顶层，
    # 以便正确恢复部分子模块被单独冻结为 eval 的情况
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, mode in modes:
            module.training = mode


def triplet_objective(z, positive, negative, accepted, margin):
    """归一化 z 上的平方欧氏三元组损失：max(d正 - d负 + margin, 0)。

    accepted 标记每个三元组的负例是否有效，只有被接受的样本参与各项统计。
    """
    # L2 归一化后，平方欧氏距离取值范围为 [0, 4]
    z, positive, negative = [F.normalize(x, dim=-1) for x in (z, positive, negative)]
    d_positive = (z - positive).square().sum(-1)
    d_negative = (z - negative).square().sum(-1)
    raw = F.relu(d_positive - d_negative + margin)

    weights = accepted.to(z.dtype)
    # 全部负例被拒绝时返回仍可反传的零，避免空集均值产生 NaN。
    denominator = weights.sum().clamp_min(1.0)
    return {
        "loss": (raw * weights).sum() / denominator,
        # 以下为监控指标：被接受样本上的平均正/负距离
        "positive_distance": (d_positive * weights).sum() / denominator,
        "negative_distance": (d_negative * weights).sum() / denominator,
        # 仍违反 margin（损失 > 0）的样本比例
        "active_fraction": ((raw > 0) * weights).sum() / denominator,
        # 负例被接受的比例（分母为全部样本）
        "accepted_fraction": weights.mean(),
    }


def speed_pattern_loss(model, views, margin, device):
    """速度模式三元组损失：以时间形变视图为负例，约束聚类表示 z 对速度模式敏感。

    views 包含 anchor / positive / negative 三个视图，以及 accepted（负例是否有效）
    与 rms_fraction（时间形变幅度）两个字段。
    """
    # 在关闭 dropout 的稳定模式下编码三个视图
    with stable_auxiliary_forward(model):
        vectors = [
            model.encode({key: value.to(device) for key, value in views[name].items()})
            for name in ("anchor", "positive", "negative")
        ]

    result = triplet_objective(*vectors, views["accepted"].to(device), margin)
    # 记录本批次时间形变的平均幅度，仅用于监控
    result["warp_rms_fraction"] = views["rms_fraction"].float().mean().to(device)
    return result
