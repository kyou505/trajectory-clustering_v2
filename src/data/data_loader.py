"""沿用样本级 80/10/10 预训练划分与确定性验证遮盖。"""

import torch
from torch.utils.data import DataLoader, random_split

from src.data.augmentation import ContrastiveTrajectoryDataset
from src.data.data_process import TrajectoryDataset
from src.data.mstm import MSTMDataset


def create_data_loaders(
    batch_size=32,
    seed=0,
    dataset_name="portoLocalTempo4Small_v1",
    base_dataset=None,
):
    """构建 MSTM 预训练所用的 train / valid / test 三个 DataLoader。

    按样本做 80/10/10 随机划分；训练集每次随机遮盖，
    验证集与测试集使用固定种子的确定性遮盖，保证评估结果可复现。
    """
    if base_dataset is None:
        base_dataset = TrajectoryDataset(dataset_name=dataset_name)
    total_size = len(base_dataset)

    # 80% 训练，剩余部分在验证与测试之间平分
    train_size = int(total_size * 0.8)
    valid_size = (total_size - train_size) // 2
    test_size = total_size - train_size - valid_size

    # 固定种子划分，保证不同实验的划分一致
    train_base, valid_base, test_base = random_split(
        base_dataset,
        [train_size, valid_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )

    # 验证集与测试集使用不同的种子偏移，避免两者遮盖位置相同
    train_dataset = MSTMDataset(train_base, deterministic=False)
    valid_dataset = MSTMDataset(valid_base, deterministic=True, seed=seed + 10_000)
    test_dataset = MSTMDataset(test_base, deterministic=True, seed=seed + 20_000)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, valid_loader, test_loader


def create_single_view_data_loader(batch_size=32, seed=0, mask_ratio=0.15,
                                   shuffle=True, base_dataset=None):
    """全量原始轨迹用于 DEC；仅附加 MSTM 字段，不生成聚类增强视图。"""
    if base_dataset is None:
        base_dataset = TrajectoryDataset()
    dataset = MSTMDataset(base_dataset, mask_ratio=mask_ratio,
                          deterministic=False, seed=seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      generator=torch.Generator().manual_seed(seed))


def create_contrastive_data_loader(
    batch_size=32,
    seed=0,
    mask_ratio=0.15,
    dropout_rate=0.1,
    max_offset_minutes=2,
    shuffle=True,
    base_dataset=None,
):
    """构建聚类阶段的对比学习 DataLoader（使用全部样本，不做划分）。

    数据流：TrajectoryDataset -> MSTMDataset（随机遮盖）
    -> ContrastiveTrajectoryDataset（附加删点视图与时间平移视图）。
    """
    if base_dataset is None:
        base_dataset = TrajectoryDataset()

    # 第一层：MSTM 随机遮盖，供掩码重建损失使用
    mstm_dataset = MSTMDataset(
        base_dataset=base_dataset,
        mask_ratio=mask_ratio,
        deterministic=False,
        seed=seed,
    )
    # 第二层：在 MSTM 样本上生成两个增强视图，供对比损失使用
    dataset = ContrastiveTrajectoryDataset(
        base_dataset=mstm_dataset,
        dropout_rate=dropout_rate,
        max_offset_minutes=max_offset_minutes,
        seed=seed,
    )

    # 固定 shuffle 的随机源，保证样本顺序可复现
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )
