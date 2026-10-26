"""聚类双视图：删点与整体时间平移；不同于速度模式对比使用的局部时间形变。"""

import torch
from torch.utils.data import Dataset

# 特殊 token 编号，与基础数据集的编码保持一致
PAD_ID = 0
CLS_ID = 1
SEP_ID = 2

# 一天按分钟划分的时间槽数量
NUM_TIME_SLOTS = 1440


def point_dropout(sample, dropout_rate=0.1, generator=None):
    """随机删除轨迹点，生成第一个视图。

    序列格式为 [CLS, p_1, ..., p_L, ..., SEP]，第 0 位是 CLS，
    有效轨迹点位于 1..L。删点后保留的点按原始顺序左对齐，其余位置填充 PAD。
    """
    location_ids = sample["location_ids"]
    time_ids = sample["time_ids"]
    length = int(sample["length"].item())

    # 至少删除 1 个点，保证视图与原样本不同
    drop_count = max(1, int(length * dropout_rate))
    keep_count = length - drop_count

    # 随机选择要保留的点，排序以维持时间顺序；+1 跳过开头的 CLS
    permutation = torch.randperm(length, generator=generator)
    keep_positions = permutation[:keep_count].sort().values + 1
    kept_locations = location_ids[keep_positions]
    kept_times = time_ids[keep_positions]

    # 重新组装序列：CLS + 保留点 + PAD... + SEP
    new_locations = torch.full_like(location_ids, PAD_ID)
    new_times = torch.full_like(time_ids, PAD_ID)
    new_locations[0] = CLS_ID
    new_times[0] = CLS_ID
    new_locations[1 : keep_count + 1] = kept_locations
    new_times[1 : keep_count + 1] = kept_times
    new_locations[-1] = SEP_ID
    new_times[-1] = SEP_ID

    # 池化只作用于保留下来的有效轨迹点；注意力覆盖所有非 PAD 位置
    pooling_mask = torch.zeros_like(sample["pooling_mask"])
    pooling_mask[1 : keep_count + 1] = True
    attention_mask = new_locations.ne(PAD_ID)

    return {
        "location_ids": new_locations,
        "time_ids": new_times,
        "attention_mask": attention_mask,
        "pooling_mask": pooling_mask,
        "length": torch.tensor(keep_count, dtype=torch.long),
    }


def time_offset(sample, max_offset_minutes=2, generator=None):
    """对整条轨迹做统一的时间平移，生成第二个视图。

    所有有效点平移相同的分钟数（1..max_offset_minutes），并在一天内循环取模。
    位置序列与各类 mask 保持不变。
    """
    new_times = sample["time_ids"].clone()
    valid_positions = sample["pooling_mask"]

    # 随机采样平移量，范围 [1, max_offset_minutes]
    offset = torch.randint(
        low=1,
        high=max_offset_minutes + 1,
        size=(),
        generator=generator,
    ).item()

    # time_ids 前 4 个编号留给特殊 token，先减去偏置还原为分钟槽，
    # 平移后对一天取模，再加回偏置
    time_slots = new_times[valid_positions] - 4
    shifted_slots = (time_slots + offset) % NUM_TIME_SLOTS
    new_times[valid_positions] = shifted_slots + 4

    return {
        "location_ids": sample["location_ids"].clone(),
        "time_ids": new_times,
        "attention_mask": sample["attention_mask"].clone(),
        "pooling_mask": sample["pooling_mask"].clone(),
        "length": sample["length"].clone(),
        "offset": torch.tensor(offset, dtype=torch.long),
    }


class ContrastiveTrajectoryDataset(Dataset):
    """在基础轨迹数据集上包装对比学习所需的双视图。

    view1 为删点视图，view2 为时间平移视图；原样本字段原样保留。
    """

    def __init__(
        self,
        base_dataset,
        dropout_rate=0.1,
        max_offset_minutes=2,
        seed=0,
    ):
        self.base_dataset = base_dataset
        self.dropout_rate = dropout_rate
        self.max_offset_minutes = max_offset_minutes
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        """每个 epoch 开始时调用，使不同 epoch 产生不同的增强结果。"""
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        sample_index = int(sample["index"].item())

        # 随机种子由 (seed, epoch, 样本索引) 决定：
        # 同一 epoch 内结果可复现，且与 DataLoader 的 worker 数无关
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch * 1_000_003 + sample_index)

        view1 = point_dropout(
            sample,
            dropout_rate=self.dropout_rate,
            generator=generator,
        )
        view2 = time_offset(
            sample,
            max_offset_minutes=self.max_offset_minutes,
            generator=generator,
        )

        result = dict(sample)
        result["view1"] = view1
        result["view2"] = view2
        return result
