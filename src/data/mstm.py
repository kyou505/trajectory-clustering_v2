"""掩码时空建模：两类 token 同位置遮盖，返回原始与遮盖视图供训练共用。"""

import torch
from torch.utils.data import Dataset

# 特殊 token 编号：0-3 依次为 PAD/CLS/SEP/MASK，真实 token 从 4 开始
MASK_ID = 3
FIRST_REAL_ID = 4

# CrossEntropyLoss 默认 ignore_index=-100，target 为 -100 的位置不参与计算损失
IGNORE_INDEX = -100


class MSTMDataset(Dataset):
    """在基础轨迹数据集上做 BERT 式的时空联合遮盖。

    位置与时间在同一组位置上遮盖；被选中的位置按 80/10/10 处理：
    80% 替换为 [MASK]，10% 替换为随机真实 token，10% 保持不变。
    """

    def __init__(
        self,
        base_dataset,
        mask_ratio=0.15,
        location_vocab_size=None,
        time_vocab_size=None,
        deterministic=False,  # 控制 MSTM 的随机遮盖是否固定（验证/测试集使用）
        seed=0,
    ):
        self.base_dataset = base_dataset
        self.mask_ratio = mask_ratio

        # base_dataset 可能是 random_split 得到的 Subset，逐层向内找到原始数据集以读取词表大小
        dataset_metadata = base_dataset
        while hasattr(dataset_metadata, "dataset"):
            dataset_metadata = dataset_metadata.dataset
        self.location_vocab_size = (
            location_vocab_size
            if location_vocab_size is not None
            else dataset_metadata.location_vocab_size
        )
        self.time_vocab_size = (
            time_vocab_size
            if time_vocab_size is not None
            else dataset_metadata.time_vocab_size
        )

        self.deterministic = deterministic
        self.seed = seed

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # 取出原始样本
        sample = self.base_dataset[idx]
        original_locations = sample["location_ids"]
        original_times = sample["time_ids"]
        pooling_mask = sample["pooling_mask"]

        # 复制一份作为模型输入，避免修改原样本
        mask_locations = original_locations.clone()
        mask_times = original_times.clone()

        # 初始化预测目标：只有被遮盖的位置才计算损失，其余位置填 IGNORE_INDEX
        location_targets = torch.full_like(original_locations, IGNORE_INDEX)
        time_targets = torch.full_like(original_times, IGNORE_INDEX)

        # 只在真实轨迹点中选择遮盖位置（不含 CLS、SEP 和 PAD）
        candidate_positions = torch.where(pooling_mask)[0]
        num_candidate = len(candidate_positions)
        # 保证轨迹再短也至少遮盖一个点
        num_masked = max(1, int(num_candidate * self.mask_ratio))

        # 随机选出遮盖位置；确定性模式下种子由 (seed, idx) 决定，保证每次结果一致
        generator = None
        if self.deterministic:
            generator = torch.Generator()
            generator.manual_seed(self.seed + idx)
        permutation = torch.randperm(num_candidate, generator=generator)
        masked_positions = candidate_positions[permutation[:num_masked]]

        # 把原始 token 作为预测答案填入 target
        location_targets[masked_positions] = original_locations[masked_positions]
        time_targets[masked_positions] = original_times[masked_positions]

        # 对每个选中位置按 80/10/10 规则扰动输入
        for position in masked_positions:
            probability = torch.rand((), generator=generator).item()
            if probability < 0.8:
                # 80%：位置与时间同时替换为 [MASK]
                mask_locations[position] = MASK_ID
                mask_times[position] = MASK_ID
            elif probability < 0.9:
                # 10%：替换为随机真实 token（跳过 0-3 特殊 token）
                mask_locations[position] = torch.randint(
                    low=FIRST_REAL_ID,
                    high=self.location_vocab_size,
                    size=(1,),
                    generator=generator,
                )
                mask_times[position] = torch.randint(
                    low=FIRST_REAL_ID,
                    high=self.time_vocab_size,
                    size=(1,),
                    generator=generator,
                )
            # 其余 10%：保持原 token 不变

        # mstm_mask 标记全部被选中的位置（含未实际替换的 10%），供运动分支排除
        mstm_mask = torch.zeros_like(original_locations, dtype=torch.bool)
        mstm_mask[masked_positions] = True

        # 保留原样本的全部字段，并追加遮盖后的输入与预测目标
        result = dict(sample)
        result.update(
            {
                "masked_location_ids": mask_locations,
                "masked_time_ids": mask_times,
                "location_targets": location_targets,
                "time_targets": time_targets,
                "mstm_mask": mstm_mask,
            }
        )
        return result
