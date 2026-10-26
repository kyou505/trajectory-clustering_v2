"""读取现有 HDF5/token 数据；分钟时间编码与旧实验保持一致。"""

import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = "portoLocalTempo4Small_v1"


def load_location_vocab(vocab_path):
    """读取位置词表（token 字符串 -> 整数 ID）。"""
    with vocab_path.open("r", encoding="utf-8") as f:
        vocab = json.load(f)
    return vocab


def encode_locations(trajectory, vocab):
    """将空格分隔的位置 token 序列编码为 ID 列表，遇到未登录 token 直接报错。"""
    tokens = trajectory.split()
    encoded = []
    for token in tokens:
        if token in vocab:
            encoded.append(vocab[token])
        else:
            raise ValueError(f"Token '{token}' not found in vocabulary.")
    return encoded


def encode_time(time_sequence):
    """将空格分隔的原始时间序列编码为分钟级时间 ID。

    原始时间每 4 个单位合并为 1 分钟，再对一天（1440 分钟）取模；
    结果整体加 4，把 0~3 留给 PAD/CLS/SEP/MASK 特殊 token。
    """
    tokens = time_sequence.split()
    encoded = []
    for token in tokens:
        if token == "[PAD]":
            encoded.append(0)
        else:
            raw_time = int(token)
            minutes_slot = (raw_time // 4) % 1440
            encoded.append(minutes_slot + 4)
    return encoded


class TrajectoryDataset(Dataset):
    """基础轨迹数据集。

    每条样本编码为 [CLS, p_1, ..., p_L, PAD, ..., PAD, SEP]，
    即 SEP 固定位于补齐后序列的末尾，长度统一为 max_length。
    """

    def __init__(self, dataset_name=DEFAULT_DATASET, data_root=None):
        self.dataset_name = dataset_name
        self.data_dir = Path(data_root or PROJECT_DIR / "data").resolve() / dataset_name
        data_dir = self.data_dir

        # 轨迹数据、轨迹长度与位置词表
        self.data = pd.read_hdf(data_dir / "data_k3.h5", key="x")
        self.length = pd.read_csv(data_dir / "trj_length.csv")["length"].to_numpy()
        self.location_vocab = load_location_vocab(data_dir / "location_vocab.json")
        self.location_vocab_size = len(self.location_vocab)
        # 1440 个分钟槽 + 4 个特殊 token
        self.time_vocab_size = 1444

        if self.data.empty:
            raise ValueError(f"Dataset is empty: {data_dir}")
        # 原始数据已补齐到相同长度，取第一条即可确定补齐长度
        padded_length = len(self.data.iloc[0]["trajectory"].split())
        self.max_length = padded_length + 2  # [CLS] 和 [SEP]

        # 数据一致性检查
        if len(self.data) != len(self.length):
            raise ValueError("Data and length files must have the same number of rows.")
        if sorted(self.location_vocab.values()) != list(range(self.location_vocab_size)):
            raise ValueError("Location vocabulary IDs must be unique and contiguous")
        for token, expected in {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3}.items():
            if self.location_vocab.get(token) != expected:
                raise ValueError(f"Unexpected special token ID: {token}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        location_tokens = encode_locations(row["trajectory"], self.location_vocab)
        time_tokens = encode_time(row["time"])
        length = int(row["trajLen"])

        # 样本级校验：长度合法且与长度文件一致
        if length < 2 or length != int(self.length[idx]) or length > self.max_length - 2:
            raise ValueError(f"Invalid trajectory length at row {idx}")
        if len(time_tokens) != len(location_tokens):
            raise ValueError(f"Location/time length mismatch at row {idx}")
        # 前 length 个必须是真实点（ID >= 4），之后必须全是 PAD（ID == 0）
        if (
            any(x < 4 for x in location_tokens[:length] + time_tokens[:length])
            or any(location_tokens[length:])
            or any(time_tokens[length:])
        ):
            raise ValueError(f"Unexpected real-point/PAD token positions at row {idx}")

        # 首尾拼接 CLS(1) 与 SEP(2)
        location_ids = torch.tensor([1, *location_tokens, 2], dtype=torch.long)
        time_ids = torch.tensor([1, *time_tokens, 2], dtype=torch.long)

        if len(location_ids) != self.max_length:
            raise ValueError(
                f"Inconsistent padded length at row {idx}: "
                f"expected {self.max_length}, got {len(location_ids)}"
            )

        attention_mask = location_ids.ne(0)
        # 只池化真实轨迹点，不包含 CLS、SEP 和补齐位置。
        pooling_mask = torch.zeros_like(location_ids, dtype=torch.bool)
        pooling_mask[1 : length + 1] = True

        return {
            "index": torch.tensor(idx, dtype=torch.long),  # 给每条轨迹增加稳定索引
            "location_ids": location_ids,
            "time_ids": time_ids,
            "attention_mask": attention_mask,
            "pooling_mask": pooling_mask,
            "length": torch.tensor(length, dtype=torch.long),
            "label": torch.tensor(row["label"], dtype=torch.long),
        }
