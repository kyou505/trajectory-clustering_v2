"""连续位置/速度输入的单分支 Transformer；用已知秒级时间重建遮盖观测。"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from src.models.encoder import mask_mean_pooling


class ContinuousMaskedEncoder(nn.Module):
    def __init__(self, d_model=256, num_heads=2, num_layers=2, dim_feedforward=1024, dropout=.1):
        super().__init__()
        if d_model % 2 or d_model % num_heads:
            raise ValueError('d_model must be even and divisible by num_heads')
        self.feature_projection = nn.Linear(3, d_model)
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.mask_token, std=.02)
        frequencies = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.) / d_model))
        self.register_buffer('time_frequencies', frequencies)
        self.input_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout,
                                           activation='gelu', batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.reconstruction_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 3))

    def forward(self, features, time, valid_mask, prediction_mask=None):
        if prediction_mask is None:
            prediction_mask = torch.zeros_like(valid_mask)
        if (prediction_mask & ~valid_mask).any() or not valid_mask.any(dim=1).all():
            raise ValueError('Invalid prediction or padding mask')
        # 在投影之前抹去目标数值，防止被遮盖的坐标或速度进入编码器。
        hidden = self.feature_projection(features.masked_fill(prediction_mask.unsqueeze(-1), 0))
        hidden = torch.where(prediction_mask.unsqueeze(-1), self.mask_token, hidden)
        angle = time.unsqueeze(-1) * self.time_frequencies
        time_embedding = torch.stack((angle.sin(), angle.cos()), dim=-1).flatten(-2)
        hidden = self.dropout(self.input_norm(hidden + time_embedding))
        hidden = self.encoder(hidden, src_key_padding_mask=~valid_mask)
        return {'reconstruction': self.reconstruction_head(hidden),
                'embedding': mask_mean_pooling(hidden, valid_mask)}


def masked_reconstruction_loss(prediction, target, mask):
    """只计算遮盖点；先对每条轨迹求均值，再对样本平均，避免长轨迹支配目标。"""
    counts = mask.sum(dim=1)
    if not (counts > 0).all():
        raise ValueError('Each sample must have at least one masked point')
    errors = F.smooth_l1_loss(prediction, target, reduction='none', beta=1.)
    per_trajectory = (errors * mask.unsqueeze(-1)).sum(dim=(1, 2)) / (counts * target.shape[-1])
    return per_trajectory.mean()


def masked_mse_per_trajectory(prediction, target, mask):
    counts = mask.sum(dim=1)
    if not (counts > 0).all():
        raise ValueError('Each sample must have masked observations')
    return ((prediction - target).square() * mask.unsqueeze(-1)).sum(dim=1) / counts.unsqueeze(-1)
