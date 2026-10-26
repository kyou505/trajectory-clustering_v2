"""Student-t 软分配、全局频率修正锐化及跨视图 KL。"""

import torch
from torch import nn
import torch.nn.functional as F

class DECClusteringLayer(nn.Module):
    def __init__(
            self,
            num_clusters,
            embedding_dim,
    ):
        super().__init__()
        self.num_clusters = num_clusters
        self.embedding_dim = embedding_dim
        self.cluster_centers = nn.Parameter(
            torch.empty(num_clusters, embedding_dim)
        )
        # 训练前由 K-means 中心覆盖；初始化顺序与原基线一致。
        nn.init.xavier_uniform_(self.cluster_centers)

    def forward(self, z):
        """
        :param z: [B, D]
        :return: q: [B, K]
        """
        # [B, 1, D] - [1, K, D] -> [B, K, D]
        difference = z.unsqueeze(1) - self.cluster_centers.unsqueeze(0)
        # ||zi - cj||²，形状为 [B, K]
        difference_squared = difference.pow(2).sum(dim=-1)
        # Student's t-distribution
        numerator = 1.0 / (1.0 + difference_squared)
        # 归一化
        q = numerator / numerator.sum(dim=1, keepdim=True).clamp(min=1e-12)

        return q

@torch.no_grad()
def target_distribution(q):
    cluster_frequency = q.sum(dim=0, keepdim=True).clamp(min=1e-12)
    weight = q.pow(2) / cluster_frequency
    p = weight / weight.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return p

class DECLoss(nn.Module):
    def forward(
            self,
            q,
            p,
            sample_weight=None,
    ):
        log_q = q.clamp(min=1e-12).log()
        per_sample_loss = F.kl_div(
            input=log_q,
            target=p,
            reduction="none",
        ).sum(dim=1)
        if sample_weight is None:
            return per_sample_loss.mean()
        sample_weight = sample_weight.detach().to(
            device=q.device,
            dtype=q.dtype,
        )
        loss = (per_sample_loss * sample_weight).sum() / sample_weight.sum()
        return loss

class ConDTCClusteringLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec_loss = DECLoss()

    def forward(self, q1, q2, p1, p2, sample_weight=None):
        if q1.shape != q2.shape:
            raise ValueError("q1 and q2 must have the same shape")
        if q1.shape != p1.shape or q2.shape != p2.shape:
            raise ValueError("Each target must match its assignment shape")
        # 轮初目标保持固定：视图 2 的目标监督视图 1，反向同理。
        p1 = p1.detach()
        p2 = p2.detach()
        loss_q1_from_p2 = self.dec_loss(q=q1, p=p2, sample_weight=sample_weight)
        loss_q2_from_p1 = self.dec_loss(q=q2, p=p1, sample_weight=sample_weight)
        loss = 0.5 * (loss_q1_from_p2 + loss_q2_from_p1)
        return {
            "loss": loss,
            "loss_view1": loss_q1_from_p2,
            "loss_view2": loss_q2_from_p1
        }
