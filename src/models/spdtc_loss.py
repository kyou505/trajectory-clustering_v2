"""联合聚类基础目标：MSTM + 对称跨视图 KL。

速度模式三元组损失单独定义，由两个训练阶段以相同权重加入。
"""
from torch import nn

from src.models.dec import ConDTCClusteringLoss
from src.models.pretrain_model import MSTMLoss


class SPDTCLoss(nn.Module):
    def __init__(self, time_loss_weight=0.1, clustering_loss_weight=0.5):
        super().__init__()
        self.mstm_loss = MSTMLoss(time_loss_weight)
        self.clustering_loss = ConDTCClusteringLoss()
        self.clustering_loss_weight = clustering_loss_weight

    def forward(self, mstm, batch, output, p1, p2):
        representation = self.mstm_loss(
            mstm["location_logits"], mstm["time_logits"],
            batch["location_targets"], batch["time_targets"],
        )
        clustering = self.clustering_loss(output["q1"], output["q2"], p1, p2)
        base_loss = representation["loss"] + self.clustering_loss_weight * clustering["loss"]
        return {
            "loss": base_loss,
            "base_loss": base_loss,
            "representation_loss": representation["loss"],
            "location_loss": representation["location_loss"],
            "time_loss": representation["time_loss"],
            "clustering_loss": clustering["loss"],
            "clustering_loss_view1": clustering["loss_view1"],
            "clustering_loss_view2": clustering["loss_view2"],
        }
