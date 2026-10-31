"""联合聚类基础目标：MSTM + 对称跨视图 KL，或原始单视图 DEC。

速度模式三元组损失单独定义，由两个训练阶段以相同权重加入。
"""
from torch import nn

from src.models.dec import ConDTCClusteringLoss, DECLoss
from src.models.pretrain_model import MSTMLoss


class SPDTCLoss(nn.Module):
    def __init__(self, time_loss_weight=0.1, clustering_loss_weight=0.5, view_mode='dual'):
        super().__init__()
        if view_mode not in {'dual', 'single'}:
            raise ValueError('view_mode must be dual or single')
        self.view_mode = view_mode
        self.mstm_loss = MSTMLoss(time_loss_weight)
        self.clustering_loss = ConDTCClusteringLoss() if view_mode == 'dual' else DECLoss()
        self.clustering_loss_weight = clustering_loss_weight

    def forward(self, mstm, batch, output, p1, p2=None):
        representation = self.mstm_loss(
            mstm["location_logits"], mstm["time_logits"],
            batch["location_targets"], batch["time_targets"],
        )
        if self.view_mode == 'single':
            if p2 is not None or output['q'].shape != p1.shape:
                raise ValueError('Single-view DEC requires one target matching q')
            clustering = {'loss': self.clustering_loss(output['q'], p1.detach())}
            view_metrics = {}
        else:
            if p2 is None:
                raise ValueError('Dual-view clustering requires two targets')
            clustering = self.clustering_loss(output['q1'], output['q2'], p1, p2)
            view_metrics = {'clustering_loss_view1': clustering['loss_view1'],
                            'clustering_loss_view2': clustering['loss_view2']}
        base_loss = representation["loss"] + self.clustering_loss_weight * clustering["loss"]
        return {
            "loss": base_loss,
            "base_loss": base_loss,
            "representation_loss": representation["loss"],
            "location_loss": representation["location_loss"],
            "time_loss": representation["time_loss"],
            "clustering_loss": clustering["loss"],
            **view_metrics,
        }
