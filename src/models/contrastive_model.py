"""双视图时空表示与 DEC 聚类，共用轨迹编码和掩码预测接口。"""

import torch
from torch import nn

from src.experiment.experiment_config import validate
from src.models.dec import DECClusteringLayer
from src.models.encoder import TrajectoryEncoder
from src.models.trajectory_encoding import TrajectoryEncodingMixin


class SPTrajectoryModel(TrajectoryEncodingMixin, nn.Module):
    """SP-DTC 主模型：轨迹编码器 + MSTM 预测头 + DEC 聚类层。

    注意：子模块的定义顺序决定参数初始化时随机数的消耗顺序，
    调整顺序会导致与旧实验结果无法对齐。
    """

    def __init__(
        self,
        location_vocab_size=148,
        time_vocab_size=1444,
        d_model=256,
        max_length=62,
        num_heads=2,
        num_layers=2,
        dim_feedforward=1024,
        dropout=0.1,
        num_clusters=12,
    ):
        super().__init__()

        # 共享的轨迹 Transformer 编码器
        self.encoder = TrajectoryEncoder(
            location_vocab_size=location_vocab_size,
            time_vocab_size=time_vocab_size,
            d_model=d_model,
            max_length=max_length,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        # DEC 聚类层：基于 Student-t 分布计算软分配 q
        self.clustering_layer = DECClusteringLayer(
            num_clusters=num_clusters,
            embedding_dim=d_model,
        )

        # MSTM 掩码预测头：共享变换 + 位置/时间两个输出层
        self.prediction_transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.location_head = nn.Linear(d_model, location_vocab_size, bias=False)
        self.time_head = nn.Linear(d_model, time_vocab_size, bias=False)
        # 输出层与输入嵌入共享权重（weight tying）
        self.location_head.weight = self.encoder.embedding.location_embedding.weight
        self.time_head.weight = self.encoder.embedding.time_embedding.weight

        # 保留基线投影头的参数与初始化顺序以便核对旧实验；
        # 当前实验不启用实例/簇 InfoNCE，速度模式损失直接作用于 z。
        # 实例级投影头
        self.instance_projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 128),
        )
        # 聚类级投影头
        self.cluster_projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, num_clusters),
        )

    def forward(self, view1, view2):
        """对两个增强视图分别编码，返回表示、软分配与投影头输出。"""
        # 轨迹级表示
        z1 = self.encode(view1)
        z2 = self.encode(view2)
        # DEC 软分配
        q1 = self.clustering_layer(z1)
        q2 = self.clustering_layer(z2)
        # 实例级投影
        head_in1 = self.instance_projector(z1)
        head_in2 = self.instance_projector(z2)
        # 聚类级投影，转置为 [num_clusters, batch]，每行对应一个簇
        head_cl1 = self.cluster_projector(z1).transpose(0, 1)
        head_cl2 = self.cluster_projector(z2).transpose(0, 1)
        return {
            "z1": z1,
            "z2": z2,
            "q1": q1,
            "q2": q2,
            "head_in1": head_in1,
            "head_in2": head_in2,
            "head_cl1": head_cl1,
            "head_cl2": head_cl2,
        }

    def load_pretrained_components(
        self,
        checkpoint_path,
        map_location="cpu",
    ):
        """从 SP-DTC 预训练 checkpoint 加载编码器与 MSTM 预测头。

        聚类层和两个投影头在预训练阶段不存在，允许且只允许这些参数缺失；
        出现任何多余或其他缺失的参数都直接报错。
        """
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)

        # 校验 checkpoint 类型与内容
        if checkpoint.get("experiment_schema") != "sp_dtc_pretrain_v1":
            raise ValueError("Expected an SP-DTC pretraining checkpoint")
        if "model_state_dict" not in checkpoint:
            raise KeyError("checkpoint_path does not contain 'model_state_dict'")
        pretrained_state = checkpoint["model_state_dict"]
        pretrained_config = validate(checkpoint["configs"])

        # 非严格加载，随后手动核对缺失/多余的参数
        incompatible = self.load_state_dict(pretrained_state, strict=False)
        missing_keys = set(incompatible.missing_keys)
        unexpected_keys = set(incompatible.unexpected_keys)
        if unexpected_keys:
            raise RuntimeError(f"incompatible keys: {unexpected_keys}")

        # 仅聚类阶段新增的模块允许缺失
        expected_missing = {
            "clustering_layer.cluster_centers",
            "instance_projector.0.weight",
            "instance_projector.0.bias",
            "instance_projector.2.weight",
            "instance_projector.2.bias",
            "cluster_projector.0.weight",
            "cluster_projector.0.bias",
            "cluster_projector.2.weight",
            "cluster_projector.2.bias",
        }
        if missing_keys != expected_missing:
            raise RuntimeError(f"unexpected missing keys: {missing_keys}")

        return {
            "epoch": checkpoint.get("epoch"),
            "valid_metrics": checkpoint.get("valid_metrics"),
            "configs": pretrained_config,
        }
