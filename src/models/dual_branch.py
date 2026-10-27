"""独立路径/节奏编码、分支内掩码预测与加权联合距离。

路径分支只读位置和序列编码；节奏分支读位置、时间和序列编码。
两者不共享参数。速度损失只调用 encode_speed，联合 DEC 仍更新两个分支。
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from src.experiment.experiment_config import validate
from src.models.dec import DECClusteringLayer
from src.models.embedding import PositionalEncoding
from src.models.encoder import TrajectoryEncoder, mask_mean_pooling


class PathEncoder(nn.Module):
    """路径分支：仅用位置嵌入 + 序列位置编码，不接触任何时间信息。"""

    def __init__(
        self,
        location_vocab_size,
        d_model,
        max_length,
        num_heads,
        num_layers,
        dim_feedforward,
        dropout,
    ):
        super().__init__()
        # padding_idx=0：填充位置的嵌入不参与梯度更新
        self.location_embedding = nn.Embedding(
            location_vocab_size, d_model, padding_idx=0
        )
        self.position_encoding = PositionalEncoding(d_model, max_length)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers, enable_nested_tensor=False
        )

    def forward(self, location_ids, attention_mask):
        h = self.location_embedding(location_ids) + self.position_encoding(
            location_ids.size(1)
        )
        h = self.dropout(self.layer_norm(h))
        # attention_mask 中 1 表示有效位置；Transformer 需要的是"需屏蔽"的掩码，故取反
        return self.transformer(h, src_key_padding_mask=~attention_mask.bool())


class PathRhythmEncoder(nn.Module):
    """两个互不共享参数的编码器：路径分支 + 节奏分支（位置 + 时间）。"""

    def __init__(
        self,
        location_vocab_size,
        time_vocab_size,
        d_model,
        max_length,
        num_heads,
        num_layers,
        dim_feedforward,
        dropout,
    ):
        super().__init__()
        self.path = PathEncoder(
            location_vocab_size,
            d_model,
            max_length,
            num_heads,
            num_layers,
            dim_feedforward,
            dropout,
        )
        self.rhythm = TrajectoryEncoder(
            location_vocab_size=location_vocab_size,
            time_vocab_size=time_vocab_size,
            d_model=d_model,
            max_length=max_length,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def forward(self, location_ids, time_ids, attention_mask):
        """返回 (路径隐状态, 节奏隐状态)，形状均为 B×L×d_model。"""
        return (
            self.path(location_ids, attention_mask),
            self.rhythm(location_ids, time_ids, attention_mask),
        )


class DualBranchPretrainModel(nn.Module):
    """双分支预训练模型：路径分支预测被遮盖的位置，节奏分支预测被遮盖的时间。"""

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
        path_weight=0.5,
        distance_scale=100.0,
    ):
        super().__init__()
        # path_weight 即联合距离中的 lambda，distance_scale 为整体距离缩放
        if not math.isfinite(path_weight) or not 0 < path_weight < 1:
            raise ValueError("path_weight must be in (0, 1)")
        if not math.isfinite(distance_scale) or distance_scale <= 0:
            raise ValueError("distance_scale must be positive and finite")
        self.path_weight, self.distance_scale = path_weight, distance_scale
        # 记录构造参数，写入预训练 checkpoint，供聚类阶段校验配置是否一致
        self.pretrain_kwargs = dict(
            location_vocab_size=location_vocab_size,
            time_vocab_size=time_vocab_size,
            d_model=d_model,
            max_length=max_length,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            path_weight=path_weight,
            distance_scale=distance_scale,
        )
        self.encoder = PathRhythmEncoder(
            location_vocab_size,
            time_vocab_size,
            d_model,
            max_length,
            num_heads,
            num_layers,
            dim_feedforward,
            dropout,
        )

        # 掩码预测头：先过一层变换，再用与嵌入共享的权重投影到词表
        self.path_prediction_transform = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU()
        )
        self.rhythm_prediction_transform = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU()
        )
        self.location_head = nn.Linear(d_model, location_vocab_size, bias=False)
        self.time_head = nn.Linear(d_model, time_vocab_size, bias=False)
        # 只在各自分支内部 tying，不能把位置嵌入跨分支共享。
        self.location_head.weight = self.encoder.path.location_embedding.weight
        self.time_head.weight = self.encoder.rhythm.embedding.time_embedding.weight

        # 统一初始化所有嵌入，并把 padding（索引 0）对应的向量置零
        for embedding in (
            self.encoder.path.location_embedding,
            self.encoder.rhythm.embedding.location_embedding,
            self.encoder.rhythm.embedding.time_embedding,
        ):
            nn.init.kaiming_uniform_(embedding.weight, nonlinearity="relu")
            with torch.no_grad():
                embedding.weight[0].zero_()

    def forward(self, location_ids, time_ids, attention_mask):
        path, rhythm = self.encoder(location_ids, time_ids, attention_mask)
        return {
            "path_hidden_states": path,
            "rhythm_hidden_states": rhythm,
            # 位置只由路径分支预测，时间只由节奏分支预测
            "location_logits": self.location_head(self.path_prediction_transform(path)),
            "time_logits": self.time_head(self.rhythm_prediction_transform(rhythm)),
        }

    def mstm(self, batch):
        """掩码时空建模（MSTM）预训练的前向入口。"""
        # 两个分支都只读 masked 输入，禁止从原始字段绕过遮盖。
        # 显式调用基类 forward，避免子类重写的 forward 被误用
        return DualBranchPretrainModel.forward(
            self,
            batch["masked_location_ids"],
            batch["masked_time_ids"],
            batch["attention_mask"],
        )

    def encode_components(self, view):
        """分别得到路径、节奏两个分支的掩码平均池化表示（未归一化）。"""
        path, rhythm = self.encoder(
            view["location_ids"], view["time_ids"], view["attention_mask"]
        )
        return {
            "path": mask_mean_pooling(path, view["pooling_mask"]),
            "rhythm": mask_mean_pooling(rhythm, view["pooling_mask"]),
        }

    def fuse(self, path, rhythm):
        """把两个分支 L2 归一化后按权重缩放并拼接，得到联合表示。"""
        # 欧氏距离精确等于 scale * [lambda D_path + (1-lambda) D_rhythm]。
        # 中心由同一坐标系下的 KMeans 初始化；不在 DEC 内另行归一化中心。
        path_coef = math.sqrt(self.distance_scale * self.path_weight)
        rhythm_coef = math.sqrt(self.distance_scale * (1 - self.path_weight))
        return torch.cat(
            (
                path_coef * F.normalize(path, dim=-1),
                rhythm_coef * F.normalize(rhythm, dim=-1),
            ),
            dim=-1,
        )

    def encode(self, view):
        """完整编码：分支表示 -> 加权融合。"""
        parts = self.encode_components(view)
        return self.fuse(parts["path"], parts["rhythm"])

    def encode_speed(self, view):
        """仅供速度损失使用的节奏分支表示。"""
        # 不执行路径分支，速度梯度也不会进入路径分支或其嵌入。
        h = self.encoder.rhythm(
            view["location_ids"], view["time_ids"], view["attention_mask"]
        )
        return mask_mean_pooling(h, view["pooling_mask"])


class DualBranchTrajectoryModel(DualBranchPretrainModel):
    """在双分支编码器之上接 DEC 聚类层，用于聚类阶段训练。"""

    def __init__(self, num_clusters=12, **kwargs):
        super().__init__(**kwargs)
        # 联合表示由两个分支拼接而成，维度为 2 * d_model
        dimension = self.encoder.path.location_embedding.embedding_dim
        self.clustering_layer = DECClusteringLayer(num_clusters, 2 * dimension)

    def forward(self, view1, view2):
        """对两个增强视图分别编码并计算软分配。"""
        z1, z2 = self.encode(view1), self.encode(view2)
        return {
            "z1": z1,
            "z2": z2,
            "q1": self.clustering_layer(z1),
            "q2": self.clustering_layer(z2),
        }

    def load_pretrained_components(self, checkpoint_path, map_location="cpu"):
        """加载双分支预训练权重（不含聚类中心），加载前严格校验兼容性。"""
        checkpoint = torch.load(
            checkpoint_path, map_location=map_location, weights_only=True
        )
        # 校验 checkpoint 类型、实验组别及关键超参数
        if checkpoint.get("experiment_schema") != "sp_dtc_pretrain_v1":
            raise ValueError("Expected an SP-DTC pretraining checkpoint")
        cfg = validate(checkpoint["configs"])
        if cfg["group"] != "dual_branch":
            raise ValueError("Dual-branch clustering requires dual-branch pretraining")
        for key in ("path_weight", "distance_scale"):
            if cfg["dual_branch"][key] != getattr(self, key):
                raise ValueError(f"Incompatible pretraining {key}")
        if checkpoint.get("model_kwargs") != self.pretrain_kwargs:
            raise ValueError("Incompatible pretraining encoder settings")

        # 先检查 key/shape，避免部分加载不兼容权重后才报错。
        state = checkpoint["model_state_dict"]
        expected = self.state_dict()
        # 预训练模型没有聚类层，聚类中心之后由 KMeans 初始化
        required = set(expected) - {"clustering_layer.cluster_centers"}
        if set(state) != required or any(
            state[k].shape != expected[k].shape for k in required
        ):
            raise ValueError("Incompatible dual-branch pretraining parameters")
        self.load_state_dict(state, strict=False)
        return {
            "epoch": checkpoint.get("epoch"),
            "valid_metrics": checkpoint.get("valid_metrics"),
            "configs": cfg,
        }
