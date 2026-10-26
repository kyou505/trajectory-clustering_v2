"""MSTM 预训练模型，与联合聚类阶段共用时空编码接口。"""
import torch
from torch import nn

from .encoder import TrajectoryEncoder
from .trajectory_encoding import TrajectoryEncodingMixin

class SPPretrainModel(TrajectoryEncodingMixin, nn.Module):
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
    ):
        super().__init__()
        self.location_vocab_size = location_vocab_size
        self.time_vocab_size = time_vocab_size

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

        self.prediction_transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

        self.location_head = nn.Linear(d_model, location_vocab_size, bias=False)
        self.time_head = nn.Linear(d_model, time_vocab_size, bias=False)

        # 与Embedding共享权重
        self.location_head.weight = self.encoder.embedding.location_embedding.weight
        self.time_head.weight = self.encoder.embedding.time_embedding.weight

        self.reset_parameters()

    def forward(
            self,
            location_ids,
            time_ids,
            attention_mask,
    ):
        hidden_states = self.hidden(location_ids, time_ids, attention_mask)
        prediction_hidden = self.prediction_transform(hidden_states)
        location_logits = self.location_head(prediction_hidden)
        time_logits = self.time_head(prediction_hidden)
        return {
            "hidden_states": hidden_states,
            "location_logits": location_logits,
            "time_logits": time_logits,
        }

    def reset_parameters(self):
        nn.init.kaiming_uniform_(
            self.encoder.embedding.location_embedding.weight,
            nonlinearity="relu",
        )
        nn.init.kaiming_uniform_(
            self.encoder.embedding.time_embedding.weight,
            nonlinearity="relu",
        )
        with torch.no_grad():
            self.encoder.embedding.location_embedding.weight[0].zero_()
            self.encoder.embedding.time_embedding.weight[0].zero_()


class MSTMLoss(nn.Module):
    def __init__(
            self,
            time_loss_weight=0.1,
            ignore_index=-100,
    ):
        super().__init__()
        self.time_loss_weight = time_loss_weight
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(
            self,
            location_logits,
            time_logits,
            location_targets,
            time_targets,
    ):
        location_vocab_size = location_logits.size(-1)
        time_vocab_size = time_logits.size(-1)
        location_loss = self.cross_entropy(
            location_logits.reshape(-1, location_vocab_size),
            location_targets.reshape(-1)
        )
        time_loss = self.cross_entropy(
            time_logits.reshape(-1, time_vocab_size),
            time_targets.reshape(-1)
        )
        total_loss = location_loss + time_loss * self.time_loss_weight
        return {
            "loss": total_loss,
            "location_loss": location_loss,
            "time_loss": time_loss,
        }
