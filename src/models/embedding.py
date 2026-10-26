"""位置、时间与序列位置嵌入；保持原基线的初始化和求和顺序。"""
import math
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=256, max_length=62):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model should be even number")

        position = torch.arange(
            max_length,
            dtype=torch.float32,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * -math.log(10000.0) / d_model
        )

        positional_encoding = torch.zeros(max_length, d_model)
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0)
        # 位置编码不参与训练
        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, sequence_length):
        if sequence_length > self.positional_encoding.size(1):
            raise ValueError(f"sequence_length {sequence_length} > self.positional_encoding.size: {self.positional_encoding.size(1)}")

        return self.positional_encoding[:, :sequence_length]

class SpatialTemporalEmbedding(nn.Module):
    def __init__(
            self,
            location_vocab_size=148,
            time_vocab_size=1444,
            d_model=256,
            max_length=62,
            dropout=0.1,
    ):
        super().__init__()
        self.location_embedding = nn.Embedding(
            num_embeddings=location_vocab_size,
            embedding_dim=d_model,
            padding_idx=0,
        )

        self.time_embedding = nn.Embedding(
            num_embeddings=time_vocab_size,
            embedding_dim=d_model,
            padding_idx=0,
        )

        self.position_encoding = PositionalEncoding(
            d_model=d_model,
            max_length=max_length,
        )

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, location_ids, time_ids):
        spatial_embedding = self.location_embedding(location_ids)
        time_embedding = self.time_embedding(time_ids)
        sequence_length = location_ids.size(1)
        position_embedding = self.position_encoding(
            sequence_length
        )
        embedding = (
            spatial_embedding
            + time_embedding
            + position_embedding
        )
        embedding = self.layer_norm(embedding)
        embedding = self.dropout(embedding)
        return embedding
