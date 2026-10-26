"""Transformer 时空编码器与有效轨迹点均值池化。"""
from torch import nn
from .embedding import SpatialTemporalEmbedding

class TrajectoryEncoder(nn.Module):
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
        if d_model % num_heads != 0:
            raise ValueError('d_model % num_heads is not a multiple of num_heads')

        self.embedding = SpatialTemporalEmbedding(
            location_vocab_size,
            time_vocab_size,
            d_model,
            max_length,
            dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, enable_nested_tensor=False)

    def forward(
            self,
            location_ids,
            time_ids,
            attention_mask,
    ):
        embedding = self.embedding(location_ids, time_ids)
        # 原始数据：True = 有效位置，False = PAD
        # src_key_padding_mask：True  = 需要忽略，False = 有效位置
        key_padding_mask = ~attention_mask.bool()
        hidden_states = self.encoder(
            embedding,
            src_key_padding_mask=key_padding_mask
        )
        return hidden_states

def mask_mean_pooling(
        hidden_states,
        pooling_mask,
):
    mask = pooling_mask.unsqueeze(-1).to(hidden_states.dtype)
    hidden_sum = (hidden_states * mask).sum(dim=1)
    valid_count = mask.sum(dim=1).clamp_min(1.0)
    trajectory_vector = hidden_sum / valid_count
    return trajectory_vector
