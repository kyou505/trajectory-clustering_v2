"""预训练与联合聚类共用的时空编码、池化和掩码预测接口。"""
from src.models.encoder import mask_mean_pooling


class TrajectoryEncodingMixin:
    def hidden(self, location_ids, time_ids, attention_mask):
        return self.encoder(location_ids, time_ids, attention_mask)

    def encode(self, view):
        h = self.hidden(view['location_ids'], view['time_ids'], view['attention_mask'])
        return mask_mean_pooling(h, view['pooling_mask'])

    def encode_speed(self, view):
        # 单分支保持原行为；双分支模型将此接口路由到节奏分支。
        return self.encode(view)

    def mstm(self, batch):
        h = self.hidden(batch['masked_location_ids'], batch['masked_time_ids'], batch['attention_mask'])
        h = self.prediction_transform(h)
        return {'location_logits': self.location_head(h), 'time_logits': self.time_head(h)}
