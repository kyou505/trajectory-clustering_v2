"""两个训练阶段共用的设备、优化器、检查点和日志工具。"""
from pathlib import Path
import json

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, rand_score, adjusted_rand_score


def clustering_metrics(labels, predictions):
    """最终聚类指标；只用于评估，不参与梯度或检查点选择。"""
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    true_values, true_ids = np.unique(labels, return_inverse=True)
    predicted_values, predicted_ids = np.unique(predictions, return_inverse=True)
    contingency = np.zeros((len(predicted_values), len(true_values)), dtype=np.int64)
    np.add.at(contingency, (predicted_ids, true_ids), 1)
    rows, columns = linear_sum_assignment(-contingency)
    return {
        'uacc': float(contingency[rows, columns].sum() / labels.size),
        'nmi': float(normalized_mutual_info_score(labels, predictions, average_method='geometric')),
        'ri': float(rand_score(labels, predictions)),
        'ari': float(adjusted_rand_score(labels, predictions)),
    }


def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def json_write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path, state):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def model_kwargs(cfg, dataset, clustering=False):
    kwargs = dict(cfg["model"])
    kwargs.update(location_vocab_size=dataset.location_vocab_size,
                  time_vocab_size=dataset.time_vocab_size,
                  max_length=dataset.max_length)
    if cfg["group"] == "dual_branch":
        kwargs.update({key: cfg["dual_branch"][key] for key in ("path_weight", "distance_scale")})
        routing = cfg['dual_branch'].get('task_routing', 'split_masked')
        if routing != 'split_masked':
            kwargs['task_routing'] = routing
    if clustering:
        kwargs["num_clusters"] = cfg["cluster"]["num_clusters"]
    return kwargs


def build_pretrain_model(cfg, dataset):
    if cfg["group"] == "dual_branch":
        from src.models.dual_branch import DualBranchPretrainModel
        return DualBranchPretrainModel(**model_kwargs(cfg, dataset))
    from src.models.pretrain_model import SPPretrainModel
    return SPPretrainModel(**model_kwargs(cfg, dataset))


def build_trajectory_model(cfg, dataset):
    if cfg["group"] == "dual_branch":
        from src.models.dual_branch import DualBranchTrajectoryModel
        return DualBranchTrajectoryModel(**model_kwargs(cfg, dataset, clustering=True))
    from src.models.contrastive_model import SPTrajectoryModel
    return SPTrajectoryModel(**model_kwargs(cfg, dataset, clustering=True))


def masked_correct_and_total(logits, targets):
    valid = targets.ne(-100)
    return int((logits.argmax(-1)[valid] == targets[valid]).sum()), int(valid.sum())


def create_optimizer(model, representation_learning_rate, clustering_learning_rate, weight_decay):
    # 编码器和预测头归入表示参数；簇中心采用单独学习率。
    representation = [param for name, param in model.named_parameters()
                      if param.requires_grad and not name.startswith("clustering_layer.")]
    centers = list(model.clustering_layer.parameters())
    if {id(p) for p in representation} & {id(p) for p in centers}:
        raise RuntimeError("Optimizer groups overlap")
    return torch.optim.AdamW([
        {"params": representation, "lr": representation_learning_rate},
        {"params": centers, "lr": clustering_learning_rate},
    ], weight_decay=weight_decay)
