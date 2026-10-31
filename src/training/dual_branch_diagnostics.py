"""保存双分支训练前后的可比表示；标签仅随快照保存用于后续评估。"""

import math

import torch

from src.training.common import json_write, save_checkpoint


def assignment_summary(q):
    """统计软分配矩阵 q（N×K）的整体特征，用于判断聚类是否塌缩。"""
    # 归一化熵：每个样本分配分布的熵除以 log(K)，取值 [0, 1]，越接近 1 越"犹豫"
    entropy = -(q * q.clamp_min(1e-12).log()).sum(1).mean()
    return {
        "normalized_entropy": float(entropy / math.log(q.size(1))),
        # 平均最大概率：反映分配的置信度
        "mean_max_probability": float(q.max(1).values.mean()),
        # 被至少一个样本作为硬分配选中的簇数，过小说明发生簇塌缩
        "occupied_clusters": int(q.argmax(1).unique().numel()),
    }


@torch.no_grad()
def save_dual_representations(model, loader, device, folder, epoch, cfg):
    """遍历全部数据，导出路径/节奏两个分支的池化表示及软分配，并写入快照与摘要。"""
    model.eval()
    collected = {key: [] for key in ("sample_ids", "labels", "path", "rhythm", "q")}

    for batch in loader:
        # 只取编码所需字段搬到设备上，标签与样本 ID 留在 CPU
        view = {
            key: batch[key].to(device)
            for key in ("location_ids", "time_ids", "attention_mask", "pooling_mask")
        }
        parts = model.encode_components(view)
        # 融合两个分支后再经聚类层得到软分配 q
        q = model.clustering_layer(model.fuse(parts["path"], parts["rhythm"]))

        collected["sample_ids"].append(batch["index"].cpu())
        collected["labels"].append(batch["label"].cpu())
        for key in ("path", "rhythm"):
            collected[key].append(parts[key].cpu())
        collected["q"].append(q.cpu())

    # 拼接后按样本 ID 排序，保证不同 epoch 的快照逐行对齐、可直接比较
    state = {key: torch.cat(value) for key, value in collected.items()}
    order = state["sample_ids"].argsort()
    state = {key: value[order] for key, value in state.items()}

    # 完整性校验：样本不能重复，表示与分配中不能出现 NaN/Inf
    if state["sample_ids"].unique().numel() != state["sample_ids"].numel():
        raise ValueError("Duplicate sample IDs in representation snapshot")
    if not all(torch.isfinite(state[key]).all() for key in ("path", "rhythm", "q")):
        raise RuntimeError("Non-finite dual-branch representations")

    # 附带元信息与当前聚类中心；表示保存的是未归一化的分支池化向量
    state.update(
        epoch=epoch,
        configs=cfg,
        representation_kind="unnormalized_branch_pooled",
        # path/rhythm 是兼容字段；mstm_speed 中第一支并非纯空间表示。
        branch_roles=dict(model.branch_roles),
        cluster_centers=model.clustering_layer.cluster_centers.detach().cpu(),
    )

    # 轻量 JSON 摘要：分配统计 + 各分支维度与平均范数，便于快速浏览
    summary = assignment_summary(state["q"])
    summary.update(
        samples=len(order),
        epoch=epoch,
        branch_roles=dict(model.branch_roles),
        path_dimension=state["path"].size(1),
        rhythm_dimension=state["rhythm"].size(1),
        path_mean_norm=float(state["path"].norm(dim=1).mean()),
        rhythm_mean_norm=float(state["rhythm"].norm(dim=1).mean()),
    )

    folder.mkdir(parents=True, exist_ok=True)
    save_checkpoint(folder / f"epoch_{epoch:03d}.pt", state)
    json_write(folder / f"epoch_{epoch:03d}.json", summary)
    return state
