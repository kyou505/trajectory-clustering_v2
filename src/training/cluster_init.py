"""全量原始表示初始化中心；每轮按配置计算原始或双视图的冻结锐化目标。"""
import torch
from sklearn.cluster import KMeans

from src.models.dec import target_distribution

@torch.no_grad()
def extract_trajectory_embeddings(
        model,
        loader,
        device,
        max_batches=None,
):
    model.eval()
    all_embeddings = []
    all_labels = []

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        view = {
            "location_ids": batch["location_ids"].to(device),
            "time_ids": batch["time_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "pooling_mask": batch["pooling_mask"].to(device),
        }
        embedding = model.encode(view)
        all_embeddings.append(embedding.cpu())
        if "label" in batch:
            all_labels.append(batch["label"].cpu())

    embedding = torch.cat(all_embeddings, dim=0)
    labels = None
    if all_labels:
        labels = torch.cat(all_labels, dim=0)

    return embedding, labels


@torch.no_grad()
def initialize_cluster_centers(
        model,
        embeddings,
        num_clusters=12,
        seed=42,
        n_init=20
):
    expected_shape = (num_clusters, embeddings.size(1))
    actual_shape = tuple(model.clustering_layer.cluster_centers.shape)
    if actual_shape != expected_shape:
        raise ValueError(f"Cluster centers shape {actual_shape} != {expected_shape}")
    kmeans = KMeans(
        n_clusters=num_clusters,
        n_init=n_init,
        random_state=seed,
    )
    kmeans.fit(embeddings.detach().cpu().numpy())
    centers = torch.from_numpy(kmeans.cluster_centers_).to(
        device=model.clustering_layer.cluster_centers.device,
        dtype = model.clustering_layer.cluster_centers.dtype
    )
    # 不能重新赋值
    model.clustering_layer.cluster_centers.copy_(centers)
    return kmeans

@torch.no_grad()
def compute_global_single_view_targets(model, loader, device, num_samples):
    """按样本 ID 汇集原始轨迹 Q，再以全量簇频率计算本轮固定的 P。"""
    model.eval()
    global_q = None
    seen = torch.zeros(num_samples, dtype=torch.bool)
    for batch in loader:
        indices = batch['index'].long().cpu()
        if indices.min().item() < 0 or indices.max().item() >= num_samples:
            raise ValueError('indices should be in [0, num_samples)')
        if indices.unique().numel() != indices.numel() or seen[indices].any():
            raise RuntimeError('indices should be unique')
        view = {key: batch[key].to(device) for key in
                ('location_ids', 'time_ids', 'attention_mask', 'pooling_mask')}
        q = model.clustering_layer(model.encode(view)).cpu()
        if global_q is None:
            global_q = torch.empty(num_samples, q.size(1), dtype=q.dtype)
        global_q[indices] = q
        seen[indices] = True
    if global_q is None:
        raise RuntimeError('no batches were processed')
    if not seen.all():
        raise RuntimeError(f'Missing {(~seen).sum().item()} samples')
    return {'q': global_q, 'p': target_distribution(global_q)}


@torch.no_grad()
def compute_global_cross_view_targets(
        model,
        loader,
        device,
        num_samples,
):
    model.eval()
    global_q1 = None
    global_q2 = None
    seen = torch.zeros(num_samples, dtype=torch.bool)
    for batch in loader:
        indices = batch["index"].long()
        view1 = {
            key: value.to(device)
            for key, value in batch["view1"].items()
            if key in {
                "location_ids",
                "time_ids",
                "attention_mask",
                "pooling_mask",
            }
        }
        view2 = {
            key: value.to(device)
            for key, value in batch["view2"].items()
            if key in {
                "location_ids",
                "time_ids",
                "attention_mask",
                "pooling_mask",
            }
        }
        z1 = model.encode(view1)
        z2 = model.encode(view2)
        q1 = model.clustering_layer(z1).cpu()
        q2 = model.clustering_layer(z2).cpu()
        if global_q1 is None:
            num_clusters = q1.size(1)
            global_q1 = torch.empty(num_samples, num_clusters, dtype=q1.dtype)
            global_q2 = torch.empty(num_samples, num_clusters, dtype=q2.dtype)
        if indices.min().item() < 0:
            raise ValueError("indices should be >= 0")
        if indices.max().item() >= num_samples:
            raise ValueError("indices should be < num_samples")
        if indices.unique().numel() != indices.numel() or seen[indices].any():
            raise RuntimeError("indices should be unique")
        global_q1[indices] = q1
        global_q2[indices] = q2
        seen[indices] = True

    if global_q1 is None:
        raise RuntimeError("no batches were processed")
    if not seen.all():
        missing_count = (~seen).sum().item()
        raise RuntimeError(f"Missing {missing_count} samples")

    global_p1 = target_distribution(global_q1)
    global_p2 = target_distribution(global_q2)
    return {
        "q1": global_q1,
        "q2": global_q2,
        "p1": global_p1,
        "p2": global_p2,
    }
