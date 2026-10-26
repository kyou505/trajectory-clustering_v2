"""联合训练：每轮全局目标更新后，以 MSTM、跨视图 KL、可选速度模式对比更新模型。"""

import json
import time

import torch
from torch.utils.data import DataLoader

from src.data.data_loader import create_contrastive_data_loader
from src.data.speed_pattern import batch_views
from src.models.contrastive_model import SPTrajectoryModel
from src.models.spdtc_loss import SPDTCLoss
from src.models.speed_pattern_loss import speed_pattern_loss
from src.training.cluster_init import (
    compute_global_cross_view_targets,
    extract_trajectory_embeddings,
    initialize_cluster_centers,
)
from src.training.common import (
    clustering_metrics,
    create_optimizer,
    json_write,
    model_kwargs,
    move_to_device,
    save_checkpoint,
)


def cluster_terms(model, batch, criterion, targets):
    """计算单个 batch 的聚类阶段损失（MSTM + 跨视图 KL）。"""
    # 全局目标按稳定样本 ID 索引，不能按当前 batch 顺序或类别编号索引。
    idx = batch["index"].long().cpu()
    device = batch["location_ids"].device
    mstm = model.mstm(batch)
    output = model(batch["view1"], batch["view2"])
    return criterion(
        mstm,
        batch,
        output,
        targets["p1"][idx].to(device),
        targets["p2"][idx].to(device),
    )


def train_spdtc(cfg, run, dataset, device, pretrained):
    """从预训练检查点出发执行 SP-DTC 联合聚类训练，返回最后一轮检查点路径。"""
    torch.manual_seed(cfg["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg["seed"])
    c = cfg["cluster"]
    speed_cfg = cfg["speed_pattern"]

    # 构建模型并加载预训练的编码器与 MSTM 头
    kwargs = model_kwargs(cfg, dataset, clustering=True)
    model = SPTrajectoryModel(**kwargs).to(device)
    info = model.load_pretrained_components(pretrained, map_location=device)
    # 预训练与聚类阶段的速度模式开关必须一致，避免混用不同设置的检查点
    if info["configs"]["speed_pattern"]["enabled"] != speed_cfg["enabled"]:
        raise ValueError("Incompatible pretraining speed-pattern switch")

    # 输出目录：已存在时直接报错，防止覆盖旧实验
    folder = run / "cluster"
    (folder / "checkpoints").mkdir(parents=True)

    # 在原始（未增强）轨迹上提取表示，用 KMeans 初始化聚类中心
    original_loader = DataLoader(
        dataset,
        batch_size=c["initialization_batch_size"],
        shuffle=False,
        num_workers=0,
    )
    embeddings, _ = extract_trajectory_embeddings(model, original_loader, device)
    initialize_cluster_centers(
        model, embeddings, num_clusters=c["num_clusters"], seed=cfg["seed"], n_init=20
    )

    optimizer = create_optimizer(
        model,
        representation_learning_rate=c["representation_learning_rate"],
        clustering_learning_rate=c["clustering_learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    criterion = SPDTCLoss(
        time_loss_weight=cfg["time_loss_weight"],
        clustering_loss_weight=c["loss_weight"],
    )

    # 训练用 loader 打乱顺序；目标计算用 loader 复用同一数据集但不打乱
    train_loader = create_contrastive_data_loader(
        batch_size=c["batch_size"], seed=cfg["seed"], base_dataset=dataset
    )
    target_loader = DataLoader(
        train_loader.dataset,
        batch_size=c["initialization_batch_size"],
        shuffle=False,
        num_workers=0,
    )

    best, history = float("inf"), []
    for epoch in range(1, c["epochs"] + 1):
        started = time.monotonic()
        # 先切换 epoch 使增强视图更新，再用当前模型在全量数据上计算本轮固定的目标分布
        train_loader.dataset.set_epoch(epoch)
        targets = compute_global_cross_view_targets(model, target_loader, device, len(dataset))

        sums, samples = {}, 0
        for batch_i, raw in enumerate(train_loader, start=1):
            views = (
                batch_views(raw, speed_cfg, seed=cfg["seed"], epoch=epoch, stage=3)
                if speed_cfg["enabled"]
                else None
            )
            batch = move_to_device(raw, device)
            model.train()
            optimizer.zero_grad(set_to_none=True)

            # 主损失：MSTM + 跨视图 KL
            terms = cluster_terms(model, batch, criterion, targets)
            # 可选：速度模式三元组损失
            extra = (
                speed_pattern_loss(model, views, speed_cfg["margin"], device)
                if views is not None
                else {}
            )
            if extra:
                terms["loss"] = terms["loss"] + speed_cfg["weight"] * extra["loss"]
                terms.update({f"speed_pattern_{name}": value for name, value in extra.items()})
            if not torch.isfinite(terms["loss"]):
                raise RuntimeError("Non-finite clustering objective")

            terms["loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("Non-finite clustering gradient")
            terms["gradient_norm_before_clip"] = grad_norm
            optimizer.step()

            # 按样本数加权累加各项指标
            size = len(raw["index"])
            samples += size
            for name, value in terms.items():
                sums[name] = sums.get(name, 0.0) + float(value.detach()) * size
            if batch_i % 25 == 0:
                print(
                    f'cluster epoch={epoch} batch={batch_i} loss={float(terms["loss"].detach()):.5f}',
                    flush=True,
                )

        metrics = {key: value / samples for key, value in sums.items()}
        metrics.update(
            samples=samples,
            batches=len(train_loader),
            elapsed_seconds=time.monotonic() - started,
        )
        history.append({"epoch": epoch, "train": metrics})
        json_write(folder / "training_history.json", history)

        # 每轮保存 last；按训练集 base_loss（不含速度模式项）另存 best
        state = {
            "experiment_schema": "sp_dtc_cluster_v1",
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_kwargs": kwargs,
            "configs": cfg,
            "train_metrics": metrics,
            "checkpoint_selection": "last",
        }
        save_checkpoint(folder / "checkpoints/spdtc_last.pt", state)
        if metrics["base_loss"] < best:
            best = metrics["base_loss"]
            save_checkpoint(folder / "checkpoints/spdtc_best.pt", state)
        print("cluster", epoch, json.dumps(metrics), flush=True)

    # 训练结束后只评估一次最终分配；真实标签只在这里用于计算指标。
    model.eval()
    labels, predictions = [], []
    with torch.no_grad():
        for raw in original_loader:
            view = {key: raw[key].to(device) for key in
                    ('location_ids', 'time_ids', 'attention_mask', 'pooling_mask')}
            q = model.clustering_layer(model.encode(view))
            labels.append(raw['label'].cpu())
            predictions.append(q.argmax(1).cpu())
    json_write(
        run / "metrics.json",
        clustering_metrics(torch.cat(labels).numpy(), torch.cat(predictions).numpy()),
    )
    return folder / "checkpoints/spdtc_last.pt"
