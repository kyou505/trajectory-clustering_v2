"""联合训练：MSTM、单视图 DEC / 跨视图 KL、可选速度模式对比。"""

import json
import time

import torch
from torch.utils.data import DataLoader

from src.data.data_loader import create_contrastive_data_loader, create_single_view_data_loader
from src.data.speed_pattern import batch_views
from src.models.spdtc_loss import SPDTCLoss
from src.models.speed_pattern_loss import speed_pattern_loss
from src.training.dual_branch_diagnostics import assignment_summary, save_dual_representations
from src.training.cluster_init import (
    compute_global_cross_view_targets,
    compute_global_single_view_targets,
    extract_trajectory_embeddings,
    initialize_cluster_centers,
)
from src.training.common import (
    build_trajectory_model,
    clustering_metrics,
    create_optimizer,
    json_write,
    model_kwargs,
    move_to_device,
    save_checkpoint,
)


def cluster_terms(model, batch, criterion, targets):
    """计算 MSTM 与配置指定的聚类损失；目标来自轮初的全量数据。"""
    # 全局目标按稳定样本 ID 索引，不能按当前 batch 顺序或类别编号索引。
    idx = batch["index"].long().cpu()
    device = batch["location_ids"].device
    mstm = model.mstm(batch)
    if criterion.view_mode == 'single':
        view = {key: batch[key] for key in
                ('location_ids', 'time_ids', 'attention_mask', 'pooling_mask')}
        q = model.clustering_layer(model.encode(view))
        return criterion(mstm, batch, {'q': q}, targets['p'][idx].to(device))
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
    view_mode = c.get('view_mode', 'dual')
    speed_cfg = cfg["speed_pattern"]

    # 构建模型并加载预训练的编码器与 MSTM 头
    kwargs = model_kwargs(cfg, dataset, clustering=True)
    model = build_trajectory_model(cfg, dataset).to(device)
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
    save_branches = cfg.get("dual_branch", {}).get("save_representations", False)
    if save_branches:
        save_dual_representations(model, original_loader, device, folder / "representations", 0, cfg)
    del embeddings

    optimizer = create_optimizer(
        model,
        representation_learning_rate=c["representation_learning_rate"],
        clustering_learning_rate=c["clustering_learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    criterion = SPDTCLoss(
        time_loss_weight=cfg["time_loss_weight"],
        clustering_loss_weight=c["loss_weight"],
        view_mode=view_mode,
    )

    # 单视图的目标直接读取原始数据，不额外生成 MSTM 遮盖或增强视图。
    loader_factory = (create_single_view_data_loader if view_mode == 'single'
                      else create_contrastive_data_loader)
    train_loader = loader_factory(
        batch_size=c["batch_size"], seed=cfg["seed"], base_dataset=dataset
    )
    target_loader = original_loader if view_mode == 'single' else DataLoader(
        train_loader.dataset,
        batch_size=c["initialization_batch_size"],
        shuffle=False,
        num_workers=0,
    )

    best, history = float("inf"), []
    for epoch in range(1, c["epochs"] + 1):
        started = time.monotonic()
        if view_mode == 'single':
            targets = compute_global_single_view_targets(model, target_loader, device, len(dataset))
        else:
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

            # 主损失：MSTM + 单视图 DEC 或跨视图 KL
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
            view_mode=view_mode,
        )
        if cfg["group"] == "dual_branch":
            # 此项来自轮初、更新前的全量分配，明确与训练损失区分。
            if view_mode == 'single':
                metrics['epoch_start_assignments_original'] = assignment_summary(targets['q'])
            else:
                metrics['epoch_start_assignments_view1'] = assignment_summary(targets['q1'])
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
    if save_branches:
        snapshot = save_dual_representations(model, original_loader, device,
                                            folder / "representations", c["epochs"], cfg)
        labels.append(snapshot["labels"])
        predictions.append(snapshot["q"].argmax(1))
    else:
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
