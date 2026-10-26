"""预训练：MSTM，可选速度模式对比；固定验证视图用于选择检查点。"""

import json

import torch

from src.data.data_loader import create_data_loaders
from src.data.speed_pattern import batch_views
from src.models.pretrain_model import MSTMLoss, SPPretrainModel
from src.models.speed_pattern_loss import speed_pattern_loss
from src.training.common import (
    json_write,
    masked_correct_and_total,
    model_kwargs,
    move_to_device,
    save_checkpoint,
)


def pretrain_pass(model, loader, criterion, cfg, device, *, epoch, optimizer=None):
    """跑一遍数据：传入 optimizer 时为训练，否则为评估。

    返回按 token 加权的 MSTM 损失/准确率，以及按样本加权的速度模式指标。
    """
    training = optimizer is not None
    model.train(training)
    speed_cfg = cfg["speed_pattern"]

    # 累加器：MSTM 损失与准确率按被遮盖 token 数加权，速度模式指标按样本数加权
    sums = {"location_loss": 0.0, "time_loss": 0.0}
    correct = {"location": 0, "time": 0}
    totals = {"location": 0, "time": 0}
    extra_sums, samples, batches = {}, 0, 0

    with torch.set_grad_enabled(training):
        for raw in loader:
            # 速度模式视图：训练时随 epoch 变化；验证时固定 epoch=0、stage=2，
            # 保证每轮验证使用相同视图，便于比较和选择检查点
            views = (
                batch_views(
                    raw,
                    speed_cfg,
                    seed=cfg["seed"],
                    epoch=epoch if training else 0,
                    stage=1 if training else 2,
                )
                if speed_cfg["enabled"]
                else None
            )
            batch = move_to_device(raw, device)
            if training:
                optimizer.zero_grad(set_to_none=True)

            # MSTM 掩码预测损失
            output = model.mstm(batch)
            terms = criterion(
                output["location_logits"],
                output["time_logits"],
                batch["location_targets"],
                batch["time_targets"],
            )
            objective = terms["loss"]

            # 可选：速度模式三元组损失
            extra = (
                speed_pattern_loss(model, views, speed_cfg["margin"], device)
                if views is not None
                else {}
            )
            if extra:
                objective = objective + speed_cfg["weight"] * extra["loss"]
            if not torch.isfinite(objective):
                raise RuntimeError("Non-finite pretraining objective")

            if training:
                objective.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # 统计指标
            size = len(raw["index"])
            for kind in ("location", "time"):
                hits, total = masked_correct_and_total(
                    output[f"{kind}_logits"], batch[f"{kind}_targets"]
                )
                correct[kind] += hits
                totals[kind] += total
                sums[f"{kind}_loss"] += float(terms[f"{kind}_loss"].detach()) * total
            for name, value in extra.items():
                extra_sums[name] = extra_sums.get(name, 0.0) + float(value.detach()) * size
            samples += size
            batches += 1

    if not samples or not all(totals.values()):
        raise RuntimeError("Empty pretraining pass")

    # 汇总：先求各项平均，再按与训练相同的权重组合
    result = {f"{k}_loss": sums[f"{k}_loss"] / totals[k] for k in totals}
    result["loss"] = result["location_loss"] + cfg["time_loss_weight"] * result["time_loss"]
    result.update({f"{k}_accuracy": correct[k] / totals[k] for k in totals})
    result.update({f"speed_pattern_{key}": value / samples for key, value in extra_sums.items()})
    result["total_objective"] = (
        result["loss"] + speed_cfg["weight"] * result.get("speed_pattern_loss", 0.0)
    )
    result.update(samples=samples, batches=batches)
    return result


def pretrain(cfg, run, dataset, device):
    """执行完整预训练，按验证集 MSTM 损失保存最优检查点，返回其路径。"""
    torch.manual_seed(cfg["seed"])
    train_loader, valid_loader, _ = create_data_loaders(
        batch_size=cfg["pretrain"]["batch_size"],
        seed=cfg["seed"],
        base_dataset=dataset,
    )

    kwargs = model_kwargs(cfg, dataset)
    model = SPPretrainModel(**kwargs).to(device)
    criterion = MSTMLoss(time_loss_weight=cfg["time_loss_weight"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["pretrain"]["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    checkpoint = run / "pretrain_best.pt"
    history, best = [], float("inf")
    for epoch in range(1, cfg["pretrain"]["epochs"] + 1):
        train = pretrain_pass(
            model, train_loader, criterion, cfg, device, epoch=epoch, optimizer=optimizer
        )
        valid = pretrain_pass(model, valid_loader, criterion, cfg, device, epoch=epoch)
        history.append({"epoch": epoch, "train": train, "valid": valid})
        print("pretrain", epoch, json.dumps({"train": train, "valid": valid}), flush=True)

        # 各组均按验证集 MSTM 选择预训练检查点，不使用真实标签或辅助损失排名。
        if valid["loss"] < best:
            best = valid["loss"]
            save_checkpoint(
                checkpoint,
                {
                    "experiment_schema": "sp_dtc_pretrain_v1",
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "valid_metrics": valid,
                    "configs": cfg,
                    "model_kwargs": kwargs,
                },
            )
        # 每轮都写出历史，训练中断也能保留已有记录
        json_write(run / "pretraining_history.json", history)

    # 释放显存，避免影响后续聚类阶段
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return checkpoint
