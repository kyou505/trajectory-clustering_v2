"""连续特征掩码预训练、固定验证、插值对照及完整轨迹表示导出。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.continuous import ContinuousDataset, FEATURE_NAMES, collate_continuous
from src.models.continuous import (
    ContinuousMaskedEncoder, masked_mse_per_trajectory, masked_reconstruction_loss,
)


def json_write(path, value):
    """先写临时文件再替换，避免中断时留下不完整的 JSON。"""
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    temporary.replace(path)


def save_checkpoint(path, payload):
    """先写临时文件再替换，避免中断时损坏已有的最佳检查点。"""
    temporary = path.with_name(path.name + '.partial')
    torch.save(payload, temporary)
    temporary.replace(path)


def make_loader(dataset, config, *, shuffle=False, seed=0):
    """构建 DataLoader；使用独立的随机数生成器，使打乱顺序只由 seed 决定。"""
    return DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=shuffle,
        num_workers=config['num_workers'],
        collate_fn=collate_continuous,
        generator=torch.Generator().manual_seed(seed),
        persistent_workers=False,
    )


def move(batch, device):
    """把批次中的所有张量移动到目标设备。"""
    return {key: value.to(device) for key, value in batch.items()}


def reconstruction_pass(model, loader, normalization, device, *, optimizer=None, gradient_clip=1.0):
    """遍历一轮数据：传入 optimizer 时训练，否则验证并逐轨迹对比模型与插值基线。"""
    training = optimizer is not None
    model.train(training)
    rows, loss_sum, count = [], 0.0, 0
    with torch.set_grad_enabled(training):
        for raw in loader:
            batch = move(raw, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(
                batch['features'], batch['time'], batch['valid_mask'], batch['prediction_mask']
            )
            loss = masked_reconstruction_loss(
                output['reconstruction'], batch['features'], batch['prediction_mask']
            )
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite reconstruction loss')
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip, error_if_nonfinite=True
                )
                optimizer.step()
            # 按轨迹数加权累计，使末尾不足一批时的平均损失不失真。
            size = len(raw['index'])
            loss_sum += float(loss.detach()) * size
            count += size
            if not training:
                # 形状为 [轨迹数, 通道数]；模型和基线在相同的遮盖位置上比较。
                mse = masked_mse_per_trajectory(
                    output['reconstruction'], batch['features'], batch['prediction_mask']
                ).cpu().numpy()
                baseline = masked_mse_per_trajectory(
                    batch['interpolation'], batch['features'], batch['prediction_mask']
                ).cpu().numpy()
                for j, index in enumerate(raw['index'].tolist()):
                    row = {'h5_index': index, 'masked_points': int(raw['prediction_mask'][j].sum())}
                    for c, name in enumerate(FEATURE_NAMES):
                        row[f'model_mse_{name}'] = float(mse[j, c])
                        row[f'interpolation_mse_{name}'] = float(baseline[j, c])
                    rows.append(row)
    if not count:
        raise ValueError('Empty reconstruction pass')
    metrics = {'masked_smooth_l1': loss_sum / count, 'trajectories': count}
    table = pd.DataFrame(rows)
    if not training:
        std = np.asarray(normalization['std'])
        for kind in ('model', 'interpolation'):
            # 先逐轨迹求 MSE 再取平均，每条轨迹权重相同；乘 std 换回物理单位。
            per_feature = table[[f'{kind}_mse_{name}' for name in FEATURE_NAMES]].mean().to_numpy()
            metrics[f'{kind}_normalized_mse'] = float(per_feature.mean())
            metrics[f'{kind}_rmse_physical_units'] = dict(
                zip(FEATURE_NAMES, (np.sqrt(per_feature) * std).tolist())
            )
    return metrics, table



def export_embeddings(model, store, indices, normalization, cfg, partitions, output_dir, device):
    """使用完整、未遮盖轨迹；导出原始池化表示，不附加 L2 归一化或簇标签。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = ContinuousDataset(
        store, indices, normalization, cfg['mask'], cfg['seed'], masked=False
    )
    loader = make_loader(dataset, cfg['pretrain'])
    embeddings, identifiers = [], []
    model.eval()
    with torch.no_grad():
        for raw in loader:
            batch = move(raw, device)
            output = model(batch['features'], batch['time'], batch['valid_mask'])
            embeddings.append(output['embedding'].cpu().numpy())
            identifiers.extend(raw['index'].tolist())
    values = np.concatenate(embeddings)
    if not np.isfinite(values).all():
        raise RuntimeError('Non-finite embeddings')
    # 按导出顺序对齐索引表，保证与表示矩阵逐行对应。
    table = partitions.set_index('h5_index').loc[identifiers].reset_index()
    np.savez_compressed(
        output_dir / 'embeddings.npz',
        embeddings=values,
        h5_index=np.asarray(identifiers),
        trajectory_id=table.trajectory_id.to_numpy(dtype=str),
        partition=table.partition.to_numpy(dtype=str),
    )
    table.to_csv(output_dir / 'embedding_index.csv', index=False)
    return {
        'trajectories': len(values),
        'dimensions': values.shape[1],
        'l2_normalized': False,
        'contains_training_and_validation': True,
        'clustering_performed': False,
    }


def train_continuous(
    cfg, store, train_indices, valid_indices, normalization, partitions, run, manifest, device
):
    """执行预训练并早停，再用最佳检查点完成验证评估和表示导出。"""
    train_data = ContinuousDataset(
        store, train_indices, normalization, cfg['mask'], cfg['seed'], training=True
    )
    # 验证掩码使用单独的种子，与训练掩码互不影响。
    valid_data = ContinuousDataset(
        store, valid_indices, normalization, cfg['mask'], cfg['seed'] + 10000
    )
    train_loader = make_loader(train_data, cfg['pretrain'], shuffle=True, seed=cfg['seed'])
    valid_loader = make_loader(valid_data, cfg['pretrain'])
    model = ContinuousMaskedEncoder(**cfg['model']).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['pretrain']['learning_rate'],
        weight_decay=cfg['pretrain']['weight_decay'],
    )
    checkpoints = run / 'checkpoints'
    checkpoints.mkdir()
    best_path = checkpoints / 'pretrain_best.pt'
    best, stale, history = float('inf'), 0, []
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}', flush=True)
    for epoch in range(1, cfg['pretrain']['epochs'] + 1):
        train_data.set_epoch(epoch)
        training, _ = reconstruction_pass(
            model,
            train_loader,
            normalization,
            device,
            optimizer=optimizer,
            gradient_clip=cfg['pretrain']['gradient_clip'],
        )
        validation, _ = reconstruction_pass(model, valid_loader, normalization, device)
        history.append({'epoch': epoch, 'train': training, 'validation': validation})
        print(json.dumps(history[-1]), flush=True)
        json_write(run / 'pretraining_history.json', history)
        # 以验证集标准化 MSE 选择模型并判断早停。
        score = validation['model_normalized_mse']
        if score < best:
            best, stale = score, 0
            save_checkpoint(best_path, {
                'schema': 'ngsim_masked_pretrain_v1',
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': cfg,
                'normalization': normalization,
                'manifest': manifest,
                'validation_metrics': validation,
            })
        else:
            stale += 1
        if stale >= cfg['pretrain']['early_stopping_patience']:
            print(f'Early stopping after epoch {epoch}', flush=True)
            break

    # 后续评估和导出都基于最佳检查点，而不是最后一轮的权重。
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    metrics, table = reconstruction_pass(model, valid_loader, normalization, device)
    table = table.merge(
        partitions[['h5_index', 'trajectory_id', 'position_speed_flag']],
        on='h5_index',
        validate='one_to_one',
    )
    table.to_csv(run / 'validation_reconstruction.csv', index=False)
    # 按位置-速度质量标记分组，查看带标记轨迹的重建误差是否不同。
    metrics['quality_flag_groups'] = {
        str(bool(flag)): {
            'trajectories': len(group),
            'model_normalized_mse': float(
                group[[f'model_mse_{n}' for n in FEATURE_NAMES]].to_numpy().mean()
            ),
        }
        for flag, group in table.groupby('position_speed_flag')
    }
    indices = np.sort(np.concatenate([train_indices, valid_indices]))
    embeddings = export_embeddings(
        model, store, indices, normalization, cfg, partitions, run / 'representations', device
    )
    result = {
        'best_epoch': checkpoint['epoch'],
        'completed_epochs': len(history),
        'validation': metrics,
        'embeddings': embeddings,
        'validation_used_for_model_selection': True,
        'independent_test_evaluation': False,
    }
    json_write(run / 'metrics.json', result)
    return result
