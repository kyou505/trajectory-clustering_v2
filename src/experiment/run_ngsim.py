"""US-101 连续特征入口：准备实验资料、只读前向检查或服务器掩码预训练。"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import importlib.metadata
import platform
from pathlib import Path
import random
import sys
import traceback

import numpy as np
import torch
import yaml

from src.data.continuous import (
    ContinuousDataset, ContinuousStore, collate_continuous, fit_normalization, sha256,
    split_trajectories,
)
from src.models.continuous import ContinuousMaskedEncoder, masked_reconstruction_loss
from src.training.train_continuous import json_write, train_continuous


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / 'configs/baseline/spdtc_us101_single_masked_pre30.yaml'


class Tee:
    """同时写入终端和日志文件；每次写入后立即刷新，中断时也能保留已有日志。"""

    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.stream.write(text)
        self.log.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.stream.flush()
        self.log.flush()


def resolve(path):
    """相对路径按项目根目录解析，不依赖当前工作目录。"""
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_config(path):
    """读取 YAML 配置并校验训练和遮盖参数，尽早拒绝非法取值。"""
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg['schema_version'] != 1 or cfg['seed'] < 0:
        raise ValueError('Unsupported schema or invalid seed')
    train = cfg['pretrain']
    for key in ['epochs', 'batch_size', 'early_stopping_patience']:
        if not isinstance(train[key], int) or train[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if not isinstance(train['num_workers'], int) or train['num_workers'] < 0:
        raise ValueError('Invalid num_workers')
    for key in ['learning_rate', 'gradient_clip']:
        if not np.isfinite(train[key]) or train[key] <= 0:
            raise ValueError(f'Invalid {key}')
    if not np.isfinite(train['weight_decay']) or train['weight_decay'] < 0:
        raise ValueError('Invalid weight_decay')
    if (
        not 0 < cfg['mask']['ratio'] < 1
        or not np.isfinite(cfg['mask']['span_seconds'])
        or cfg['mask']['span_seconds'] <= 0
    ):
        raise ValueError('Invalid mask settings')
    return cfg


def forward_check(cfg, store, valid, normalization):
    """真实整轨迹的 CPU 前向检查；不创建优化器，不更新参数。"""
    torch.set_num_threads(2)
    # 选验证集中最短和最长的轨迹，覆盖补齐和最大长度两种边界情况。
    lengths = store.metadata.loc[valid, 'length'].to_numpy()
    selected = np.unique([valid[np.argmin(lengths)], valid[np.argmax(lengths)]])
    data = ContinuousDataset(store, selected, normalization, cfg['mask'], cfg['seed'] + 10000)
    batch = collate_continuous([data[i] for i in range(len(data))])
    model = ContinuousMaskedEncoder(**cfg['model']).eval()
    with torch.no_grad():
        output = model(
            batch['features'], batch['time'], batch['valid_mask'], batch['prediction_mask']
        )
        loss = masked_reconstruction_loss(
            output['reconstruction'], batch['features'], batch['prediction_mask']
        )
    if not torch.isfinite(loss) or not torch.isfinite(output['embedding']).all():
        raise RuntimeError('Forward check produced non-finite values')
    return {
        'passed': True,
        'untrained_forward_only': True,
        'optimizer_steps': 0,
        'trajectory_ids': store.metadata.loc[selected, 'trajectory_id'].tolist(),
        'lengths': batch['length'].tolist(),
        'input_shape': list(batch['features'].shape),
        'reconstruction_shape': list(output['reconstruction'].shape),
        'embedding_shape': list(output['embedding'].shape),
        'masked_points': batch['prediction_mask'].sum(dim=1).tolist(),
        'untrained_loss_for_finiteness_check_only': float(loss),
    }


def run_experiment(cfg, output_dir, *, check_only=False, allow_cpu=False):
    """准备数据并保存运行记录，然后执行前向检查或正式训练；失败时写出 failed.json。"""
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite: {output_dir}')
    if not check_only and not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError(
            'Formal training requires CUDA; '
            'use --check-only for local checks or explicitly --allow-cpu'
        )
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg['seed'])
        torch.backends.cudnn.benchmark = False
    output_dir.mkdir(parents=True)
    with (output_dir / 'train.log').open('w', encoding='utf-8') as log, \
            redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
        try:
            data_path = resolve(cfg['data']['path'])
            store = ContinuousStore(data_path, cfg['data']['speed_source'])
            train, valid, partitions, split_report = split_trajectories(store, cfg['data'])
            normalization = fit_normalization(store, train)
            print(
                f'Train: {len(train):,}; validation: {len(valid):,}; '
                'full trajectories, no truncation',
                flush=True,
            )

            # 训练前先保存划分、标准化参数、配置和版本指纹，便于追溯。
            partitions.to_csv(output_dir / 'split_manifest.csv', index=False)
            json_write(output_dir / 'split_report.json', split_report)
            json_write(output_dir / 'normalization.json', normalization)
            (output_dir / 'config.yaml').write_text(
                yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8'
            )
            manifest = {
                'h5_path': str(data_path),
                'h5_sha256': sha256(data_path),
                'split_sha256': sha256(output_dir / 'split_manifest.csv'),
                'normalization_sha256': sha256(output_dir / 'normalization.json'),
                # 记录 src 下全部 Python 文件的哈希，只用于追溯代码版本。
                'source_sha256': {
                    str(p.relative_to(ROOT)): sha256(p)
                    for p in sorted((ROOT / 'src').rglob('*.py'))
                },
            }
            json_write(output_dir / 'manifest.json', manifest)
            json_write(output_dir / 'environment.json', {
                'python': platform.python_version(),
                'torch': str(torch.__version__),
                'packages': {
                    name: importlib.metadata.version(name)
                    for name in ['numpy', 'pandas', 'h5py', 'PyYAML']
                },
                'cuda_available': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda,
                'device': (
                    'cpu'
                    if check_only or not torch.cuda.is_available()
                    else torch.cuda.get_device_name(0)
                ),
                'check_only': check_only,
                'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            })

            if check_only:
                result = forward_check(cfg, store, valid, normalization)
                json_write(output_dir / 'check.json', result)
            else:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                result = train_continuous(
                    cfg,
                    store,
                    train,
                    valid,
                    normalization,
                    partitions,
                    output_dir,
                    manifest,
                    device,
                )
            json_write(output_dir / 'completed.json', {
                'stage': 'forward_check' if check_only else 'pretrain_and_export',
                'success': True,
            })
            print(result, flush=True)
            print(f'Output: {output_dir}', flush=True)
            return result
        except Exception as error:
            # 先记录失败信息再抛出，让运行目录能反映失败状态。
            json_write(output_dir / 'failed.json', {
                'type': type(error).__name__,
                'message': str(error),
            })
            traceback.print_exc()
            raise


def main():
    """解析命令行参数，确定输出目录并启动实验。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--data', type=Path, help='Override the continuous HDF5 path')
    parser.add_argument(
        '--output-dir', type=Path, help='New directory; existing directories are never overwritten'
    )
    parser.add_argument(
        '--check-only', action='store_true', help='CPU forward check only; no optimizer or training'
    )
    parser.add_argument(
        '--allow-cpu', action='store_true', help='Explicitly allow formal CPU training'
    )
    args = parser.parse_args()
    cfg = load_config(resolve(args.config))
    if args.data:
        cfg['data']['path'] = str(resolve(args.data))
    if args.output_dir:
        output_dir = resolve(args.output_dir)
    else:
        # 默认目录带 UTC 时间戳和种子，每次运行互不覆盖。
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        base = ROOT / 'artifacts/ngsim_checks' if args.check_only else resolve(cfg['output_root'])
        output_dir = base / cfg['name'] / f'{stamp}_seed{cfg["seed"]}'
    run_experiment(cfg, output_dir, check_only=args.check_only, allow_cpu=args.allow_cpu)


if __name__ == '__main__':
    main()
