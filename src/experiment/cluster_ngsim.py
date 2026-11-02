"""对完整 US-101 轨迹表示和运动统计量分别做 HDBSCAN；不训练，不指定簇数。"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import yaml

from src.data.continuous import sha256
from src.training.train_continuous import json_write


ROOT = Path(__file__).resolve().parents[2]
# 每条轨迹的运动统计量，顺序需与 load_inputs 中逐行计算的取值一致。
STAT_FEATURES = [
    'duration_s',
    'longitudinal_distance_m',
    'mean_speed_m_s',
    'std_speed_m_s',
    'net_speed_change_m_s',
    'low_speed_fraction',
    'lateral_end_m',
    'lateral_range_m',
]


def resolve(path):
    """相对路径按项目根目录解析，不依赖当前工作目录。"""
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_inputs(run, data_path):
    """核对数据版本与逐行索引，保持原训练划分，不重新筛选轨迹。"""
    manifest = json.loads((run / 'manifest.json').read_text())
    if sha256(data_path) != manifest['h5_sha256']:
        raise ValueError('HDF5 differs from the pretraining input')
    cfg = yaml.safe_load((run / 'config.yaml').read_text())
    with np.load(run / 'representations/embeddings.npz', allow_pickle=False) as packed:
        embeddings = packed['embeddings'].astype(np.float64)
        indices, ids = packed['h5_index'], packed['trajectory_id']
        partitions = packed['partition']

    # 表示文件与索引表必须逐行对齐，否则簇标签会对应到错误的轨迹。
    index = pd.read_csv(run / 'representations/embedding_index.csv')
    for key, values in [('h5_index', indices), ('trajectory_id', ids), ('partition', partitions)]:
        if not np.array_equal(index[key].to_numpy(), values):
            raise ValueError(f'Embedding index mismatch: {key}')
    if embeddings.ndim != 2 or len(embeddings) != len(index) or not np.isfinite(embeddings).all():
        raise ValueError('Invalid embeddings')
    if not index.trajectory_id.is_unique or not index.partition.isin(['train', 'validation']).all():
        raise ValueError('Invalid trajectory IDs or partitions')

    with h5py.File(data_path, 'r') as data:
        if not np.array_equal(data['tracks/trajectory_id'].asstr()[:][indices], ids):
            raise ValueError('HDF5 trajectory IDs do not match embeddings')
        values, times, offsets = data['features'][:], data['time'][:], data['offsets'][:]
        # 速度来源与预训练配置保持一致。
        if cfg['data']['speed_source'] == 'smoothed':
            values[:, 2] = data['speed_smoothed'][:]

    rows = []
    for i in indices:
        # 所有轨迹点连续存放，offsets 给出第 i 条轨迹的起止位置。
        left, right = offsets[i:i + 2]
        x, y, speed = values[left:right].astype(np.float64).T
        time = times[left:right]
        if len(time) < 2 or not (np.diff(time) > 0).all():
            raise ValueError('Invalid trajectory time')
        # 起止均速取首尾各 1 秒；这些量只作统计对照，不作为行为标签。
        net_speed = speed[time >= time[-1] - 1].mean() - speed[time <= time[0] + 1].mean()
        rows.append([
            time[-1] - time[0],
            y[-1] - y[0],
            speed.mean(),
            speed.std(),
            net_speed,
            (speed < 0.5).mean(),
            x[-1] - x[0],
            np.ptp(x),
        ])
    descriptors = pd.DataFrame(rows, columns=STAT_FEATURES)
    if not np.isfinite(descriptors.to_numpy()).all():
        raise ValueError('Invalid motion statistics')
    metadata = index[['h5_index', 'trajectory_id', 'partition', 'recording_id']]
    return embeddings, pd.concat([metadata, descriptors], axis=1)


def prepare_spaces(embeddings, descriptors, output):
    """沿用已有实验：统计量标准化，PCA 保留 99% 方差，所有变换只拟合训练集。"""
    train = descriptors.partition.eq('train').to_numpy()
    if train.sum() < 2:
        raise ValueError('At least two training trajectories are required for PCA')
    scaler = StandardScaler().fit(descriptors.loc[train, STAT_FEATURES])
    statistics = scaler.transform(descriptors[STAT_FEATURES])
    spaces, summary = {}, {}
    arrays = {'statistics_mean': scaler.mean_, 'statistics_scale': scaler.scale_}
    for name, values in [('embedding', embeddings), ('statistics', statistics)]:
        pca = PCA(n_components=0.99, svd_solver='full', whiten=False).fit(values[train])
        # 显式点积与原分析保持一致；编码器表示不做额外标准化或 L2 归一化。
        spaces[name] = np.einsum('nd,kd->nk', values - pca.mean_, pca.components_, optimize=False)
        summary[name] = {
            'input_dimensions': values.shape[1],
            'pca_dimensions': int(pca.n_components_),
            'retained_train_variance': float(pca.explained_variance_ratio_.sum()),
            'fit_partition': 'train',
            'whiten': False,
        }
        arrays.update({
            f'{name}_pca_mean': pca.mean_,
            f'{name}_pca_components': pca.components_,
            f'{name}_coordinates': spaces[name],
        })
    # 保存变换参数和降维坐标，便于复现或对新轨迹做相同变换。
    np.savez_compressed(output / 'feature_spaces.npz', **arrays)
    return spaces, summary


def save_assignments(name, model, descriptors, output):
    """保存轨迹 ID、簇号和原单位统计量；-1 为未入簇样本，不命名为行为类别。"""
    assigned = descriptors.assign(cluster=model.labels_, membership_strength=model.probabilities_)
    assigned.to_csv(output / f'{name}_assignments.csv', index=False)

    # 每个簇（含 -1）记录规模和各统计量的四分位数，用于事后查看簇的运动特征。
    rows = []
    for label, group in assigned.groupby('cluster', sort=True):
        row = {'cluster': int(label), 'count': len(group), 'fraction': len(group) / len(assigned)}
        for feature in STAT_FEATURES:
            for suffix, quantile in [('q25', 0.25), ('median', 0.5), ('q75', 0.75)]:
                row[f'{feature}_{suffix}'] = float(group[feature].quantile(quantile))
        rows.append(row)
    pd.DataFrame(rows).to_csv(output / f'{name}_cluster_summary.csv', index=False)

    labels, counts = np.unique(model.labels_[model.labels_ >= 0], return_counts=True)
    return {
        'clusters': len(labels),
        'cluster_sizes': dict(zip(map(str, labels), map(int, counts))),
        'noise_count': int((model.labels_ < 0).sum()),
        'noise_fraction': float((model.labels_ < 0).mean()),
    }


def run_clustering(run, data, output, min_cluster_size=50, min_samples=10):
    """在表示和统计量两个空间上用相同参数聚类，并写出结果摘要与输入文件哈希。"""
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    if min_cluster_size < 2 or min_samples < 1:
        raise ValueError('min_cluster_size must be >= 2 and min_samples must be >= 1')
    embeddings, descriptors = load_inputs(run, data)
    if min_samples > len(embeddings):
        raise ValueError('min_samples exceeds the number of trajectories')
    output.mkdir(parents=True)
    spaces, preprocessing = prepare_spaces(embeddings, descriptors, output)
    parameters = {
        'min_cluster_size': min_cluster_size,
        'min_samples': min_samples,
        'metric': 'euclidean',
        'cluster_selection_method': 'eom',
        'allow_single_cluster': False,
    }
    results = {}
    for name, values in spaces.items():
        model = HDBSCAN(**parameters, n_jobs=2, copy=True).fit(values)
        results[name] = save_assignments(name, model, descriptors, output)
        print(f'{name}: {results[name]}', flush=True)
    summary = {
        'trajectories': len(embeddings),
        'hdbscan': parameters,
        'preprocessing': preprocessing,
        'statistics_features': STAT_FEATURES,
        'low_speed_threshold_m_s': 0.5,
        'net_speed_change_endpoint_window_s': 1.0,
        'results': results,
        'all_exported_train_and_validation_used': True,
        'independent_test_evaluation': False,
    }
    json_write(output / 'summary.json', summary)

    # 记录输入文件和本脚本的哈希，便于追溯结果来源。
    paths = [
        data,
        run / 'config.yaml',
        run / 'manifest.json',
        run / 'representations/embeddings.npz',
        run / 'representations/embedding_index.csv',
        Path(__file__),
    ]
    json_write(output / 'manifest.json', {
        'source_run': str(run),
        'source_data': str(data),
        'sklearn': sklearn.__version__,
        'numpy': np.__version__,
        'sha256': {str(p): sha256(p) for p in paths},
    })
    print(f'Output: {output}', flush=True)
    return summary


def main():
    """解析命令行参数，确定输出目录并启动聚类。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--run', type=Path, required=True, help='Completed pretraining run directory'
    )
    parser.add_argument(
        '--data', type=Path, default=Path('data/NGSIM/us101_processed/trajectories.h5')
    )
    parser.add_argument(
        '--output-dir', type=Path, help='New output directory; never overwrites results'
    )
    parser.add_argument('--min-cluster-size', type=int, default=50)
    parser.add_argument('--min-samples', type=int, default=10)
    args = parser.parse_args()
    run = resolve(args.run)
    # 默认目录带 UTC 时间戳，每次运行互不覆盖。
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = (
        resolve(args.output_dir)
        if args.output_dir
        else ROOT / 'artifacts/ngsim_clustering' / run.name / stamp
    )
    # 限制底层线性代数库的线程数，与 HDBSCAN 的 n_jobs=2 保持一致。
    with threadpool_limits(limits=2):
        run_clustering(run, resolve(args.data), output, args.min_cluster_size, args.min_samples)


if __name__ == '__main__':
    main()
