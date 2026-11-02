"""NGSIM 连续特征、整轨迹划分、训练集标准化与连续时间段遮盖。"""
import hashlib
import itertools

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


FEATURE_NAMES = ['relative_x_m', 'relative_y_m', 'speed_m_s']


def sha256(path):
    """分块计算文件的 SHA-256，用于记录数据和产物的版本。"""
    value = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


class ContinuousStore:
    """一次读取约几十 MB 的连续数据；不向模型暴露来源编号或车型。"""

    def __init__(self, path, speed_source='raw'):
        if speed_source not in {'raw', 'smoothed'}:
            raise ValueError('speed_source must be raw or smoothed')
        self.path, self.speed_source = str(path), speed_source
        with h5py.File(path, 'r') as data:
            if data.attrs.get('schema') != 'ngsim_continuous_v1':
                raise ValueError('Expected ngsim_continuous_v1 HDF5')
            self.frame_interval_s = float(data.attrs['frame_interval_ms']) / 1000
            self.features = data['features'][:]
            # 选用平滑速度时只替换速度列，位置列保持不变。
            if speed_source == 'smoothed':
                self.features[:, 2] = data['speed_smoothed'][:]
            self.time = data['time'][:].astype(np.float32)
            self.offsets = data['offsets'][:]
            columns = {k: v[:] for k, v in data['tracks'].items() if k != 'trajectory_id'}
            columns['trajectory_id'] = data['tracks/trajectory_id'].asstr()[:]
        self.metadata = pd.DataFrame(columns)
        self.metadata.insert(0, 'h5_index', np.arange(len(self.metadata)))

        # 加载时校验形状、数值和轨迹边界，尽早发现文件损坏或版本不符。
        if self.features.shape != (len(self.time), 3) or not np.isfinite(self.features).all():
            raise ValueError('Invalid continuous features')
        if not np.isfinite(self.time).all() or self.frame_interval_s <= 0:
            raise ValueError('Invalid time values')
        if (
            len(self.offsets) != len(self.metadata) + 1
            or self.offsets[0] != 0
            or self.offsets[-1] != len(self.time)
        ):
            raise ValueError('Invalid offsets')
        if not np.array_equal(np.diff(self.offsets), self.metadata.length):
            raise ValueError('Trajectory lengths do not match offsets')
        # 遮盖时要保留首尾两点，因此每条轨迹至少需要 3 个点。
        if not self.metadata.trajectory_id.is_unique or (self.metadata.length < 3).any():
            raise ValueError('Duplicate trajectory IDs or insufficient length for masking')

    def trajectory(self, index):
        """返回第 index 条轨迹的特征和时间；结果是共享内存的视图，不应原地修改。"""
        start, end = self.offsets[index : index + 2]
        return self.features[start:end], self.time[start:end]


def split_trajectories(store, config):
    """按推导录像批次留出，并隔离可能重复覆盖的时间区间；不拆开整条轨迹。"""
    train_recordings = set(config['train_recordings'])
    valid_recordings = set(config['validation_recordings'])
    if not train_recordings or not valid_recordings or train_recordings & valid_recordings:
        raise ValueError('Training and validation recordings must be nonempty and disjoint')
    meta = store.metadata.copy()
    expected = train_recordings | valid_recordings
    if not expected.issubset(set(meta.recording_id)):
        raise ValueError('Configured recording is absent from HDF5')
    guard = float(config['overlap_guard_seconds']) * 1000  # 换算为毫秒，与时间戳单位一致
    if not np.isfinite(guard) or guard < 0:
        raise ValueError('overlap_guard_seconds must be nonnegative')

    # 找出所选批次之间两两重叠的时钟区间。
    bounds = meta[meta.recording_id.isin(expected)].groupby('recording_id').agg(
        start=('start_time_ms', 'min'), end=('end_time_ms', 'max')
    )
    intervals = []
    for a, b in itertools.combinations(bounds.index, 2):
        start = max(bounds.loc[a, 'start'], bounds.loc[b, 'start'])
        end = min(bounds.loc[a, 'end'], bounds.loc[b, 'end'])
        if start <= end:
            intervals.append({
                'recordings': [int(a), int(b)],
                'start_ms': int(start),
                'end_ms': int(end),
                'guard_seconds': guard / 1000,
            })
    # 轨迹时间段与任一重叠区间（含前后保护时长）相交即标记。
    overlap = np.zeros(len(meta), dtype=bool)
    for interval in intervals:
        overlap |= (meta.start_time_ms <= interval['end_ms'] + guard) & (
            meta.end_time_ms >= interval['start_ms'] - guard
        )

    # 依次应用筛选条件，每条轨迹的 partition 记录其最先命中的排除原因。
    eligible = meta.vehicle_class.eq(config['vehicle_class']) & meta.recording_id.isin(expected)
    meta['clock_overlap_flag'] = overlap
    meta['position_speed_flag'] = meta.position_speed_mismatch_pairs.gt(0)
    meta['partition'] = 'excluded_scope'
    if config['exclude_position_speed_mismatch']:
        meta.loc[eligible & meta.position_speed_flag, 'partition'] = 'excluded_position_speed_flag'
        eligible &= ~meta.position_speed_flag
    meta.loc[eligible & overlap, 'partition'] = 'excluded_clock_overlap'
    eligible &= ~overlap
    meta.loc[eligible & meta.recording_id.isin(train_recordings), 'partition'] = 'train'
    meta.loc[eligible & meta.recording_id.isin(valid_recordings), 'partition'] = 'validation'
    train = meta.loc[meta.partition.eq('train'), 'h5_index'].to_numpy()
    valid = meta.loc[meta.partition.eq('validation'), 'h5_index'].to_numpy()
    if not len(train) or not len(valid):
        raise ValueError('Empty training or validation partition after selection')
    report = {
        'partition_counts': {k: int(v) for k, v in meta.partition.value_counts().items()},
        'flagged_retained_trajectories': int(meta.loc[eligible, 'position_speed_flag'].sum()),
        'clock_overlap_intervals': intervals,
        'recording_id_is_inferred': True,
        'driver_independence_verified': False,
        'per_recording': meta.groupby(['recording_id', 'partition'])
        .size()
        .rename('count')
        .reset_index()
        .to_dict('records'),
    }
    return train, valid, meta, report


def fit_normalization(store, train_indices):
    """只按训练集有效点估计均值/标准差，不逐轨迹缩放速度或归一化时长。"""
    total = np.zeros(3, dtype=np.float64)
    squared = np.zeros(3, dtype=np.float64)
    count = 0
    for index in train_indices:
        values = store.trajectory(index)[0].astype(np.float64)
        total += values.sum(axis=0)
        squared += (values * values).sum(axis=0)
        count += len(values)
    if not count:
        raise ValueError('Cannot fit normalization on empty training data')
    mean = total / count
    # 浮点误差可能使方差略小于 0，先截断；近似常量的通道不缩放，避免除零。
    std = np.sqrt(np.maximum(squared / count - mean * mean, 0))
    std = np.where(std < 1e-6, 1.0, std)
    return {
        'feature_names': FEATURE_NAMES,
        'mean': mean.tolist(),
        'std': std.tolist(),
        'training_points': count,
        'training_trajectories': len(train_indices),
        'weighting': 'one vote per valid training point',
        'time_normalized': False,
    }


def span_mask(length, ratio, span_points, rng):
    """三个物理通道同时遮挡；保留首尾，允许插值基线使用相同的可见上下文。"""
    if length < 3 or not 0 < ratio < 1 or span_points < 1:
        raise ValueError('Invalid mask settings')
    target = min(length - 2, max(1, round(length * ratio)))
    mask = np.zeros(length, dtype=bool)
    # 反复放置时间段直到达到目标点数；新时间段可与已遮盖部分相接或重叠。
    while mask.sum() < target:
        candidates = np.flatnonzero(~mask[1:-1]) + 1
        start = int(rng.choice(candidates))
        size = min(span_points, target - int(mask.sum()), length - 1 - start)
        mask[start : start + size] = True
    return mask


class ContinuousDataset(Dataset):
    """按需标准化轨迹并生成遮盖；验证集同时提供线性插值基线。"""

    def __init__(
        self, store, indices, normalization, mask_config, seed, *, training=False, masked=True
    ):
        self.store, self.indices = store, np.asarray(indices, dtype=np.int64)
        self.mean = np.asarray(normalization['mean'], dtype=np.float32)
        self.std = np.asarray(normalization['std'], dtype=np.float32)
        self.mask_config, self.seed = mask_config, int(seed)
        self.training, self.masked, self.epoch = training, masked, 0
        self.span_points = max(1, round(mask_config['span_seconds'] / store.frame_interval_s))

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        """设置当前 epoch，使训练掩码逐轮变化。"""
        self.epoch = int(epoch)

    def __getitem__(self, position):
        index = int(self.indices[position])
        raw, time = self.store.trajectory(index)
        values = (raw - self.mean) / self.std
        mask = np.zeros(len(values), dtype=bool)
        if self.masked:
            # 验证掩码固定；训练掩码由 epoch 和轨迹索引决定，不依赖 worker 调度。
            rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, self.epoch if self.training else 0, index])
            )
            mask = span_mask(len(values), self.mask_config['ratio'], self.span_points, rng)
        baseline = values.copy()
        # 只有验证时才需要基线；用可见点按真实时间逐通道线性插值。
        if self.masked and not self.training:
            visible = ~mask
            for column in range(3):
                baseline[mask, column] = np.interp(
                    time[mask], time[visible], values[visible, column]
                )
        return {
            'index': index,
            'features': torch.from_numpy(values),
            # time 是共享内存的视图，复制后再交给 torch，避免张量与原数组共用内存。
            'time': torch.from_numpy(time.copy()),
            'prediction_mask': torch.from_numpy(mask),
            'interpolation': torch.from_numpy(baseline),
        }


def collate_continuous(items):
    """补齐值与合法零值可相同，但有效位置只由真实长度决定。"""
    lengths = torch.tensor([len(item['time']) for item in items], dtype=torch.long)
    batch, width = len(items), int(lengths.max())
    output = {
        'index': torch.tensor([item['index'] for item in items]),
        'length': lengths,
        'features': torch.zeros(batch, width, 3),
        'time': torch.zeros(batch, width),
        'prediction_mask': torch.zeros(batch, width, dtype=torch.bool),
        'valid_mask': torch.arange(width)[None, :] < lengths[:, None],
        'interpolation': torch.zeros(batch, width, 3),
    }
    for i, item in enumerate(items):
        length = int(lengths[i])
        for key in ['features', 'time', 'prediction_mask', 'interpolation']:
            output[key][i, :length] = item[key]
    return output
