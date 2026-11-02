"""将 US-101 CSV 整理为整条轨迹的连续数值 HDF5，不生成词表或行为标签。"""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


ROOT = Path(__file__).resolve().parents[3]
FT_TO_M = 0.3048  # 原始数据的距离和速度单位均为英尺
FRAME_MS = 100  # 10 Hz 采样，相邻帧相隔 100 ms
REQUIRED_COLUMNS = {
    'Vehicle_ID', 'Frame_ID', 'Total_Frames', 'Global_Time', 'Local_X',
    'Local_Y', 'v_Vel', 'v_Acc', 'v_Class', 'Lane_ID', 'Location',
}


def file_hash(path):
    """分块计算 SHA-256，避免把数 GB 的 CSV 一次读入内存。"""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path, value):
    """写出 UTF-8 JSON；allow_nan=False 使 NaN/Inf 直接报错，而不是写成非标准 JSON。"""
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
        encoding='utf-8',
    )


def prepare_tracks(frame, smoothing_points=11, mismatch_threshold=5.0):
    """结构错误整轨隔离；运动学异常只标记，避免任意删点或拼接不连续轨迹。"""
    if not REQUIRED_COLUMNS.issubset(frame.columns):
        raise ValueError(f'Missing columns: {sorted(REQUIRED_COLUMNS - set(frame.columns))}')
    if frame.empty or not frame.Location.eq('us-101').all():
        raise ValueError('Input must contain only nonempty US-101 data')
    for column in ['Vehicle_ID', 'Frame_ID', 'Global_Time']:
        values = frame[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f'Invalid integer trajectory key: {column}')

    raw_rows = len(frame)
    # 必须比较所有原始字段，不能先丢弃字段后把冲突记录误当重复行。
    frame = frame.drop_duplicates().copy()
    duplicates = raw_rows - len(frame)
    # 同一批次内该差值恒定，可据此推导批次；车辆 ID 只有与批次组合才能唯一标识轨迹。
    frame['recording_id'] = frame.Global_Time - FRAME_MS * frame.Frame_ID
    frame = frame.sort_values(['recording_id', 'Vehicle_ID', 'Global_Time', 'Frame_ID'])
    accepted, rejected, indices, smooth_parts, recording_reports = [], [], [], [], []
    offset = 0
    for recording_id, recording in frame.groupby('recording_id', sort=True):
        recording_reports.append({
            'recording_id': int(recording_id),
            'unique_rows': len(recording),
            'vehicle_tracks': int(recording.Vehicle_ID.nunique()),
            'start_ms': int(recording.Global_Time.min()),
            'end_ms': int(recording.Global_Time.max()),
        })
    for (recording_id, vehicle_id), part in frame.groupby(
        ['recording_id', 'Vehicle_ID'], sort=True
    ):
        trajectory_id = f'us101:{int(recording_id)}:{int(vehicle_id)}'
        n = len(part)
        timestamp = part.Global_Time.to_numpy(dtype=np.int64)
        frames = part.Frame_ID.to_numpy(dtype=np.int64)
        xyz = part[['Local_X', 'Local_Y', 'v_Vel']].to_numpy(dtype=np.float64) * FT_TO_M

        # 结构检查：收集全部不合格原因后再整轨隔离，便于在报告中追溯。
        reasons = []
        if part.Global_Time.duplicated().any():
            reasons.append('conflicting_same_vehicle_time')
        if n < 2:
            reasons.append('fewer_than_two_points')
        if not (np.diff(timestamp) == FRAME_MS).all() or not (np.diff(frames) == 1).all():
            reasons.append('noncontinuous_time_or_frame')
        if (
            part.Total_Frames.isna().any()
            or part.Total_Frames.nunique() != 1
            or part.Total_Frames.iloc[0] != n
        ):
            reasons.append('declared_frame_count_mismatch')
        if (
            part.v_Class.isna().any()
            or part.v_Class.nunique() != 1
            or part.v_Class.iloc[0] not in (1, 2, 3)
        ):
            reasons.append('invalid_or_inconsistent_vehicle_class')
        if not np.isfinite(xyz).all():
            reasons.append('nonfinite_input_features')
        if np.any(xyz[:, 2] < 0):
            reasons.append('negative_recorded_speed')
        if part.Lane_ID.isna().any() or not np.equal(part.Lane_ID, np.floor(part.Lane_ID)).all():
            reasons.append('invalid_lane_id')
        if reasons:
            rejected.append({
                'trajectory_id': trajectory_id,
                'points': n,
                'reasons': '|'.join(reasons),
            })
            continue

        speed = xyz[:, 2]
        # 主 features 保留记录速度；平滑版本另存，方便做平滑敏感性对照。
        short_for_smoothing = n < smoothing_points
        smooth = speed.copy() if short_for_smoothing else savgol_filter(speed, smoothing_points, 2)
        clipped_count = int((smooth < 0).sum())
        smooth_parts.append(np.maximum(smooth, 0).astype(np.float32))
        # 纵向位置差分速度与相邻两帧平均记录速度之差，仅作诊断。
        error = np.abs(np.diff(xyz[:, 1]) / (FRAME_MS / 1000) - (speed[1:] + speed[:-1]) / 2)
        raw_acc = part.v_Acc.to_numpy(dtype=np.float64)  # 未换算单位，仍为 ft/s²
        indices.append({
            'trajectory_id': trajectory_id,
            'recording_id': int(recording_id),
            'vehicle_id': int(vehicle_id),
            'vehicle_class': int(part.v_Class.iloc[0]),
            'offset': offset,
            'length': n,
            'duration_s': float((timestamp[-1] - timestamp[0]) / 1000),
            'start_time_ms': int(timestamp[0]),
            'end_time_ms': int(timestamp[-1]),
            'start_frame': int(frames[0]),
            'end_frame': int(frames[-1]),
            'origin_x_m': float(xyz[0, 0]),
            'origin_y_m': float(xyz[0, 1]),
            # 以下均为质量标记，不作为剔除规则。
            'position_speed_error_max_m_s': float(error.max()),
            'position_speed_mismatch_pairs': int((error > mismatch_threshold).sum()),
            'reverse_longitudinal_step_pairs': int((np.diff(xyz[:, 1]) < -0.05).sum()),
            'raw_acceleration_boundary_points': int(
                np.isclose(np.abs(raw_acc), 11.2, rtol=0, atol=1e-8).sum()
            ),
            'raw_acceleration_missing_points': int((~np.isfinite(raw_acc)).sum()),
            'smoothing_negative_clipped_points': clipped_count,
            'short_for_smoothing': short_for_smoothing,
        })
        accepted.append(part)
        offset += n
    if not accepted:
        raise ValueError('No structurally valid trajectories remain')

    # 将保留的轨迹拼成连续数组，并把位置和时间转为相对各轨迹起点的值。
    clean = pd.concat(accepted, ignore_index=True)
    index = pd.DataFrame(indices)
    lengths = index.length.to_numpy(dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    origins = np.repeat(index[['origin_x_m', 'origin_y_m']].to_numpy(), lengths, axis=0)
    features = clean[['Local_X', 'Local_Y', 'v_Vel']].to_numpy(dtype=np.float64) * FT_TO_M
    features[:, :2] -= origins
    time = (
        clean.Global_Time.to_numpy(dtype=np.int64)
        - np.repeat(index.start_time_ms.to_numpy(), lengths)
    ) / 1000
    return {
        'clean': clean,
        'index': index,
        'offsets': offsets,
        'features': features.astype(np.float32),
        'time': time,
        'speed_smoothed': np.concatenate(smooth_parts),
        'rejected': rejected,
        'raw_rows': raw_rows,
        'duplicates_removed': duplicates,
        'deduplicated_rows': len(frame),
        'recordings': recording_reports,
    }


def write_h5(path, prepared, metadata):
    """变长轨迹连续存放，offsets[i]:offsets[i+1] 对应第 i 条，不提前补零。"""
    clean, index = prepared['clean'], prepared['index']
    # 模式 'x' 在目标已存在时报错，防止覆盖已有结果。
    with h5py.File(path, 'x') as handle:
        handle.attrs['schema'] = 'ngsim_continuous_v1'
        handle.attrs['feature_names'] = json.dumps(['relative_x_m', 'relative_y_m', 'speed_m_s'])
        handle.attrs['time_unit'] = 'second'
        handle.attrs['feature_units'] = json.dumps(['metre', 'metre', 'metre/second'])
        handle.attrs['source_sha256'] = metadata['source_sha256']
        handle.attrs['frame_interval_ms'] = FRAME_MS
        handle.attrs['speed_smoothing_points'] = metadata['smoothing']['window_points']
        handle.attrs['speed_smoothing_polyorder'] = 2
        handle.attrs['speed_smoothing_negative_clipped'] = True
        handle.attrs['normalized'] = False
        handle.attrs['padded'] = False
        handle.attrs['has_behavior_labels'] = False
        for name in ['features', 'time', 'speed_smoothed', 'offsets']:
            handle.create_dataset(
                name, data=prepared[name], compression='gzip', compression_opts=4, shuffle=True
            )
        tracks = handle.create_group('tracks')
        tracks.create_dataset(
            'trajectory_id', data=index.trajectory_id.to_numpy(), dtype=h5py.string_dtype('utf-8')
        )
        for column in index.columns.drop('trajectory_id'):
            tracks.create_dataset(column, data=index[column].to_numpy())
        # 下列字段仅用于追溯和事后解释，不自动拼入模型输入。
        observations = handle.create_group('observations')
        for name, values in {
            'global_time_ms': clean.Global_Time.to_numpy(dtype=np.int64),
            'frame_id': clean.Frame_ID.to_numpy(dtype=np.int32),
            'lane_id': clean.Lane_ID.to_numpy(dtype=np.int16),
            'raw_acceleration_m_s2': (clean.v_Acc.to_numpy() * FT_TO_M).astype(np.float32),
        }.items():
            observations.create_dataset(
                name, data=values, compression='gzip', compression_opts=4, shuffle=True
            )


def verify_h5(path, prepared):
    """写回校验全部点和轨迹边界，避免只检查文件能否打开。"""
    with h5py.File(path, 'r') as handle:
        offsets = handle['offsets'][:]
        lengths = handle['tracks/length'][:]
        np.testing.assert_array_equal(offsets, prepared['offsets'])
        np.testing.assert_array_equal(np.diff(offsets), lengths)
        np.testing.assert_array_equal(handle['features'][:], prepared['features'])
        np.testing.assert_array_equal(handle['time'][:], prepared['time'])
        np.testing.assert_array_equal(handle['speed_smoothed'][:], prepared['speed_smoothed'])
        np.testing.assert_array_equal(
            handle['observations/global_time_ms'][:], prepared['clean'].Global_Time
        )
        np.testing.assert_array_equal(
            handle['observations/frame_id'][:], prepared['clean'].Frame_ID
        )
        np.testing.assert_array_equal(handle['observations/lane_id'][:], prepared['clean'].Lane_ID)
        np.testing.assert_array_equal(
            handle['tracks/trajectory_id'].asstr()[:], prepared['index'].trajectory_id
        )
        assert np.isfinite(handle['features'][:]).all()
        assert (handle['features'][:, 2] >= 0).all()
        # 逐条检查：时间从 0 开始且等间隔，位置从原点开始。
        for start, end in zip(offsets[:-1], offsets[1:]):
            time = handle['time'][start:end]
            np.testing.assert_allclose(np.diff(time), FRAME_MS / 1000, rtol=0, atol=1e-10)
            assert time[0] == 0
            np.testing.assert_array_equal(handle['features'][start, :2], [0, 0])
    return {
        'all_point_values_roundtrip': True,
        'trajectory_boundaries': True,
        'time_continuity': True,
        'zero_origin': True,
        'finite_nonnegative_speed': True,
    }


def build(source, output_dir, smoothing_points=11, mismatch_threshold=5.0):
    """执行完整流程：校验参数、整理轨迹、写出 HDF5 并生成配置与质量报告。"""
    source, output_dir = Path(source).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite: {output_dir}')
    if smoothing_points < 3 or smoothing_points % 2 == 0:
        raise ValueError('smoothing_points must be an odd integer >= 3')
    if not np.isfinite(mismatch_threshold) or mismatch_threshold <= 0:
        raise ValueError('mismatch_threshold must be positive and finite')
    before = source.stat()
    source_sha256 = file_hash(source)
    print(f'Reading {source}', flush=True)
    frame = pd.read_csv(source, thousands=',', dtype={'Location': 'category'})
    prepared = prepare_tracks(frame, smoothing_points, mismatch_threshold)
    del frame  # 原始表占用内存较大，整理后尽早释放
    index = prepared['index']
    settings = {
        'source': str(source),
        'source_bytes': before.st_size,
        'source_sha256': source_sha256,
        'generator_sha256': file_hash(__file__),
        'sampling_interval_ms': FRAME_MS,
        'feet_to_metre': FT_TO_M,
        'sample_unit': 'whole_observed_vehicle_trajectory',
        'vehicle_class_filter': None,
        'recording_id_rule': 'Global_Time - 100 * Frame_ID (inferred, not an official recording name)',
        'feature_names': ['relative_x_m', 'relative_y_m', 'speed_m_s'],
        'main_feature_speed': 'recorded_speed_after_unit_conversion',
        'smoothing': {
            'separate_dataset': 'speed_smoothed',
            'method': 'Savitzky-Golay',
            'window_points': smoothing_points,
            'polyorder': 2,
            'clip_negative': True,
            'short_track_rule': 'copy recorded speed and flag',
        },
        'position_speed_mismatch_threshold_m_s': mismatch_threshold,
        'kinematic_flags_are_exclusion_rules': False,
        'standardization_applied': False,
        'train_validation_split_created': False,
        'labels_generated': False,
    }
    report = {
        'raw_rows': prepared['raw_rows'],
        'duplicate_rows_removed': prepared['duplicates_removed'],
        'deduplicated_rows': prepared['deduplicated_rows'],
        'retained_trajectories': len(index),
        'retained_points': len(prepared['clean']),
        'rejected_trajectories': len(prepared['rejected']),
        'rejected_points': sum(row['points'] for row in prepared['rejected']),
        'vehicle_classes': {
            str(k): {'trajectories': len(g), 'points': int(g.length.sum())}
            for k, g in index.groupby('vehicle_class')
        },
        'length_points': {
            'min': int(index.length.min()),
            'median': float(index.length.median()),
            'max': int(index.length.max()),
        },
        'position_speed_mismatch_trajectories': int(
            (index.position_speed_mismatch_pairs > 0).sum()
        ),
        'position_speed_mismatch_pairs': int(index.position_speed_mismatch_pairs.sum()),
        'raw_acceleration_boundary_points': int(index.raw_acceleration_boundary_points.sum()),
        'smoothing_negative_clipped_points': int(index.smoothing_negative_clipped_points.sum()),
        'short_for_smoothing_trajectories': int(index.short_for_smoothing.sum()),
        'recordings': prepared['recordings'],
    }
    # 两两比较批次的时钟区间，只报告重叠，不做修复。
    report['recording_clock_overlaps'] = [
        {
            'recording_a': a['recording_id'],
            'recording_b': b['recording_id'],
            'overlap_s': (min(a['end_ms'], b['end_ms']) - max(a['start_ms'], b['start_ms'])) / 1000,
        }
        for i, a in enumerate(prepared['recordings'])
        for b in prepared['recordings'][i + 1 :]
        if min(a['end_ms'], b['end_ms']) > max(a['start_ms'], b['start_ms'])
    ]
    print(
        f"Retained {len(index):,} trajectories / {len(prepared['clean']):,} points; writing HDF5",
        flush=True,
    )
    output_dir.mkdir(parents=True)
    # 先写 .partial，校验通过且源文件未变后才改为正式文件名。
    partial = output_dir / 'trajectories.h5.partial'
    write_h5(partial, prepared, settings)
    report['verification'] = verify_h5(partial, prepared)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('Source changed during processing; output remains .partial')
    output = output_dir / 'trajectories.h5'
    partial.rename(output)
    index.to_csv(output_dir / 'trajectory_index.csv', index=False)
    # 显式指定列名，保证零条隔离轨迹时仍写出表头。
    pd.DataFrame(
        prepared['rejected'], columns=['trajectory_id', 'points', 'reasons']
    ).to_csv(output_dir / 'rejected_trajectories.csv', index=False)
    dump_json(output_dir / 'processing_config.json', settings)
    dump_json(output_dir / 'quality_report.json', report)
    # 校验和需在其余文件全部写完后计算。
    dump_json(output_dir / 'checksums.json', {
        p.name: file_hash(p) for p in sorted(output_dir.iterdir()) if p.is_file()
    })
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f'Output: {output}', flush=True)


def main():
    """解析命令行参数并启动构建。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/NGSIM/us101.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/NGSIM/us101_processed')
    parser.add_argument('--smoothing-points', type=int, default=11)
    parser.add_argument('--mismatch-threshold', type=float, default=5.0)
    args = parser.parse_args()
    build(args.source, args.output_dir, args.smoothing_points, args.mismatch_threshold)


if __name__ == '__main__':
    main()
