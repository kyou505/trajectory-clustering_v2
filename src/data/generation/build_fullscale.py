"""扩展为三路径/九路径数据，保留小规模样本及其词表编号。"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import build_small as PILOT

ROOT = Path(__file__).resolve().parents[3]
PILOT_DIR = ROOT / 'data/portoLocalTempo4Small_v1'
BASE_PATHS = PILOT.BASE_PATHS
REFERENCE = Path(__file__).resolve().parent / 'resources/original_porto_reference.json'
# full：九条路径 × 四种节奏 = 36 类；scale：仅前三条路径，用于控制路径多样性与类别数不变
NAMES = {'full': 'portoLocalTempo4Full_v1', 'scale': 'portoLocalTempo4Scale_v1'}

MAX_LEN = 60  # 轨迹 token 序列的固定长度，不足部分用 [PAD] 补齐
PILOT_REPLICATES = 200  # 小规模数据每条路径的 family 数
EXTRA_REPLICATES = 800  # 扩展后每条路径额外新增的 family 数
FAMILIES_PER_ROUTE = PILOT_REPLICATES + EXTRA_REPLICATES


def dump(path, value):
    """以 UTF-8、缩进格式写出 JSON，禁止 NaN/Inf。"""
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    Path(path).write_text(text + '\n')


def load_bases(pilot_dir=PILOT_DIR, base_paths=BASE_PATHS):
    """路径坐标已经随项目保存；旧 Porto 统计仅用于保持生成元数据一致。"""
    bases = json.loads(Path(base_paths).read_text())
    if len(bases) != 9 or len({b['trip_id'] for b in bases}) != 9:
        raise ValueError('Expected nine distinct base paths')

    # 前三条路径必须与小规模数据完全一致，保证小规模样本可被原样嵌入
    pilot = json.loads((Path(pilot_dir) / 'base_paths.json').read_text())
    if bases[:3] != pilot:
        raise ValueError('First three base paths must match the preserved pilot')

    # 原始 Porto 数据集的规模统计，仅写入元数据作对照
    reference = json.loads(REFERENCE.read_text())
    config = {
        'total_samples': reference['samples'],
        'num_clusters': reference['classes'],
        'num_base_trajectories': reference['base_routes'],
        'cluster_size_range': reference['cluster_size_range'],
    }
    return bases, config


def nested_rows(bases):
    """生成每条路径 1000 个 family（每个 family 含四种节奏模式）的样本行。

    前 200 个 family 与小规模数据逐一相同（同一随机种子 42）；
    额外 800 个使用独立的确定性随机流（种子 43）生成，而不是复制旧样本。
    """
    rows = []
    streams = [(PILOT_REPLICATES, 42, 0), (EXTRA_REPLICATES, 43, PILOT_REPLICATES)]
    for count, seed, offset in streams:
        for row in PILOT.make_rows(bases, count, seed):
            base = row['base_index']
            # make_rows 内部编号为 base * count + i，这里还原出 i 并加上偏移量
            replicate = row['family_id'] - base * count + offset
            mode = PILOT.MODES.index(row['temporal_mode'])
            row['family_id'] = base * FAMILIES_PER_ROUTE + replicate
            row['replicate_index'] = replicate
            # 记录该行在小规模数据中的位置（每条路径 200 family × 4 模式 = 800 行），
            # 不属于小规模数据的行记为 -1
            if base < 3 and replicate < PILOT_REPLICATES:
                row['pilot_row_index'] = base * PILOT_REPLICATES * 4 + replicate * 4 + mode
            else:
                row['pilot_row_index'] = -1
            rows.append(row)
    return sorted(rows, key=lambda r: (r['family_id'], r['label']))


def pilot_geohashes(frame):
    """从小规模数据中反推 token 编号 -> geohash 的映射，按编号顺序返回 geohash 列表。"""
    mapping = {}
    for row in frame.itertuples():
        tokens = row.trajectory.split()[:row.trajLen]  # 去掉末尾的 [PAD]
        for token, code in zip(tokens, row.trajHash):
            idx = int(token)
            if idx in mapping and mapping[idx] != code:
                raise ValueError('Inconsistent pilot vocabulary')
            mapping[idx] = code
    # 编号必须从 0 开始连续，才能直接作为后续词表的前缀
    if set(mapping) != set(range(len(mapping))):
        raise ValueError('Non-contiguous pilot vocabulary')
    return [mapping[i] for i in range(len(mapping))]


def encode(rows, geohashes):
    """把 geohash 序列和时间序列编码为空格分隔的 token 字符串，并构建词表。"""
    raw_id = {code: str(i) for i, code in enumerate(geohashes)}
    # 词表：特殊 token 占据前 4 个编号，位置 token i 映射到 i + 4
    vocab = dict(PILOT.PORTO.SPECIAL_TOKENS)
    vocab.update({str(i): i + 4 for i in range(len(geohashes))})

    output = []
    for source in rows:
        row = dict(source)
        pad = ['[PAD]'] * (MAX_LEN - row['trajLen'])
        row['trajectory'] = ' '.join([raw_id[c] for c in row['trajHash']] + pad)
        row['time'] = ' '.join([str(t) for t in row.pop('time_values')] + pad)
        output.append(row)
    return pd.DataFrame(output), vocab


def check_pilot_embedding(frame, pilot):
    """校验小规模数据的每一行都原样保留在新数据中（仅 family_id 被重新编号）。"""
    subset = frame[frame.pilot_row_index >= 0].sort_values('pilot_row_index')
    if subset.pilot_row_index.tolist() != list(range(len(pilot))):
        raise ValueError('Pilot subset is incomplete')

    fields = [
        'trajectory', 'time', 'label', 'trajLen', 'duration', 'base_trip_id', 'temporal_mode',
        'path_length_m', 'traj', 'trajHash', 'progress', 'continuous_seconds',
        'segment_time_fractions',
    ]
    for a, b in zip(subset.to_dict('records'), pilot.to_dict('records')):
        for field in fields:
            if not np.array_equal(np.asarray(a[field]), np.asarray(b[field])):
                raise ValueError(f'Pilot changed at row {a["pilot_row_index"]}, field {field}')
    return {
        'pilot_rows_preserved': len(subset),
        'original_model_input_tokens_preserved': True,
        'metadata_difference': (
            'family_id remapped to base_index * 1000 + replicate_index; '
            'pilot_row_index records the original row'
        ),
    }


def validate_data(frame):
    """结构性不变量严格校验；量化后的模板探针仅作诊断用途。

    基于三锚点插值的启发式判断在分钟取整后可能与真实标签不一致，
    但这并不意味着生成有误或节奏信息完全丢失。
    因此保留这些样本及其生成标签，不挑选“容易”的样本。
    """
    observed, truth, endpoints, mismatches = [], [], [], []
    for index, row in enumerate(frame.itertuples()):
        # 连续时间戳需严格递增，且首尾时间差等于 duration（分钟）
        seconds = np.asarray(row.continuous_seconds)
        if not np.all(np.diff(seconds) > 0) or not np.isclose(seconds[-1] - seconds[0], row.duration * 60):
            raise ValueError('Invalid continuous timestamps/duration')

        # 连续时间下的三段耗时占比，最近的节奏模板必须就是该样本的标签
        expected = PILOT.MODES.index(row.temporal_mode)
        continuous = np.asarray(row.segment_time_fractions)
        nearest = int(((continuous[None, :] - PILOT.PROFILES) ** 2).sum(1).argmin())
        if nearest != expected:
            raise ValueError('Generated continuous rhythm disagrees with its label')

        # 编码后的时间 token 为 15 秒一格，整除 4 得到分钟（与模型实际输入的量化一致），需单调不减
        raw = np.array([int(t) for t in row.time.split() if t != '[PAD]'])
        minute = raw // 4
        if len(raw) != row.trajLen or not np.all(np.diff(minute) >= 0):
            raise ValueError('Invalid encoded timestamps/count')

        # 诊断：在 0、1/3、2/3、1 的路程进度处插值分钟时间，计算三段耗时占比
        anchors = np.interp([0, 1 / 3, 2 / 3, 1], row.progress, minute)
        fractions = np.diff(anchors) / (anchors[-1] - anchors[0])
        if not np.isfinite(fractions).all():
            raise ValueError('Non-finite quantized rhythm diagnostic')
        predicted = int(((fractions[None, :] - PILOT.PROFILES) ** 2).sum(1).argmin())
        if predicted != expected:
            # 只记录不一致的样本，不剔除
            mismatches.append({
                'row_index': index,
                'family_id': row.family_id,
                'true_mode': row.temporal_mode,
                'template_prediction': PILOT.MODES[predicted],
                'observed_fractions': fractions.tolist(),
            })
        observed.append(fractions)
        truth.append(expected)
        endpoints.append((int(raw[0]), int(raw[-1])))

    # 同一 family 内四种模式必须共享几何轨迹、时长和点数，仅节奏不同
    grouped = frame.groupby('family_id')
    if not (
        grouped.size().eq(4).all()
        and grouped.temporal_mode.nunique().eq(4).all()
        and grouped.trajectory.nunique().eq(1).all()
        and grouped.duration.nunique().eq(1).all()
        and grouped.trajLen.nunique().eq(1).all()
    ):
        raise ValueError('Broken matched four-mode family')

    # 同一 family 内编码后的起止时间 token 也必须相同
    starts = pd.Series([p[0] for p in endpoints]).groupby(frame.family_id).nunique()
    ends = pd.Series([p[1] for p in endpoints]).groupby(frame.family_id).nunique()
    if not starts.eq(1).all() or not ends.eq(1).all():
        raise ValueError('Matched modes differ in start or total duration')

    observed, truth = np.asarray(observed), np.asarray(truth)
    per_class = frame.label.value_counts().sort_index()
    return {
        'samples': len(frame),
        'clusters': int(frame.label.nunique()),
        'per_class': {str(k): int(v) for k, v in per_class.items()},
        'same_geometry_duration_start_and_count_within_four_mode_family': True,
        'continuous_time_strictly_increasing': True,
        'continuous_rhythm_labels_valid': True,
        'encoded_time_nondecreasing': True,
        'known_profile_sanity_accuracy_after_minute_encoding': 1 - len(mismatches) / len(frame),
        'profile_sanity_note': (
            'Known-profile, three-anchor interpolation heuristic, '
            'not clustering performance or proof of information loss.'
        ),
        'quantized_template_mismatches_retained': mismatches,
        'samples_filtered_by_template_probe': 0,
        'mean_observed_fractions_by_mode': {
            m: observed[truth == i].mean(0).tolist() for i, m in enumerate(PILOT.MODES)
        },
    }


def write_dataset(name, frame, vocab, bases, old_config, pilot, *, output_root, pilot_dir):
    """校验数据后写出 h5 数据、长度表、词表、配置、质量报告及校验和。"""
    quality = validate_data(frame)
    quality.update(check_pilot_embedding(frame, pilot))
    quality['duplicate_encoded_location_time_pairs'] = int(frame.duplicated(['trajectory', 'time']).sum())
    quality['family_count'] = int(frame.family_id.nunique())
    quality['independent_base_routes'] = len(bases)

    # 在小规模数据的生成配置基础上更新为扩展数据的配置
    config = json.loads((pilot_dir / 'generation_config.json').read_text())
    config.update(
        dataset=name,
        synthetic_controlled_pilot=False,
        synthetic_controlled_scale_experiment=True,
        num_bases=len(bases),
        samples_per_class=FAMILIES_PER_ROUTE,
        total_samples=len(frame),
        num_clusters=len(bases) * 4,
        source_base_trip_ids=[r['trip_id'] for r in bases],
        generation_streams=[
            {'seed': 42, 'replicates_per_route': PILOT_REPLICATES},
            {'seed': 43, 'replicates_per_route': EXTRA_REPLICATES},
        ],
        vocabulary_rule=(
            'Preserve all pilot token IDs, append scale-route tokens, '
            'then additional full-route tokens.'
        ),
        original_porto_reference={
            'samples': old_config['total_samples'],
            'classes': old_config['num_clusters'],
            'base_routes': old_config['num_base_trajectories'],
            'cluster_size_range': old_config['cluster_size_range'],
        },
        limitations=[
            'Synthetic rhythms on three or nine fixed Porto routes; '
            'more samples do not mean more independent routes.',
            'Changing sample count at fixed epochs also changes optimizer update count.',
            'The full dataset changes both route diversity and class count; '
            'use the 3-route scale control to hold those fixed.',
            'Profile sanity uses the known generating rule and is not model performance.',
        ],
    )

    out = output_root / name
    out.mkdir(parents=True, exist_ok=False)  # 目录已存在则报错，避免覆盖
    frame.to_hdf(out / 'data_k3.h5', key='x', mode='w', complevel=5, complib='zlib')
    lengths = frame[['trajLen']].rename(columns={'trajLen': 'length'})
    lengths.to_csv(out / 'trj_length.csv', index=False)
    dump(out / 'location_vocab.json', vocab)
    dump(out / 'base_paths.json', bases)
    dump(out / 'generation_config.json', config)
    dump(out / 'quality_report.json', quality)

    # 最后对目录内所有文件计算 sha256
    files = sorted(p for p in out.iterdir() if p.is_file())
    dump(out / 'checksums.json', {p.name: PILOT.sha256(p) for p in files})

    summary = {
        'dataset': name,
        'samples': len(frame),
        'classes': len(bases) * 4,
        'vocabulary_size': len(vocab),
        'pilot_rows_preserved': quality['pilot_rows_preserved'],
        'signal_sanity': quality['known_profile_sanity_accuracy_after_minute_encoding'],
        'duplicate_encoded_pairs': quality['duplicate_encoded_location_time_pairs'],
    }
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--pilot-dir', type=Path, default=PILOT_DIR)
    parser.add_argument('--base-paths', type=Path, default=BASE_PATHS)
    args = parser.parse_args()

    # 任一目标目录已存在则直接退出，不覆盖已有数据
    for name in NAMES.values():
        if (args.output_root / name).exists():
            raise FileExistsError(f'Refusing to overwrite {name}')

    pilot = pd.read_hdf(args.pilot_dir / 'data_k3.h5', key='x')
    bases, old_config = load_bases(args.pilot_dir, args.base_paths)
    print('Loaded the nine original Porto routes; generating 36,000 samples.', flush=True)
    rows = nested_rows(bases)
    scale_rows = [r for r in rows if r['base_index'] < 3]

    # 词表按嵌套顺序构建：小规模 token 编号不变 -> 追加 scale 新增 token -> 追加 full 新增 token
    initial = pilot_geohashes(pilot)
    scale_codes = initial + sorted({c for r in scale_rows for c in r['trajHash']} - set(initial))
    full_codes = scale_codes + sorted({c for r in rows for c in r['trajHash']} - set(scale_codes))
    scale, scale_vocab = encode(scale_rows, scale_codes)
    full, full_vocab = encode(rows, full_codes)

    # rows 按 family_id 排序，前三条路径排在最前，因此 scale 应是 full 的前缀
    if not scale.equals(full.iloc[:len(scale)].reset_index(drop=True)):
        raise ValueError('Scale dataset is not an exact row/token subset of full dataset')

    write_dataset(
        NAMES['scale'], scale, scale_vocab, bases[:3], old_config, pilot,
        output_root=args.output_root, pilot_dir=args.pilot_dir,
    )
    write_dataset(
        NAMES['full'], full, full_vocab, bases, old_config, pilot,
        output_root=args.output_root, pilot_dir=args.pilot_dir,
    )


if __name__ == '__main__':
    main()
