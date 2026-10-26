"""生成三条基础路径 × 四种局部速度模式；拒绝覆盖已有数据。"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from . import encoding as PORTO

ROOT = Path(__file__).resolve().parents[3]
BASE_PATHS = Path(__file__).resolve().parent / 'resources/base_paths.json'
MODES = ['reference', 'early_fast', 'middle_fast', 'late_fast']
PROFILES = np.array([[1/3]*3, [.2, .4, .4], [.4, .2, .4], [.4, .4, .2]])
DEFAULT_NAME = 'portoLocalTempo4Small_v1'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def arc_progress(coords):
    p = np.radians(np.asarray(coords, dtype=float))
    delta = np.diff(p, axis=0)
    a = np.sin(delta[:, 1]/2)**2 + np.cos(p[:-1, 1])*np.cos(p[1:, 1])*np.sin(delta[:, 0]/2)**2
    distance = 6371000 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    cumulative = np.r_[0., np.cumsum(distance)]
    if cumulative[-1] <= 0:
        raise ValueError('zero-length route')
    return cumulative / cumulative[-1], cumulative[-1]


def load_bases(base_file=None):
    """默认读取归档的九条路径中的前三条，无需扫描原始 Porto CSV。"""
    bases = json.loads(Path(base_file or BASE_PATHS).read_text())
    if base_file is None:
        bases = bases[:3]
    if len(bases) != 3 or len({b['trip_id'] for b in bases}) != 3:
        raise ValueError('pilot requires exactly three distinct base paths')
    return bases


def make_rows(bases, per_class, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for base_i, base in enumerate(bases):
        xy = np.asarray(base['polyline'], dtype=float)
        progress, length_m = arc_progress(xy)
        unique = np.r_[True, np.diff(progress) > 1e-12]
        grid = np.linspace(0, 1, 49)
        resampled = np.stack([np.interp(grid, progress[unique], xy[unique, k]) for k in (0, 1)], axis=1)
        for replicate in range(per_class):
            # The four modes share every nuisance draw, preventing marginal shortcuts.
            start_s = float(rng.uniform(8*3600-300, 8*3600+300))
            duration_s = float(rng.uniform(18*60, 22*60))
            perturb = rng.uniform(-.008, .008, 3)
            perturb -= perturb.mean()
            drop = rng.choice(np.arange(1, 48), 4, replace=False)
            keep = np.setdiff1d(np.arange(49), drop)
            coordinates = resampled[keep].copy()
            meters = rng.normal(0, 1, coordinates.shape)
            coordinates[:, 0] += meters[:, 0] / (111320*np.cos(np.radians(coordinates[:, 1])))
            coordinates[:, 1] += meters[:, 1] / 111320
            locations = [PORTO.encode_geohash(lon, lat, 7) for lon, lat in coordinates]
            for mode_i, mode in enumerate(MODES):
                weights = PROFILES[mode_i] + perturb
                curve = PchipInterpolator([0, 1/3, 2/3, 1], np.r_[0., np.cumsum(weights)])
                seconds = start_s + duration_s*curve(grid[keep])
                assert np.all(np.diff(seconds) > 0)
                assert np.isclose(seconds[-1]-seconds[0], duration_s)
                raw_slots = np.floor(seconds/15).astype(int)
                rows.append({'traj': coordinates.tolist(), 'time_values': raw_slots.tolist(),
                    'label': base_i*4+mode_i, 'trajLen': len(keep), 'trajHash': locations.copy(),
                    'duration': duration_s/60, 'base_trip_id': base['trip_id'], 'temporal_mode': mode,
                    'base_index': base_i, 'family_id': base_i*per_class+replicate,
                    'progress': grid[keep].tolist(), 'continuous_seconds': seconds.tolist(),
                    'segment_time_fractions': weights.tolist(), 'path_length_m': float(length_m)})
    return rows


def validate(frame):
    observed, expected = [], []
    for row in frame.itertuples():
        raw = np.array([int(x) for x in row.time.split() if x != '[PAD]'])
        minute = raw // 4  # Same quantization as the actual ConDTC input.
        assert np.all(np.diff(minute) >= 0)
        anchors = np.interp([0, 1/3, 2/3, 1], row.progress, minute)
        fractions = np.diff(anchors) / (anchors[-1]-anchors[0])
        observed.append(fractions)
        expected.append(MODES.index(row.temporal_mode))
    observed = np.array(observed)
    predicted = ((observed[:, None, :]-PROFILES[None, :, :])**2).sum(2).argmin(1)
    correct = predicted == expected
    for _, family in frame.groupby('family_id'):
        assert len(family) == 4
        assert family.trajectory.nunique() == 1
        assert family.duration.nunique() == 1
        assert family.trajLen.nunique() == 1
        endpoints = [tuple(np.array([int(x) for x in t.split() if x != '[PAD]'])[[0, -1]]) for t in family.time]
        assert len(set(endpoints)) == 1
    assert correct.all(), 'minute quantization has erased or inverted some rhythm labels'
    return {'samples':len(frame), 'clusters':int(frame.label.nunique()),
        'per_class':{str(k):int(v) for k,v in frame.label.value_counts().sort_index().items()},
        'same_geometry_duration_start_and_count_within_four_mode_family':True,
        'continuous_time_strictly_increasing':True, 'encoded_time_nondecreasing':True,
        'known_profile_sanity_accuracy_after_minute_encoding':float(correct.mean()),
        'profile_sanity_note':'Checks the designed signal using known generating profiles and path progress; not a learned baseline or clustering result.',
        'mean_observed_fractions_by_mode':{m:observed[np.array(expected)==i].mean(0).tolist() for i,m in enumerate(MODES)}}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output-dir',type=Path,default=ROOT/'data'/DEFAULT_NAME)
    p.add_argument('--base-paths',type=Path)
    p.add_argument('--per-class',type=int,default=200)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f'refusing to overwrite: {args.output_dir}')
    if args.per_class < 10:
        raise ValueError('per-class must be >= 10')
    bases=load_bases(args.base_paths)
    rows=make_rows(bases,args.per_class,args.seed)
    vocab=PORTO.encode_and_pad_rows(rows,60)
    frame=pd.DataFrame(rows)
    quality=validate(frame)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    frame.to_hdf(args.output_dir/'data_k3.h5',key='x',mode='w')
    frame[['trajLen']].rename(columns={'trajLen':'length'}).to_csv(args.output_dir/'trj_length.csv',index=False)
    (args.output_dir/'location_vocab.json').write_text(json.dumps(vocab,indent=2)+'\n')
    (args.output_dir/'base_paths.json').write_text(json.dumps(bases,indent=2)+'\n')
    config={'dataset':args.output_dir.name,'version':1,'synthetic_controlled_pilot':True,
        'seed':args.seed,'num_bases':3,'modes':MODES,'samples_per_class':args.per_class,
        'labels':'4 * base_index + mode_index','source_base_trip_ids':[b['trip_id'] for b in bases],
        'profiles':dict(zip(MODES,PROFILES.tolist())), 'segment_definition':'equal cumulative arc length',
        'time_curve':'monotone PCHIP through segment-boundary cumulative durations',
        'duration_minutes':[18,22],'start_minutes':[475,485],'spatial_noise_std_m':1,
        'resampled_points':49,'deleted_interior_points':4,'padded_length':60,
        'time_fraction_jitter':'uniform(-0.008,0.008), centered to sum zero',
        'pointwise_time_noise':False,'training_time_quantization_seconds':60,
        'nuisance_draws_paired_across_modes':True,
        'limitations':['Three public Porto paths only; constructed rhythms, not real behavior labels.',
         'Known rhythm signatures are intentionally separable; this is a mechanism pilot, not a hard benchmark.',
         'No old data or checkpoint is overwritten.']}
    (args.output_dir/'generation_config.json').write_text(json.dumps(config,indent=2)+'\n')
    (args.output_dir/'quality_report.json').write_text(json.dumps(quality,indent=2)+'\n')
    files=['data_k3.h5','trj_length.csv','location_vocab.json','base_paths.json','generation_config.json','quality_report.json']
    (args.output_dir/'checksums.json').write_text(json.dumps({f:sha256(args.output_dir/f) for f in files},indent=2)+'\n')
    print(json.dumps(quality,indent=2))


if __name__=='__main__':
    main()
