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
NAMES = {'full': 'portoLocalTempo4Full_v1', 'scale': 'portoLocalTempo4Scale_v1'}


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def load_bases(pilot_dir=PILOT_DIR, base_paths=BASE_PATHS):
    """路径坐标已经随项目保存；旧 Porto 统计仅用于保持生成元数据一致。"""
    bases = json.loads(Path(base_paths).read_text())
    if len(bases) != 9 or len({b['trip_id'] for b in bases}) != 9:
        raise ValueError('Expected nine distinct base paths')
    pilot = json.loads((Path(pilot_dir) / 'base_paths.json').read_text())
    if bases[:3] != pilot:
        raise ValueError('First three base paths must match the preserved pilot')
    reference = json.loads(REFERENCE.read_text())
    config = {'total_samples': reference['samples'], 'num_clusters': reference['classes'],
              'num_base_trajectories': reference['base_routes'],
              'cluster_size_range': reference['cluster_size_range']}
    return bases, config


def nested_rows(bases):
    # Keep the pilot's first 200 families per route exactly; use a separate
    # deterministic stream for the additional 800, not copies of old samples.
    rows = []
    for count, seed, offset in [(200, 42, 0), (800, 43, 200)]:
        for row in PILOT.make_rows(bases, count, seed):
            base = row['base_index']
            replicate = row['family_id'] - base*count + offset
            mode = PILOT.MODES.index(row['temporal_mode'])
            row['family_id'] = base*1000 + replicate
            row['replicate_index'] = replicate
            row['pilot_row_index'] = base*800 + replicate*4 + mode if base < 3 and replicate < 200 else -1
            rows.append(row)
    return sorted(rows, key=lambda r: (r['family_id'], r['label']))


def pilot_geohashes(frame):
    mapping = {}
    for row in frame.itertuples():
        for token, code in zip(row.trajectory.split()[:row.trajLen], row.trajHash):
            idx = int(token)
            if idx in mapping and mapping[idx] != code:
                raise ValueError('Inconsistent pilot vocabulary')
            mapping[idx] = code
    if set(mapping) != set(range(len(mapping))):
        raise ValueError('Non-contiguous pilot vocabulary')
    return [mapping[i] for i in range(len(mapping))]


def encode(rows, geohashes):
    raw_id = {code: str(i) for i, code in enumerate(geohashes)}
    vocab = dict(PILOT.PORTO.SPECIAL_TOKENS)
    vocab.update({str(i): i+4 for i in range(len(geohashes))})
    output = []
    for source in rows:
        row = dict(source)
        pad = ['[PAD]']*(60-row['trajLen'])
        row['trajectory'] = ' '.join([raw_id[c] for c in row['trajHash']]+pad)
        row['time'] = ' '.join([str(t) for t in row.pop('time_values')]+pad)
        output.append(row)
    return pd.DataFrame(output), vocab


def check_pilot_embedding(frame, pilot):
    subset = frame[frame.pilot_row_index >= 0].sort_values('pilot_row_index')
    if subset.pilot_row_index.tolist() != list(range(len(pilot))):
        raise ValueError('Pilot subset is incomplete')
    fields = ['trajectory', 'time', 'label', 'trajLen', 'duration', 'base_trip_id', 'temporal_mode',
              'path_length_m', 'traj', 'trajHash', 'progress', 'continuous_seconds', 'segment_time_fractions']
    for a, b in zip(subset.to_dict('records'), pilot.to_dict('records')):
        for field in fields:
            if not np.array_equal(np.asarray(a[field]), np.asarray(b[field])):
                raise ValueError(f'Pilot changed at row {a["pilot_row_index"]}, field {field}')
    return {'pilot_rows_preserved': len(subset), 'original_model_input_tokens_preserved': True,
            'metadata_difference': 'family_id remapped to base_index * 1000 + replicate_index; pilot_row_index records the original row'}


def validate_data(frame):
    """Structural invariants are strict; the quantized template probe is diagnostic.

    An interpolated three-anchor heuristic can disagree after minute rounding
    without showing a generation error or loss of all rhythm information.
    Keep those samples and their generating labels; do not select easy cases.
    """
    observed, truth, endpoints, mismatches = [], [], [], []
    for index, row in enumerate(frame.itertuples()):
        seconds = np.asarray(row.continuous_seconds)
        if not np.all(np.diff(seconds) > 0) or not np.isclose(seconds[-1]-seconds[0], row.duration*60):
            raise ValueError('Invalid continuous timestamps/duration')
        expected = PILOT.MODES.index(row.temporal_mode)
        if int(((np.asarray(row.segment_time_fractions)[None,:]-PILOT.PROFILES)**2).sum(1).argmin()) != expected:
            raise ValueError('Generated continuous rhythm disagrees with its label')
        raw = np.array([int(t) for t in row.time.split() if t != '[PAD]'])
        minute = raw//4
        if len(raw) != row.trajLen or not np.all(np.diff(minute) >= 0):
            raise ValueError('Invalid encoded timestamps/count')
        anchors = np.interp([0, 1/3, 2/3, 1], row.progress, minute)
        fractions = np.diff(anchors)/(anchors[-1]-anchors[0])
        if not np.isfinite(fractions).all():
            raise ValueError('Non-finite quantized rhythm diagnostic')
        predicted = int(((fractions[None,:]-PILOT.PROFILES)**2).sum(1).argmin())
        if predicted != expected:
            mismatches.append({'row_index':index,'family_id':row.family_id,'true_mode':row.temporal_mode,
                               'template_prediction':PILOT.MODES[predicted],'observed_fractions':fractions.tolist()})
        observed.append(fractions)
        truth.append(expected)
        endpoints.append((int(raw[0]), int(raw[-1])))
    grouped = frame.groupby('family_id')
    if not (grouped.size().eq(4).all() and grouped.temporal_mode.nunique().eq(4).all()
            and grouped.trajectory.nunique().eq(1).all() and grouped.duration.nunique().eq(1).all()
            and grouped.trajLen.nunique().eq(1).all()):
        raise ValueError('Broken matched four-mode family')
    starts = pd.Series([p[0] for p in endpoints]).groupby(frame.family_id).nunique()
    ends = pd.Series([p[1] for p in endpoints]).groupby(frame.family_id).nunique()
    if not starts.eq(1).all() or not ends.eq(1).all():
        raise ValueError('Matched modes differ in start or total duration')
    observed, truth = np.asarray(observed), np.asarray(truth)
    return {'samples':len(frame),'clusters':int(frame.label.nunique()),
            'per_class':{str(k):int(v) for k,v in frame.label.value_counts().sort_index().items()},
            'same_geometry_duration_start_and_count_within_four_mode_family':True,
            'continuous_time_strictly_increasing':True,'continuous_rhythm_labels_valid':True,
            'encoded_time_nondecreasing':True,
            'known_profile_sanity_accuracy_after_minute_encoding':1-len(mismatches)/len(frame),
            'profile_sanity_note':'Known-profile, three-anchor interpolation heuristic, not clustering performance or proof of information loss.',
            'quantized_template_mismatches_retained':mismatches,'samples_filtered_by_template_probe':0,
            'mean_observed_fractions_by_mode':{m:observed[truth==i].mean(0).tolist() for i,m in enumerate(PILOT.MODES)}}


def write_dataset(name, frame, vocab, bases, old_config, pilot, *, output_root, pilot_dir):
    quality = validate_data(frame)
    quality.update(check_pilot_embedding(frame, pilot))
    quality['duplicate_encoded_location_time_pairs'] = int(frame.duplicated(['trajectory', 'time']).sum())
    quality['family_count'] = int(frame.family_id.nunique())
    quality['independent_base_routes'] = len(bases)
    config = json.loads((pilot_dir/'generation_config.json').read_text())
    config.update(dataset=name, synthetic_controlled_pilot=False, synthetic_controlled_scale_experiment=True,
                  num_bases=len(bases), samples_per_class=1000, total_samples=len(frame), num_clusters=len(bases)*4,
                  source_base_trip_ids=[r['trip_id'] for r in bases],
                  generation_streams=[{'seed':42,'replicates_per_route':200}, {'seed':43,'replicates_per_route':800}],
                  vocabulary_rule='Preserve all pilot token IDs, append scale-route tokens, then additional full-route tokens.',
                  original_porto_reference={'samples': old_config['total_samples'], 'classes': old_config['num_clusters'],
                                            'base_routes': old_config['num_base_trajectories'], 'cluster_size_range': old_config['cluster_size_range']},
                  limitations=['Synthetic rhythms on three or nine fixed Porto routes; more samples do not mean more independent routes.',
                               'Changing sample count at fixed epochs also changes optimizer update count.',
                               'The full dataset changes both route diversity and class count; use the 3-route scale control to hold those fixed.',
                               'Profile sanity uses the known generating rule and is not model performance.'])
    out = output_root/name
    out.mkdir(parents=True, exist_ok=False)
    frame.to_hdf(out/'data_k3.h5', key='x', mode='w', complevel=5, complib='zlib')
    frame[['trajLen']].rename(columns={'trajLen':'length'}).to_csv(out/'trj_length.csv', index=False)
    dump(out/'location_vocab.json', vocab)
    dump(out/'base_paths.json', bases)
    dump(out/'generation_config.json', config)
    dump(out/'quality_report.json', quality)
    files = sorted(p for p in out.iterdir() if p.is_file())
    dump(out/'checksums.json', {p.name:PILOT.sha256(p) for p in files})
    print(json.dumps({'dataset':name,'samples':len(frame),'classes':len(bases)*4,'vocabulary_size':len(vocab),
                      'pilot_rows_preserved':quality['pilot_rows_preserved'], 'signal_sanity':quality['known_profile_sanity_accuracy_after_minute_encoding'],
                      'duplicate_encoded_pairs':quality['duplicate_encoded_location_time_pairs']}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ROOT/'data')
    parser.add_argument('--pilot-dir', type=Path, default=PILOT_DIR)
    parser.add_argument('--base-paths', type=Path, default=BASE_PATHS)
    args = parser.parse_args()
    for name in NAMES.values():
        if (args.output_root/name).exists():
            raise FileExistsError(f'Refusing to overwrite {name}')
    pilot = pd.read_hdf(args.pilot_dir/'data_k3.h5', key='x')
    bases, old_config = load_bases(args.pilot_dir, args.base_paths)
    print('Loaded the nine original Porto routes; generating 36,000 samples.', flush=True)
    rows = nested_rows(bases)
    scale_rows = [r for r in rows if r['base_index'] < 3]
    initial = pilot_geohashes(pilot)
    scale_codes = initial + sorted({c for r in scale_rows for c in r['trajHash']} - set(initial))
    full_codes = scale_codes + sorted({c for r in rows for c in r['trajHash']} - set(scale_codes))
    scale, scale_vocab = encode(scale_rows, scale_codes)
    full, full_vocab = encode(rows, full_codes)
    if not scale.equals(full.iloc[:len(scale)].reset_index(drop=True)):
        raise ValueError('Scale dataset is not an exact row/token subset of full dataset')
    write_dataset(NAMES['scale'], scale, scale_vocab, bases[:3], old_config, pilot, output_root=args.output_root, pilot_dir=args.pilot_dir)
    write_dataset(NAMES['full'], full, full_vocab, bases, old_config, pilot, output_root=args.output_root, pilot_dir=args.pilot_dir)


if __name__ == '__main__':
    main()
