"""验证数据边界、遮盖泄漏和连续编码；不运行正式训练或优化器更新。"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.continuous import (
    ContinuousDataset, collate_continuous, fit_normalization, span_mask, split_trajectories,
)
from src.models.continuous import (
    ContinuousMaskedEncoder, masked_mse_per_trajectory, masked_reconstruction_loss,
)
from src.training.train_continuous import (
    make_loader, reconstruction_pass, save_checkpoint, export_embeddings,
)


@pytest.fixture(autouse=True)
def small_cpu_workload():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    yield
    torch.set_num_threads(old)


def make_store():
    lengths = [31, 43, 37]
    features = []
    times = []
    for i, length in enumerate(lengths):
        time = np.arange(length, dtype=np.float32) / 10
        times.append(time)
        features.append(np.column_stack([time * .1, time * 5, np.full(length, 5 + i)]).astype(np.float32))
    metadata = pd.DataFrame({'h5_index': [0, 1, 2], 'trajectory_id': ['a', 'b', 'c'],
                             'length': lengths, 'partition': ['train', 'train', 'validation']})
    return SimpleNamespace(frame_interval_s=.1, metadata=metadata,
                           trajectory=lambda i: (features[i], times[i]), features=features)


def setup_dataset(store, *, training=False, indices=None):
    normalization = fit_normalization(store, [0, 1])
    dataset = ContinuousDataset(store, [0, 1, 2] if indices is None else indices, normalization,
                                {'ratio': .3, 'span_seconds': .5}, 42, training=training)
    return dataset, normalization


@pytest.mark.parametrize('length', [3, 160, 1328])
def test_span_mask_has_correct_budget_and_visible_endpoints(length):
    mask = span_mask(length, .2, 20, np.random.default_rng(42))
    assert mask.sum() == min(length - 2, max(1, round(length * .2)))
    assert not mask[0] and not mask[-1]


def test_normalization_never_uses_validation_values():
    store = make_store()
    before = fit_normalization(store, [0, 1])
    store.features[2][:] = 1e6
    after = fit_normalization(store, [0, 1])
    assert before == after
    assert before['training_points'] == 74


def test_recording_holdout_purges_whole_overlapping_tracks():
    meta = pd.DataFrame({
        'h5_index': range(6), 'recording_id': [100, 100, 200, 200, 100, 200],
        'vehicle_id': [7, 8, 7, 8, 9, 9], 'vehicle_class': [2, 2, 2, 2, 3, 2],
        'start_time_ms': [0, 9000, 10000, 16000, 2000, 18000],
        'end_time_ms': [7000, 12000, 14000, 19000, 4000, 21000],
        'position_speed_mismatch_pairs': [0, 0, 0, 1, 0, 0],
    })
    store = SimpleNamespace(metadata=meta)
    cfg = {'train_recordings': [100], 'validation_recordings': [200], 'vehicle_class': 2,
           'overlap_guard_seconds': 0, 'exclude_position_speed_mismatch': False}
    train, valid, table, report = split_trajectories(store, cfg)
    assert train.tolist() == [0]
    assert valid.tolist() == [3, 5]
    assert table.loc[1, 'partition'] == table.loc[2, 'partition'] == 'excluded_clock_overlap'
    assert table.loc[4, 'partition'] == 'excluded_scope'
    assert report['flagged_retained_trajectories'] == 1
    with pytest.raises(ValueError):
        split_trajectories(store, {**cfg, 'validation_recordings': [100]})


def test_mask_reproducibility_and_valid_zero_padding():
    store = make_store()
    valid, _ = setup_dataset(store)
    old = valid[0]['prediction_mask'].clone()
    valid.set_epoch(7)
    assert torch.equal(old, valid[0]['prediction_mask'])
    train, _ = setup_dataset(store, training=True)
    first = train[0]['prediction_mask'].clone()
    train.set_epoch(3)
    assert not torch.equal(first, train[0]['prediction_mask'])
    batch = collate_continuous([valid[0], valid[1]])
    assert batch['valid_mask'][0, 0] and batch['valid_mask'][0, :31].all()
    assert not batch['valid_mask'][0, 31:].any()
    assert not batch['prediction_mask'][0, 31:].any()


def test_interpolation_uses_visible_points_and_physical_time():
    store = make_store()
    dataset, _ = setup_dataset(store)
    item = dataset[0]
    # 这些合成通道是时间的线性函数，使用同样的可见点应恢复全部掩码目标。
    np.testing.assert_allclose(item['features'].numpy(), item['interpolation'].numpy(), atol=1e-6)
    old = item['interpolation'][item['prediction_mask']].clone()
    store.features[0][item['prediction_mask'].numpy()] += 1000
    changed = dataset[0]
    torch.testing.assert_close(old, changed['interpolation'][changed['prediction_mask']])


def test_encoder_cannot_see_masked_targets_and_ignores_padding():
    dataset, _ = setup_dataset(make_store())
    batch = collate_continuous([dataset[0], dataset[1]])
    model = ContinuousMaskedEncoder(d_model=16, num_heads=2, num_layers=1, dim_feedforward=32, dropout=0).eval()
    def forward(features, time=batch['time']):
        return model(features, time, batch['valid_mask'], batch['prediction_mask'])
    base = forward(batch['features'])
    changed = batch['features'].clone()
    changed[batch['prediction_mask']] = 10000
    altered_time = batch['time'].clone()
    changed[~batch['valid_mask']] = -10000
    altered_time[~batch['valid_mask']] = 9999
    again = forward(changed, altered_time)
    torch.testing.assert_close(base['reconstruction'][batch['valid_mask']], again['reconstruction'][batch['valid_mask']])
    torch.testing.assert_close(base['embedding'], again['embedding'])
    single = collate_continuous([dataset[0]])
    alone = model(single['features'], single['time'], single['valid_mask'], single['prediction_mask'])
    torch.testing.assert_close(base['embedding'][0], alone['embedding'][0], atol=2e-6, rtol=1e-5)
    features = batch['features'].clone().requires_grad_()
    forward(features)['reconstruction'].sum().backward()
    assert torch.count_nonzero(features.grad[batch['prediction_mask']]) == 0
    assert torch.isfinite(features.grad).all()


def test_loss_uses_only_masked_points_and_equal_trajectory_weight():
    target = torch.zeros(2, 4, 3)
    predicted = torch.ones_like(target)
    mask = torch.tensor([[True, False, False, False], [True, True, True, False]])
    loss = masked_reconstruction_loss(predicted, target, mask)
    assert loss.item() == .5
    predicted[~mask] = 10000
    assert masked_reconstruction_loss(predicted, target, mask).item() == .5
    torch.testing.assert_close(masked_mse_per_trajectory(predicted, target, mask), torch.ones(2, 3))


def test_validation_checkpoint_and_embedding_exports_without_training(tmp_path):
    store = make_store()
    dataset, normalization = setup_dataset(store)
    loader_cfg = {'batch_size': 2, 'num_workers': 0}
    loader = make_loader(dataset, loader_cfg)
    model = ContinuousMaskedEncoder(d_model=16, num_heads=2, num_layers=1, dim_feedforward=32, dropout=0)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    metrics, rows = reconstruction_pass(model, loader, normalization, torch.device('cpu'))
    assert len(rows) == 3 and metrics['interpolation_normalized_mse'] < 1e-10
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])
    checkpoint = tmp_path / 'checkpoint.pt'
    save_checkpoint(checkpoint, {'model_state_dict': model.state_dict(), 'normalization': normalization})
    saved = torch.load(checkpoint, weights_only=True)
    model.load_state_dict(saved['model_state_dict'])
    cfg = {'pretrain': loader_cfg, 'mask': {'ratio': .2, 'span_seconds': .5}, 'seed': 42}
    summary = export_embeddings(model, store, np.arange(3), normalization, cfg, store.metadata,
                                tmp_path / 'representations', torch.device('cpu'))
    with np.load(tmp_path / 'representations/embeddings.npz', allow_pickle=False) as data:
        assert data['embeddings'].shape == (3, 16)
        assert data['trajectory_id'].tolist() == ['a', 'b', 'c']
    assert summary['clustering_performed'] is False
