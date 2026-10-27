"""双分支的梯度边界、距离等价性及完整训练链路回归。"""
from copy import deepcopy
import json

import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from src.data.mstm import MSTMDataset
from src.experiment.experiment_config import load, validate, DEFAULT_GROUPS
from src.experiment.run import run_directory
from src.models.dual_branch import DualBranchPretrainModel, DualBranchTrajectoryModel
from src.models.pretrain_model import MSTMLoss, SPPretrainModel
from src.models.contrastive_model import SPTrajectoryModel
from src.models.dec import target_distribution
from src.models.speed_pattern_loss import speed_pattern_loss
from src.training.cluster_init import initialize_cluster_centers
from src.training.common import build_pretrain_model, build_trajectory_model, model_kwargs
from src.training.train_pretrain import pretrain
from src.training.train_spdtc import train_spdtc


class TinyTrajectories(Dataset):
    """含 PAD 与固定末尾 SEP；所有 index 连续，走真实增强与训练函数。"""
    location_vocab_size = 24
    time_vocab_size = 1444
    max_length = 10

    def __len__(self):
        return 24

    def __getitem__(self, index):
        route, rhythm = index % 2, (index // 2) % 2
        location = torch.tensor([1, 4+route*7, 5+route*7, 6+route*7,
                                 7+route*7, 8+route*7, 9+route*7, 0, 0, 2])
        times = ([104, 108, 112, 128, 144, 164] if rhythm == 0 else
                 [104, 124, 140, 156, 160, 164])
        clock = torch.tensor([1, *times, 0, 0, 2])
        pool = location.ge(4)
        return {"index": torch.tensor(index), "label": torch.tensor(route*2+rhythm),
                "location_ids": location, "time_ids": clock,
                "pooling_mask": pool, "attention_mask": location.ne(0),
                "length": torch.tensor(6)}


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(2)
    torch.manual_seed(42)


@pytest.fixture
def cfg():
    c = load("dual_branch", "small")
    c["dataset"] = "tiny_test"
    c["model"].update(d_model=16, num_heads=2, num_layers=1, dim_feedforward=32, dropout=0.1)
    c["cluster"].update(num_clusters=4, epochs=1, batch_size=8, initialization_batch_size=8)
    c["pretrain"].update(epochs=1, batch_size=8)
    return validate(c)


def view_batch():
    return next(iter(DataLoader(TinyTrajectories(), batch_size=4)))


def nonzero_grad(module):
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


def test_path_is_time_invariant_and_parameters_are_independent(cfg):
    model = build_pretrain_model(cfg, TinyTrajectories()).eval()
    view = view_batch()
    changed = deepcopy(view)
    changed["time_ids"][changed["pooling_mask"]] += 37
    first, second = model.encode_components(view), model.encode_components(changed)
    torch.testing.assert_close(first["path"], second["path"], rtol=0, atol=0)
    assert not torch.allclose(first["rhythm"], second["rhythm"])
    assert not ({id(p) for p in model.encoder.path.parameters()} &
                {id(p) for p in model.encoder.rhythm.parameters()})
    padded = deepcopy(view)
    padded["location_ids"][~view["attention_mask"]] = 20
    padded["time_ids"][~view["attention_mask"]] = 30
    after = model.encode_components(padded)
    for key in first:
        torch.testing.assert_close(first[key], after[key])


def test_speed_gradient_and_optimizer_step_do_not_modify_path(cfg):
    model = build_pretrain_model(cfg, TinyTrajectories()).train()
    anchor = view_batch()
    negative = deepcopy(anchor)
    negative["time_ids"][:, 2:5] += 20
    views = {"anchor": anchor, "positive": anchor, "negative": negative,
             "accepted": torch.ones(4, dtype=torch.bool), "rms_fraction": torch.ones(4)*0.1}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
    path_before = {k: v.clone() for k, v in model.encoder.path.state_dict().items()}
    modes_before = [m.training for m in model.modules()]
    speed_pattern_loss(model, views, margin=5.0, device="cpu")["loss"].backward()
    assert all(p.grad is None for p in model.encoder.path.parameters())
    assert nonzero_grad(model.encoder.rhythm)
    assert modes_before == [m.training for m in model.modules()]
    optimizer.step()
    for key, value in model.encoder.path.state_dict().items():
        torch.testing.assert_close(value, path_before[key], rtol=0, atol=0)


@pytest.mark.parametrize("task,updated,untouched", [("location", "path", "rhythm"), ("time", "rhythm", "path")])
def test_masked_tasks_route_to_their_branch(cfg, task, updated, untouched):
    model = build_pretrain_model(cfg, TinyTrajectories()).eval()
    batch = next(iter(DataLoader(MSTMDataset(TinyTrajectories(), deterministic=True), batch_size=4)))
    output = model.mstm(batch)
    # 修改未遮盖原始字段不应影响掩码任务的输入。
    changed = deepcopy(batch)
    changed["location_ids"].fill_(9)
    changed["time_ids"].fill_(80)
    torch.testing.assert_close(output[f"{task}_logits"], model.mstm(changed)[f"{task}_logits"])
    F.cross_entropy(output[f"{task}_logits"].flatten(0, 1), batch[f"{task}_targets"].flatten()).backward()
    assert nonzero_grad(getattr(model.encoder, updated))
    assert all(p.grad is None for p in getattr(model.encoder, untouched).parameters())


def test_joint_distance_and_kmeans_use_same_coordinates(cfg):
    model = build_trajectory_model(cfg, TinyTrajectories()).eval()
    parts = model.encode_components(view_batch())
    z = model.fuse(parts["path"], parts["rhythm"])
    a, b = F.normalize(parts["path"], dim=1), F.normalize(parts["rhythm"], dim=1)
    expected = model.distance_scale * (model.path_weight * (a[0]-a[1]).square().sum() +
                                      (1-model.path_weight) * (b[0]-b[1]).square().sum())
    torch.testing.assert_close((z[0]-z[1]).square().sum(), expected)
    torch.testing.assert_close(z.norm(dim=1), torch.full((4,), model.distance_scale**0.5))
    km = initialize_cluster_centers(model, z.detach(), num_clusters=4, n_init=1)
    torch.testing.assert_close(model.clustering_layer.cluster_centers,
                               torch.from_numpy(km.cluster_centers_))
    distances = (z[:, None] - model.clustering_layer.cluster_centers).square().sum(-1)
    expected_q = 1 / (1 + distances)
    expected_q /= expected_q.sum(1, keepdim=True)
    torch.testing.assert_close(model.clustering_layer(z), expected_q)
    assert torch.equal(model.clustering_layer(z).argmax(1), torch.tensor(km.predict(z.detach().numpy())))


def test_dec_backpropagates_to_both_branches_and_centers(cfg):
    model = build_trajectory_model(cfg, TinyTrajectories()).eval()
    q = model.clustering_layer(model.encode(view_batch()))
    p = target_distribution(q)
    F.kl_div(q.log(), p, reduction="batchmean").backward()
    assert nonzero_grad(model.encoder.path)
    assert nonzero_grad(model.encoder.rhythm)
    assert nonzero_grad(model.clustering_layer)


def test_pretraining_checkpoint_roundtrip_and_rejects_mismatch(cfg, tmp_path):
    pre = build_pretrain_model(cfg, TinyTrajectories()).eval()
    checkpoint = tmp_path / "pre.pt"
    state = {"experiment_schema": "sp_dtc_pretrain_v1", "configs": cfg,
             "model_state_dict": pre.state_dict(), "model_kwargs": model_kwargs(cfg, TinyTrajectories())}
    torch.save(state, checkpoint)
    joint = build_trajectory_model(cfg, TinyTrajectories()).eval()
    joint.load_pretrained_components(checkpoint)
    torch.testing.assert_close(pre.encode(view_batch()), joint.encode(view_batch()), rtol=0, atol=0)
    altered = deepcopy(cfg)
    altered["dual_branch"]["path_weight"] = 0.3
    with pytest.raises(ValueError, match="path_weight"):
        build_trajectory_model(altered, TinyTrajectories()).load_pretrained_components(checkpoint)
    altered = deepcopy(cfg)
    altered["model"]["num_heads"] = 4
    with pytest.raises(ValueError, match="encoder settings"):
        build_trajectory_model(altered, TinyTrajectories()).load_pretrained_components(checkpoint)
    state["configs"] = load("tempo_only", "small")
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="dual-branch pretraining"):
        joint.load_pretrained_components(checkpoint)


@pytest.mark.parametrize("value", [0, 1, -0.1, float("nan")])
def test_invalid_weights_rejected(cfg, value):
    cfg["dual_branch"]["path_weight"] = value
    with pytest.raises(ValueError):
        validate(cfg)


def test_configs_and_baseline_factory_preserve_original_initialization(cfg):
    assert DEFAULT_GROUPS == ("baseline", "tempo_only")
    assert "/dual_branch/" in str(run_directory(cfg, "dual_branch", "test"))
    for group in ("baseline", "tempo_only"):
        single = load(group, "small")
        single["model"] = cfg["model"]
        for factory, cls, clustering in ((build_pretrain_model, SPPretrainModel, False),
                                         (build_trajectory_model, SPTrajectoryModel, True)):
            torch.manual_seed(73)
            direct = cls(**model_kwargs(single, TinyTrajectories(), clustering=clustering))
            torch.manual_seed(73)
            built = factory(single, TinyTrajectories())
            for key in direct.state_dict():
                torch.testing.assert_close(direct.state_dict()[key], built.state_dict()[key], rtol=0, atol=0)
            built.eval()
            torch.testing.assert_close(built.encode(view_batch()), built.encode_speed(view_batch()), rtol=0, atol=0)


def test_full_pretrain_cluster_pipeline_on_tiny_data(cfg, tmp_path):
    dataset = TinyTrajectories()
    pretrained = pretrain(cfg, tmp_path, dataset, torch.device("cpu"))
    checkpoint = train_spdtc(cfg, tmp_path, dataset, torch.device("cpu"), pretrained)
    state = torch.load(checkpoint, weights_only=True)
    assert state["epoch"] == 1
    assert state["model_state_dict"]["clustering_layer.cluster_centers"].shape == (4, 32)
    initial = torch.load(tmp_path / "cluster/representations/epoch_000.pt", weights_only=True)
    final = torch.load(tmp_path / "cluster/representations/epoch_001.pt", weights_only=True)
    assert torch.equal(final["sample_ids"], torch.arange(len(dataset)))
    assert not torch.allclose(initial["path"], final["path"])
    assert not torch.allclose(initial["rhythm"], final["rhythm"])
    restored = build_trajectory_model(cfg, dataset).eval()
    restored.load_state_dict(state["model_state_dict"])
    with torch.no_grad():
        q = restored.clustering_layer(restored.fuse(final["path"], final["rhythm"]))
    torch.testing.assert_close(q, final["q"])
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert all(0 <= metrics[k] <= 1 for k in ("uacc", "nmi", "ri"))
    assert -1 <= metrics["ari"] <= 1
