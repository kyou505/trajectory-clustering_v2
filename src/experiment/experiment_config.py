"""单分支 v3 / 路径节奏双分支 v4；统一采用采样点序号定位时间扰动。"""
from copy import deepcopy
from pathlib import Path
import math

import yaml

ROOT = Path(__file__).resolve().parents[2]
# tempo_only 沿用旧实验组名，模块与配置字段统一使用 speed_pattern。
GROUPS = {"baseline": False, "tempo_only": True, "dual_branch": True}
DEFAULT_GROUPS = ("baseline", "tempo_only")
# 每个方法只维护一份完整配置；文件名说明方法、数据规模和预训练预算。
# group 保持历史标识，以兼容已有检查点和离线结果。
CONFIG_FILES = {
    "baseline": "configs/baseline/condtc_porto_full_baseline_pre30.yaml",
    "tempo_only": "configs/ablation/spdtc_porto_full_speed_only_pre30.yaml",
    "dual_branch": "configs/architecture/spdtc_porto_full_dual_branch_pre30.yaml",
}
PRESETS = {
    "small": ("portoLocalTempo4Small_v1", 2400, 12, 15),
    "scale": ("portoLocalTempo4Scale_v1", 12000, 12, 30),
    "full": ("portoLocalTempo4Full_v1", 36000, 36, 30),
}
DEFAULT = {
    "schema_version": 3, "group": "baseline", "dataset": PRESETS["small"][0],
    "data_root": "data", "seed": 42,
    "model": {"d_model": 256, "num_heads": 2, "num_layers": 2, "dim_feedforward": 1024, "dropout": 0.1},
    "pretrain": {"epochs": 15, "batch_size": 32, "learning_rate": 0.00015},
    "cluster": {"epochs": 6, "batch_size": 32, "initialization_batch_size": 256,
                "num_clusters": 12, "representation_learning_rate": 0.00005,
                "clustering_learning_rate": 0.00015, "loss_weight": 0.5},
    "time_loss_weight": 0.1, "weight_decay": 0.01,
    "speed_pattern": {"enabled": False, "warp_axis": "sample_index", "weight": 1.0, "margin": 0.2,
                      "positive_dropout": 0.1, "amplitude": [0.7, 1.3],
                      "width": [0.12, 0.28], "center": [0.2, 0.8],
                      "minimum_rms_fraction": 0.035, "attempts": 16},
}


def validate(cfg):
    cfg = deepcopy(cfg)
    if not isinstance(cfg, dict) or type(cfg.get('schema_version')) is not int or cfg['schema_version'] not in (3, 4):
        raise ValueError('Training requires schema v3 (single branch) or v4 (dual branch), with sample_index warping')
    reference = deepcopy(DEFAULT)
    if cfg['schema_version'] == 4:
        reference['dual_branch'] = {'path_weight': 0.5, 'distance_scale': 100.0,
                                    'save_representations': True}
        if cfg.get('group') != 'dual_branch':
            raise ValueError('Schema v4 is reserved for the dual_branch architecture')
    elif cfg.get('group') == 'dual_branch':
        raise ValueError('dual_branch requires schema v4 and explicit branch settings')

    def structure(value, reference, path="root"):
        if isinstance(reference, dict):
            if not isinstance(value, dict) or set(value) != set(reference):
                raise ValueError(f"Unexpected configuration keys at {path}")
            for key in reference:
                structure(value[key], reference[key], f"{path}.{key}")
        elif isinstance(reference, bool):
            if type(value) is not bool:
                raise ValueError(f"{path} must be boolean")
        elif isinstance(reference, int):
            if type(value) is not int or value < (0 if path == "root.seed" else 1):
                raise ValueError(f"{path} must be a valid integer")
        elif isinstance(reference, float):
            zero_allowed = path in {"root.model.dropout", "root.weight_decay"}
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                    or (value < 0 if zero_allowed else value <= 0)):
                raise ValueError(f"Invalid finite number at {path}")
        elif isinstance(reference, list):
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError(f"{path} must be [lower, upper]")
            for item in value:
                structure(item, 1.0, path)
            if value[0] > value[1]:
                raise ValueError(f"{path} bounds are reversed")
        elif not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path} must be nonempty text")

    structure(cfg, reference)
    if cfg["group"] not in GROUPS:
        raise ValueError("Unsupported schema/group")
    if cfg["speed_pattern"]["enabled"] != GROUPS[cfg["group"]]:
        raise ValueError("Group does not match the speed-pattern switch")
    if cfg['schema_version'] == 4 and cfg['dual_branch']['path_weight'] >= 1:
        raise ValueError('path_weight must be in (0, 1)')
    if Path(cfg["dataset"]).name != cfg["dataset"] or cfg["dataset"] in {".", ".."}:
        raise ValueError("dataset must be a directory name")
    model, speed = cfg["model"], cfg["speed_pattern"]
    if speed['warp_axis'] != 'sample_index':
        raise ValueError('Only sample_index speed-pattern warping is supported')
    if model["d_model"] % model["num_heads"] or model["d_model"] % 2 or model["dropout"] >= 1:
        raise ValueError("Invalid Transformer dimensions/dropout")
    if cfg["cluster"]["num_clusters"] < 2:
        raise ValueError("At least two clusters are required")
    for name, _, clusters, _ in PRESETS.values():
        if cfg["dataset"] == name and cfg["cluster"]["num_clusters"] != clusters:
            raise ValueError(f"{name} requires {clusters} clusters for this experiment")
    if not 0 < speed["positive_dropout"] < 0.5:
        raise ValueError("positive_dropout must be in (0, .5)")
    if speed["center"][1] >= 1 or speed["minimum_rms_fraction"] >= 1:
        raise ValueError("Invalid normalized progress/displacement bounds")
    return cfg


def load_config(path):
    return validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def load(group, preset="small", seed=None, pretrain_epochs=None):
    if preset not in PRESETS or group not in GROUPS:
        raise ValueError("Unknown preset/group")
    cfg = load_config(ROOT / CONFIG_FILES[group])
    # small/scale 仅改变数据集、簇数和默认预训练轮数；其余参数共用方法配置。
    # full 的预训练轮数直接采用 YAML，显式命令行参数优先于这两者。
    if preset != "full":
        dataset, _, clusters, epochs = PRESETS[preset]
        cfg["dataset"] = dataset
        cfg["cluster"]["num_clusters"] = clusters
        cfg["pretrain"]["epochs"] = epochs
    if seed is not None:
        cfg["seed"] = seed
    if pretrain_epochs is not None:
        cfg["pretrain"]["epochs"] = pretrain_epochs
    return validate(cfg)


def resolve_data_root(cfg):
    path = Path(cfg["data_root"]).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def common_signature(cfg):
    signature = validate(cfg)
    signature.pop("group")
    signature["speed_pattern"].pop("enabled")
    return signature
