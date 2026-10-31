"""服务器运行入口：配置检查 → 预训练 → 初始化中心 → 联合聚类。"""

import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
import importlib.metadata
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import torch
import yaml

from src.data.data_process import TrajectoryDataset
from src.experiment.experiment_config import (
    ROOT,
    GROUPS,
    DEFAULT_GROUPS,
    PRESETS,
    load,
    load_config,
    common_signature,
    validate,
    resolve_data_root,
)
from src.training.common import json_write


class TeeStream:
    """同时写入终端和日志文件的输出流，每次写入后立即刷新。"""

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


def digest(path):
    """记录实际运行所用文件，供套件补跑时确认版本一致。"""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        # 按 1 MiB 分块读取，避免大文件一次性载入内存
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def source_manifest():
    """计算所有源码、配置和依赖清单的哈希，用于确认代码版本。"""
    files = list((ROOT / "src").rglob("*.py")) + list(
        (ROOT / "configs").rglob("*.yaml")
    )
    files.append(ROOT / "requirements.txt")
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)}


def environment():
    """记录运行环境：Python/PyTorch/关键依赖版本及 GPU 信息。"""
    packages = ("numpy", "pandas", "scipy", "scikit-learn", "tables", "PyYAML")
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def run_directory(cfg, group, run_id):
    """参照 con-dtc，按方法、实验配置和运行时间存放结果。"""
    # 实验组 -> 结果目录类别
    category = {
        "baseline": "baseline",
        "tempo_only": "speed_only",
        "dual_branch": "dual_branch",
    }[group]
    method = "condtc" if group == "baseline" else "spdtc"
    # 内置 Porto 预设的数据集用 porto_<preset> 命名，其他数据集直接用原名
    dataset = next(
        (
            f"porto_{preset}"
            for preset, values in PRESETS.items()
            if values[0] == cfg["dataset"]
        ),
        cfg["dataset"],
    )
    variant = category
    if cfg.get('dual_branch', {}).get('task_routing', 'split_masked') == 'mstm_speed':
        variant += '_mstm_speed'
    if cfg['cluster'].get('view_mode', 'dual') == 'single':
        variant += '_single_view'
    name = f"{method}_{dataset}_{variant}_pre{cfg['pretrain']['epochs']}"
    return ROOT / "runs" / category / name / run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, default="small")
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=None)
    parser.add_argument(
        "--config", type=Path, help="单个完整 YAML；不能同时指定 --groups"
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--pretrain-epochs", type=int)
    parser.add_argument(
        "--data-root", type=Path, help="只读数据根目录；默认 SP-DTC/data"
    )
    parser.add_argument(
        "--allow-cpu", action="store_true", help="显式允许服务器 CPU 训练"
    )
    parser.add_argument(
        "--suite-dir",
        type=Path,
        help="指定一组的时间目录，补跑同一批次的另一组；代码/数据/配置须一致",
    )
    args = parser.parse_args()

    # ---- 加载配置：单个 YAML，或按实验组 + 预设加载 ----
    if args.config:
        if args.groups:
            parser.error("--config and --groups cannot be used together")
        configs = [load_config(args.config)]
        groups = [configs[0]["group"]]
    else:
        groups = args.groups or list(DEFAULT_GROUPS)
        configs = [load(group, args.preset) for group in groups]
    if len(set(groups)) != len(groups):
        parser.error("Duplicate groups")

    # 命令行参数覆盖配置文件中的对应项，然后统一校验
    for cfg in configs:
        if args.seed is not None:
            cfg["seed"] = args.seed
        if args.pretrain_epochs is not None:
            cfg["pretrain"]["epochs"] = args.pretrain_epochs
        if args.data_root is not None:
            cfg["data_root"] = str(args.data_root.resolve())
        validate(cfg)
    # 同一批次的各组只允许在速度模式开关上不同，保证对照公平
    if any(common_signature(cfg) != common_signature(configs[0]) for cfg in configs):
        raise ValueError(
            "Configurations differ beyond the speed-pattern switch; "
            "run dual_branch as a separate command"
        )
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("默认要求 CUDA。仅在确实要用服务器 CPU 训练时指定 --allow-cpu。")

    # ---- 加载数据集并记录数据文件哈希 ----
    cfg = configs[0]
    dataset = TrajectoryDataset(cfg["dataset"], resolve_data_root(cfg))
    if len(dataset) < max(10, cfg["cluster"]["num_clusters"]):
        raise ValueError("Insufficient samples for splitting/clustering")
    manifest = {
        p.name: digest(p) for p in sorted(dataset.data_dir.iterdir()) if p.is_file()
    }
    print(
        f"dataset: {cfg['dataset']}, samples: {len(dataset)}, groups: {groups}",
        flush=True,
    )

    # 延迟导入训练模块，使配置/数据检查失败时能尽早退出
    from src.training.train_pretrain import pretrain
    from src.training.train_spdtc import train_spdtc

    # 套件身份：公共配置 + 数据哈希 + 源码哈希，三者一致才允许补跑
    sources = source_manifest()
    identity = {
        "common_config": common_signature(configs[0]),
        "dataset_checksums": manifest,
        "source_checksums": sources,
    }
    if args.suite_dir:
        # 补跑：沿用已有套件的运行 ID，并核对目录与身份
        reference = args.suite_dir.resolve()
        run_id = reference.name
        if reference not in {
            run_directory(configs[0], group, run_id) for group in GROUPS
        }:
            raise ValueError(
                "suite-dir must point to a matching method/configuration/time directory"
            )
        if json.loads((reference / "suite_identity.json").read_text()) != identity:
            raise ValueError("Suite data/configuration/code differs; start a new suite")
    else:
        # 新套件：以时间戳 + 种子作为运行 ID
        run_id = f"{datetime.now():%Y%m%d_%H%M%S_%f}_seed{configs[0]['seed']}"
        if any(run_directory(configs[0], group, run_id).exists() for group in GROUPS):
            raise FileExistsError(f"Run identifier already exists: {run_id}")
    print("run identifier:", run_id, flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 逐组运行：预训练 → 联合聚类 ----
    for group in groups:
        cfg = next(c for c in configs if c["group"] == group)
        run = run_directory(cfg, group, run_id)
        if run.exists():
            # 已存在的目录：已完成则跳过，未完成则拒绝覆盖
            if json.loads((run / "suite_identity.json").read_text()) != identity:
                raise ValueError("Run data/configuration/code differs; start a new suite")
            if (run / "completed.json").exists():
                print("已完成，跳过：", run, flush=True)
                continue
            raise FileExistsError(f"保留未完成运行，请另开套件：{run}")

        # 写入复现所需的全部元信息
        run.mkdir(parents=True)
        json_write(run / "suite_identity.json", identity)
        print("run directory:", run, flush=True)
        (run / "config.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
        )
        json_write(run / "dataset_checksums.json", manifest)
        json_write(run / "source_checksums.json", sources)
        json_write(run / "environment.json", environment())

        started = time.monotonic()
        # "x" 模式：日志文件已存在时直接报错，防止覆盖
        with (run / "train.log").open("x", encoding="utf-8") as log:
            with redirect_stdout(TeeStream(sys.stdout, log)), redirect_stderr(
                TeeStream(sys.stderr, log)
            ):
                try:
                    # 每组开始前重置全部随机种子，各组互不影响
                    random.seed(cfg["seed"])
                    np.random.seed(cfg["seed"])
                    torch.manual_seed(cfg["seed"])
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                    print("group:", group, "config:", json.dumps(cfg), flush=True)
                    pretrained = pretrain(cfg, run, dataset, device)
                    selected = train_spdtc(cfg, run, dataset, device, pretrained)
                    peak_memory = (
                        torch.cuda.max_memory_allocated()
                        if device.type == "cuda"
                        else None
                    )
                    json_write(
                        run / "completed.json",
                        {
                            "status": "complete",
                            "elapsed_seconds": time.monotonic() - started,
                            "selected_checkpoint": str(selected.relative_to(run)),
                            "peak_cuda_allocated_bytes": peak_memory,
                        },
                    )
                except Exception as exc:
                    # 记录失败原因后继续抛出，便于队列脚本感知失败
                    json_write(
                        run / "failed.json",
                        {"error_type": type(exc).__name__, "message": str(exc)},
                    )
                    raise
    print("completed requested groups:", groups, "run identifier:", run_id, flush=True)


if __name__ == "__main__":
    main()
