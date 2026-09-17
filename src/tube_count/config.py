from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


@dataclass
class GenerateConfig:
    root: str = "data"
    image_size: int = 256
    n_train: int = 360
    n_val: int = 72
    n_test: int = 80
    min_count: int = 0
    max_count: int = 16
    min_radius: int = 8
    max_radius: int = 34
    overlap_prob: float = 0.5
    clutter_prob: float = 0.85
    noise_std: float = 10.0


@dataclass
class ModelConfig:
    base_channels: int = 32
    stride: int = 4


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 8
    lr: float = 0.001
    weight_decay: float = 0.0001
    num_workers: int = 0
    log_interval: int = 10
    save_dir: str = "artifacts/checkpoints"
    log_path: str = "artifacts/logs/train.jsonl"


@dataclass
class EvalConfig:
    iou_threshold: float = 0.5
    overlay_dir: str = "artifacts/overlays"
    metrics_path: str = "artifacts/metrics.json"
    report_path: str = "artifacts/REPORT.md"
    n_overlays: int = 12
    conf_sweep_min: float = 0.15
    conf_sweep_max: float = 0.85
    conf_sweep_steps: int = 15


@dataclass
class Config:
    seed: int = 42
    generate: GenerateConfig = field(default_factory=GenerateConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def load(cls, path: str | Path | None) -> Config:
        if path is None:
            return cls()
        raw = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Config {path} must be a mapping")
        return _from_dict(cls, raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _from_dict(cls: type[T], data: dict[str, Any]) -> T:
    kwargs: dict[str, Any] = {}
    for item in fields(cls):  # type: ignore[arg-type]
        if item.name not in data or data[item.name] is None:
            continue
        value = data[item.name]
        if item.default_factory is not MISSING:
            prototype = item.default_factory()
            if is_dataclass(prototype):
                if not isinstance(value, dict):
                    raise TypeError(f"Expected mapping for {item.name}")
                kwargs[item.name] = _from_dict(type(prototype), value)
                continue
        kwargs[item.name] = value
    return cls(**kwargs)
