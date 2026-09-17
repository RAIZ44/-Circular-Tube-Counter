from __future__ import annotations

import json
import os
import random
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but this PyTorch build cannot use CUDA")
        if device.type == "xpu" and not (
            hasattr(torch, "xpu") and torch.xpu.is_available()
        ):
            raise RuntimeError(
                "Intel XPU was requested, but it is unavailable. Install the PyTorch "
                "XPU wheel and update the Intel graphics driver."
            )
        if device.type == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            raise RuntimeError("MPS was requested, but it is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_enabled(device: torch.device) -> bool:
    """Use FP16 autocast on CUDA and Intel XPU accelerators."""
    return device.type in {"cuda", "xpu"}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    manifest = Path(path).resolve()
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                for field in ("image_path", "source_json"):
                    if record.get(field):
                        record[field] = str(resolve_manifest_path(record[field], manifest))
                records.append(record)
    return records


def resolve_manifest_path(value: str, manifest: Path) -> Path:
    """Resolve portable paths; also read legacy manifests after moving the project."""
    normalized = value.replace("\\", "/")
    path = Path(normalized)
    absolute = path.is_absolute() or PureWindowsPath(value).is_absolute()
    candidate = path if absolute else manifest.parent / path
    if candidate.is_file():
        return candidate.resolve()
    # A bounded migration fallback: only the known sibling dataset directory,
    # never a drive-wide filename search or an unrelated matching basename.
    parts = normalized.split("/")
    if absolute and "pipe-datasets" in parts:
        tail = Path(*parts[parts.index("pipe-datasets"):])
        for parent in manifest.parents:
            migrated = parent / tail
            if migrated.is_file():
                return migrated.resolve()
    return candidate.resolve()


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            portable = dict(record)
            for field in ("image_path", "source_json"):
                if portable.get(field) and Path(portable[field]).is_absolute():
                    portable[field] = Path(os.path.relpath(portable[field], output.resolve().parent)).as_posix()
            handle.write(json.dumps(portable, separators=(",", ":")) + "\n")


def save_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
