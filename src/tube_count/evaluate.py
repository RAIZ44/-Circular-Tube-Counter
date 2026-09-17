from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from tube_count.baselines import mean_count_baseline, run_hough
from tube_count.config import Config
from tube_count.dataset import make_loader
from tube_count.infer import apply_threshold, infer_loader
from tube_count.metrics import count_metrics, detection_f1
from tube_count.model import TubeNet
from tube_count.utils import dump_json, resolve_device, set_seed, sha256_file
from tube_count.viz import overlay_boxes


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[TubeNet, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("config") or {}
    model_cfg = cfg_dict.get("model") or {}
    model = TubeNet(
        base_channels=int(model_cfg.get("base_channels", 32)),
        stride=int(model_cfg.get("stride", 4)),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, ckpt


def _rows_metrics(rows: list[dict], iou_threshold: float) -> dict:
    preds = np.array([r["pred_count"] for r in rows], dtype=np.float64)
    gts = np.array([r["gt_count"] for r in rows], dtype=np.float64)
    counts = count_metrics(preds, gts)
    det = detection_f1(
        [r["pred_boxes"] for r in rows],
        [r["gt_boxes"] for r in rows],
        iou_threshold,
    )
    return {**counts, **det}


def write_overlays(rows: list[dict], out_dir: Path, n: int) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(rows, key=lambda r: abs(r["pred_count"] - r["gt_count"]), reverse=True)
    # mix high-error, empty, and a few typical frames
    picked: list[dict] = []
    seen = set()
    for row in ranked[: max(n // 2, 1)] + rows[: n]:
        if row["path"] in seen:
            continue
        seen.add(row["path"])
        picked.append(row)
        if len(picked) >= n:
            break
    paths: list[str] = []
    for i, row in enumerate(picked):
        dest = out_dir / f"{i:02d}_{Path(row['path']).stem}_gt{row['gt_count']}_pred{row['pred_count']}.png"
        overlay_boxes(
            row["image"],
            row["gt_boxes"],
            row["pred_boxes"],
            row["gt_count"],
            row["pred_count"],
            dest,
        )
        paths.append(str(dest))
    return paths


def write_report(path: Path, payload: dict) -> None:
    test = payload["test"]
    lines = [
        "# Tube-count evaluation report",
        "",
        f"- Checkpoint: `{payload['checkpoint']}`",
        f"- SHA256: `{payload['checkpoint_sha256']}`",
        f"- Device: `{payload['device']}`",
        f"- Confidence threshold: `{payload['conf_threshold']}`",
        "",
        "## Test metrics (detector)",
        "",
        f"- MAE: **{test['mae']:.3f}**",
        f"- RMSE: {test['rmse']:.3f}",
        f"- Exact-match accuracy: {test['exact_match_pct']:.1f}%",
        f"- Overcount: {test['overcount_pct']:.1f}%  |  Undercount: {test['undercount_pct']:.1f}%",
        f"- Precision / Recall / F1 @ IoU {test['iou_threshold']}: "
        f"{test['precision']:.3f} / {test['recall']:.3f} / {test['f1']:.3f}",
        "",
        "## Baselines on the same test split",
        "",
        f"- Always-predict mean train count: MAE {payload['baselines']['mean_count']['mae']:.3f} "
        f"(constant={payload['baselines']['mean_count']['constant']})",
        f"- OpenCV HoughCircles: MAE {payload['baselines']['hough']['mae']:.3f}, "
        f"F1 {payload['baselines']['hough']['f1']:.3f}",
        "",
        "## Overlays",
        "",
    ]
    for p in payload.get("overlays", []):
        lines.append(f"- `{p}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run_evaluate(cfg: Config, checkpoint: str | Path, device_name: str = "auto") -> dict:
    set_seed(cfg.seed)
    device = resolve_device(device_name)
    ckpt_path = Path(checkpoint)
    model, ckpt = load_checkpoint(ckpt_path, device)
    g, m, e = cfg.generate, cfg.model, cfg.eval
    loader = make_loader(
        g.root, "test", g.image_size, m.stride, cfg.train.batch_size, cfg.train.num_workers, False, cfg.seed, False
    )
    raw = infer_loader(model, loader, device, g.image_size, m.stride, conf_threshold=0.05, progress=True)
    thr = float(ckpt.get("conf_threshold", 0.3))
    rows = apply_threshold(raw, thr)
    test = _rows_metrics(rows, e.iou_threshold)
    overlays = write_overlays(rows, Path(e.overlay_dir), e.n_overlays)
    print("Computing baselines on test…")
    mean_m = mean_count_baseline(cfg, "test")
    hough_m = run_hough(cfg, "test")
    payload = {
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "device": str(device),
        "seed": cfg.seed,
        "conf_threshold": thr,
        "epoch": ckpt.get("epoch"),
        "val_mae_at_save": ckpt.get("val_mae"),
        "test": test,
        "baselines": {"mean_count": mean_m, "hough": hough_m},
        "overlays": overlays,
        "beats_mean_count": bool(test["mae"] < mean_m["mae"]),
        "beats_hough": bool(test["mae"] < hough_m["mae"]),
    }
    dump_json(e.metrics_path, payload)
    write_report(Path(e.report_path), payload)
    print(f"test MAE={test['mae']:.3f} RMSE={test['rmse']:.3f} exact={test['exact_match_pct']:.1f}%")
    print(f"det P/R/F1={test['precision']:.3f}/{test['recall']:.3f}/{test['f1']:.3f}")
    print(f"baseline mean-count MAE={mean_m['mae']:.3f}  hough MAE={hough_m['mae']:.3f}")
    print(f"wrote {e.metrics_path} and {e.report_path}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a tube-count checkpoint on the test split")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--checkpoint", type=str, default="artifacts/checkpoints/best.pt")
    p.add_argument("--device", type=str, default="auto")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = Config.load(args.config)
    run_evaluate(cfg, args.checkpoint, device_name=args.device)


if __name__ == "__main__":
    main()
