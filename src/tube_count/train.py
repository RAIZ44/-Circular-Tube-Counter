from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from tube_count.config import Config
from tube_count.dataset import make_loader
from tube_count.infer import apply_threshold, calibrate_threshold, infer_loader
from tube_count.metrics import count_metrics, detection_f1
from tube_count.model import TubeNet, detection_loss
from tube_count.utils import dump_json, ensure_dir, resolve_device, set_seed, sha256_file


def run_train(cfg: Config, device_name: str = "auto") -> Path:
    set_seed(cfg.seed)
    device = resolve_device(device_name)
    g, m, t, e = cfg.generate, cfg.model, cfg.train, cfg.eval
    train_loader = make_loader(
        g.root, "train", g.image_size, m.stride, t.batch_size, t.num_workers, True, cfg.seed, True
    )
    val_loader = make_loader(
        g.root, "val", g.image_size, m.stride, t.batch_size, t.num_workers, False, cfg.seed, False
    )
    model = TubeNet(base_channels=m.base_channels, stride=m.stride).to(device)
    opt = AdamW(model.parameters(), lr=t.lr, weight_decay=t.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=max(t.epochs, 1))
    save_dir = ensure_dir(t.save_dir)
    log_path = Path(t.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()
    best_mae = float("inf")
    best_path = save_dir / "best.pt"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device}  params={n_params:,}  train={len(train_loader.dataset)}  val={len(val_loader.dataset)}")
    for epoch in range(1, t.epochs + 1):
        model.train()
        t0 = time.time()
        running = {"loss": 0.0, "hm_loss": 0.0, "rad_loss": 0.0}
        n_seen = 0
        for step, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device)
            gt_hm = batch["heatmap"].to(device)
            gt_rad = batch["radius"].to(device)
            mask = batch["radius_mask"].to(device)
            pred_hm, pred_rad = model(images)
            loss, parts = detection_loss(pred_hm, pred_rad, gt_hm, gt_rad, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            bs = images.size(0)
            n_seen += bs
            for k in running:
                running[k] += parts[k] * bs
            if step % t.log_interval == 0 or step == len(train_loader):
                lr = sched.get_last_lr()[0]
                print(
                    f"epoch {epoch:02d} step {step:04d}/{len(train_loader):04d}  "
                    f"loss={parts['loss']:.4f} hm={parts['hm_loss']:.4f} "
                    f"rad={parts['rad_loss']:.4f} lr={lr:.6f}"
                )
        sched.step()
        train_avg = {k: v / max(n_seen, 1) for k, v in running.items()}
        raw_val = infer_loader(model, val_loader, device, g.image_size, m.stride, conf_threshold=0.05)
        sweep = np.linspace(e.conf_sweep_min, e.conf_sweep_max, e.conf_sweep_steps)
        thr, _ = calibrate_threshold(raw_val, sweep)
        val_rows = apply_threshold(raw_val, thr)
        preds = np.array([r["pred_count"] for r in val_rows], dtype=np.float64)
        gts = np.array([r["gt_count"] for r in val_rows], dtype=np.float64)
        counts = count_metrics(preds, gts)
        det = detection_f1([r["pred_boxes"] for r in val_rows], [r["gt_boxes"] for r in val_rows], e.iou_threshold)
        elapsed = time.time() - t0
        record = {
            "epoch": epoch,
            "lr": sched.get_last_lr()[0],
            "train": train_avg,
            "val": {**counts, **det, "conf_threshold": thr},
            "seconds": elapsed,
        }
        with log_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        print(
            f"epoch {epoch:02d} done  train_loss={train_avg['loss']:.4f}  "
            f"val_mae={counts['mae']:.3f} exact={counts['exact_match_pct']:.1f}%  "
            f"f1={det['f1']:.3f} thr={thr:.3f}  {elapsed:.1f}s"
        )
        if counts["mae"] < best_mae:
            best_mae = counts["mae"]
            payload = {
                "model": model.state_dict(),
                "config": cfg.to_dict(),
                "epoch": epoch,
                "val_mae": counts["mae"],
                "val_metrics": {**counts, **det},
                "conf_threshold": thr,
                "seed": cfg.seed,
            }
            torch.save(payload, best_path)
            dump_json(
                save_dir / "best.meta.json",
                {
                    "path": str(best_path),
                    "sha256": sha256_file(best_path),
                    "epoch": epoch,
                    "val_mae": counts["mae"],
                    "conf_threshold": thr,
                    "seed": cfg.seed,
                    "params": n_params,
                },
            )
            print(f"  saved {best_path}  val_mae={best_mae:.3f}")
    print(f"best checkpoint: {best_path}  val_mae={best_mae:.3f}")
    return best_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the circular-tube detector")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--epochs", type=int, default=None)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = Config.load(args.config)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    run_train(cfg, device_name=args.device)


if __name__ == "__main__":
    main()
