# Artifacts

Produced by `python -m tube_count.train` and `python -m tube_count.evaluate`
on the seed-42 synthetic split (360/72/80 images). The images themselves are
not committed; regenerate with `python -m tube_count.generate --config configs/default.yaml`.

| Path | What |
| --- | --- |
| `checkpoints/best.pt` | Best val-MAE weights (epoch 10) + calibrated `conf_threshold=0.3` |
| `checkpoints/best.meta.json` | SHA256, val MAE, seed, param count |
| `logs/train.jsonl` | Per-epoch loss, LR, val MAE/F1 |
| `metrics.json` | Test count + detection metrics and baselines |
| `REPORT.md` | Same numbers in Markdown |
| `overlays/*.png` | Green GT ellipse, red predicted box |
