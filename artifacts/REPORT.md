# Tube-count evaluation report

- Checkpoint: `artifacts/checkpoints/best.pt`
- SHA256: `dd78e022ec81a3aa61c12c46cd0b19740f17f53f52b8f5ef246bc5866625ba65`
- Device: `cpu`
- Confidence threshold: `0.3`

## Test metrics (detector)

- MAE: **0.463**
- RMSE: 0.814
- Exact-match accuracy: 62.5%
- Overcount: 12.5%  |  Undercount: 25.0%
- Precision / Recall / F1 @ IoU 0.5: 0.878 / 0.859 / 0.869

## Baselines on the same test split

- Always-predict mean train count: MAE 4.362 (constant=8)
- OpenCV HoughCircles: MAE 7.513, F1 0.474

## Overlays

- `artifacts/overlays/00_000069_gt14_pred11.png`
- `artifacts/overlays/01_000010_gt12_pred10.png`
- `artifacts/overlays/02_000013_gt10_pred12.png`
- `artifacts/overlays/03_000027_gt5_pred7.png`
- `artifacts/overlays/04_000037_gt13_pred11.png`
- `artifacts/overlays/05_000079_gt13_pred11.png`
- `artifacts/overlays/06_000000_gt1_pred1.png`
- `artifacts/overlays/07_000001_gt13_pred12.png`
- `artifacts/overlays/08_000002_gt7_pred7.png`
- `artifacts/overlays/09_000003_gt3_pred3.png`
- `artifacts/overlays/10_000004_gt1_pred1.png`
- `artifacts/overlays/11_000005_gt3_pred3.png`
