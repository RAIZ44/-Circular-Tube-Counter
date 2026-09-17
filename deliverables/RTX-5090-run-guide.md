# RTX 5090 setup and dataset split

CUDA is verified on the NVIDIA GeForce RTX 5090 (approximately 32 GB).
Installed: PyTorch 2.11.0+cu128, CUDA runtime 12.8. Nine tests passed and a
short GPU training/evaluation run completed. The full run has not started.

The available images are already grouped and split:

| Set | Images |
|---|---:|
| Training | 10,650 |
| Validation | 1,332 |
| Test | 1,330 |

To train a fresh v2 model and then evaluate the reserved test set, open
PowerShell and run:

```powershell
Set-Location 'C:\Users\Owner\Projects\Pipe Counter New\pipe-center-counter-poc'
.\scripts\train_rtx5090.ps1
```

The launcher uses CUDA, 512-pixel images, batch size 16, four loader workers,
up to 40 epochs and early stopping after eight epochs without improvement.
It refuses to overwrite an existing v2 checkpoint.

Results are written to runs/pipe_center_v2 inside the project. The best model
is best.pt and the final test evaluation is test_metrics.json. The test set
is reserved for evaluation; threshold tuning uses validation data.

This evaluates held-out groups from the supplied exports. It does not measure
performance on a new camera installation. The existing v1 checkpoint remains
available; it should not be used to claim an independent result on the v2 split.

The official CUDA installation source is the
[PyTorch previous-versions page](https://pytorch.org/get-started/previous-versions/).
