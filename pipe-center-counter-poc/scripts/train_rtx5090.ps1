# Train a fresh model and evaluate its reserved test split after training.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$projectPython = Join-Path (Split-Path -Parent $projectRoot) '.venv\Scripts\python.exe'
Push-Location $projectRoot
try {
    if (Test-Path -LiteralPath 'runs\pipe_center_v2\best.pt') {
        throw 'runs\pipe_center_v2 already has a checkpoint. Choose a new output directory to preserve it.'
    }
    $env:OMP_NUM_THREADS = '4'
    & $projectPython scripts\train.py --manifest-dir data\processed_v2 --output-dir runs\pipe_center_v2 --image-size 512 --batch-size 16 --epochs 40 --patience 8 --num-workers 4 --device cuda
    if ($LASTEXITCODE -ne 0) { throw 'Training failed; evaluation was not started.' }
    & $projectPython scripts\evaluate.py --checkpoint runs\pipe_center_v2\best.pt --manifest data\processed_v2\test.jsonl --output runs\pipe_center_v2\test_metrics.json --batch-size 16 --num-workers 4 --device cuda
    if ($LASTEXITCODE -ne 0) { throw 'Evaluation failed.' }
} finally {
    Pop-Location
}
