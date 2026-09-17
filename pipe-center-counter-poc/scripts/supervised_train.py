"""Persistent local training job with logs, provenance and explicit status."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from datetime import datetime, timezone
import zipfile
import argparse

import torch
from pipe_counter.detection import DETECTOR_ARCHITECTURES


PROJECT = Path(__file__).resolve().parents[1]
SUPERVISION = PROJECT / "runs" / "supervision"
RUN = PROJECT / "runs" / "pipe_center_v2"


def now():
    return datetime.now(timezone.utc).isoformat()


def write_status(state):
    criteria_path=PROJECT/'ACCEPTANCE_CRITERIA.json'
    if criteria_path.exists():
        criteria=json.loads(criteria_path.read_text(encoding='utf-8'))
        state.pop('provisional_exact_count_target',None)
        state['minimum_exact_count_accuracy']=criteria['minimum_exact_count_accuracy']
        state['acceptance_criteria_file']=str(criteria_path)
    state['active_training']=state.get('phase')=='training'
    state["updated_at"] = now()
    temp = SUPERVISION / "status.tmp"
    temp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temp.replace(SUPERVISION / "status.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default='pipe_center_v2')
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--init-checkpoint', default=None)
    parser.add_argument('--selection-metric', choices=['mean_absolute_error','exact_count_accuracy'], default='mean_absolute_error')
    parser.add_argument('--architecture', choices=['PipeCenterNet','ResNet18CenterNet',*DETECTOR_ARCHITECTURES], default='PipeCenterNet')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--crop-probability', type=float, default=0.0)
    parser.add_argument('--crop-min-scale', type=float, default=0.5)
    parser.add_argument('--box-nms-threshold', type=float, default=.5)
    args = parser.parse_args()
    if not args.run_name.replace('_','').isalnum():
        raise ValueError('Use only letters, digits and underscores in run names')
    global RUN
    RUN = PROJECT / 'runs' / args.run_name
    SUPERVISION.mkdir(parents=True, exist_ok=True)
    # Exclusive creation refuses a duplicate supervisor or an unreviewed rerun.
    with (SUPERVISION / "job.lock").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps({"pid": os.getpid(), "started_at": now()}))
    state = {"phase": "starting", "supervisor_pid": os.getpid(), "started_at": now(),
             "run_dir": str(RUN), "automation_id": "pipe-counter-production-readiness",
             "test_evaluated": False, "production_ready": False,
             "intended_use": "Fully automatic counts; model artifact for later Azure hosting",
             "minimum_exact_count_accuracy": 0.95}
    domain_path = PROJECT / 'OPERATING_SCOPE.json'
    if domain_path.exists():
        state['operating_scope'] = json.loads(domain_path.read_text(encoding='utf-8'))
    try:
        if (RUN / "best.pt").exists() or (RUN / "history.csv").exists():
            raise RuntimeError("Refusing to overwrite an existing training run")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; refusing a silent CPU fallback")
        RUN.mkdir(parents=True, exist_ok=True)
        manifest_dir = PROJECT / "data" / "processed_v2"
        provenance = {
            "started_at": now(), "python": sys.version, "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "manifest_sha256": {s: hashlib.sha256((manifest_dir / f"{s}.jsonl").read_bytes()).hexdigest()
                                for s in ("train", "val", "test")},
            "test_policy": "Reserved. First run trains and validates only; select a candidate before test evaluation.",
            "architecture": args.architecture,
            "operating_scope": state.get('operating_scope'),
        }
        geometry_path=PROJECT/'data/geometry_overrides_v1.json'
        if args.init_checkpoint:
            init_path=(PROJECT/args.init_checkpoint).resolve()
            provenance['initial_checkpoint']={'path':str(init_path),'sha256':hashlib.sha256(init_path.read_bytes()).hexdigest()}
        if args.architecture in DETECTOR_ARCHITECTURES and geometry_path.exists():
            provenance['geometry_overrides_sha256']=hashlib.sha256(geometry_path.read_bytes()).hexdigest()
        if args.pretrained:
            if args.architecture in DETECTOR_ARCHITECTURES:
                if args.architecture=='FCOSResNet50FPN':
                    from pipe_counter.fcos import WEIGHTS_NAME, WEIGHTS_SHA256
                else:
                    from pipe_counter.detection import WEIGHTS_NAME, WEIGHTS_SHA256
                weights = PROJECT / '.cache/torch/checkpoints' / WEIGHTS_NAME
                expected = WEIGHTS_SHA256
                source_url = 'https://download.pytorch.org/models/' + WEIGHTS_NAME
                dataset_name = 'COCO'
            else:
                weights = PROJECT / '.cache/torch/checkpoints/resnet18-f37072fd.pth'
                expected = 'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
                source_url = 'https://download.pytorch.org/models/resnet18-f37072fd.pth'
                dataset_name = 'ImageNet-1K'
            digest = hashlib.sha256(weights.read_bytes()).hexdigest()
            if digest != expected:
                raise RuntimeError('Pretrained weights do not match the audited official download')
            provenance['pretraining'] = {'source':source_url,
                                        'sha256':digest, 'dataset':dataset_name,
                                        'note':'External generic pretraining; no pipe validation/test labels used for fitting.'}
        (RUN / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        with zipfile.ZipFile(RUN / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for folder in ("src/pipe_counter", "scripts", "tests"):
                for path in (PROJECT / folder).glob("*.py"):
                    archive.write(path, path.relative_to(PROJECT))
            for name in ("pyproject.toml", "requirements-local-lock.txt"):
                archive.write(PROJECT / name, name)
            if domain_path.exists():
                archive.write(domain_path, 'OPERATING_SCOPE.json')
            if args.architecture in DETECTOR_ARCHITECTURES and geometry_path.exists():
                archive.write(geometry_path, 'data/geometry_overrides_v1.json')
            criteria_path=PROJECT/'ACCEPTANCE_CRITERIA.json'
            if criteria_path.exists():
                archive.write(criteria_path,'ACCEPTANCE_CRITERIA.json')
        trainer = 'scripts/train_detector.py' if args.architecture in DETECTOR_ARCHITECTURES else 'scripts/train.py'
        command = [sys.executable, "-u", trainer, "--manifest-dir", "data/processed_v2",
                   "--output-dir", str(RUN), "--image-size", str(args.image_size), "--batch-size", str(args.batch_size),
                   "--epochs", str(args.epochs), "--patience", str(args.patience), "--learning-rate", str(args.learning_rate),
                   "--selection-metric", args.selection_metric, "--num-workers", "4", "--device", "cuda"]
        command.extend(['--architecture', args.architecture])
        if args.architecture not in DETECTOR_ARCHITECTURES:
            command.extend(['--crop-probability', str(args.crop_probability), '--crop-min-scale', str(args.crop_min_scale)])
        else:
            command.extend(['--box-nms-threshold',str(args.box_nms_threshold)])
        if args.pretrained:
            command.append('--pretrained')
        if args.init_checkpoint:
            command.extend(['--init-checkpoint', args.init_checkpoint])
        env = dict(os.environ, OMP_NUM_THREADS="4", PYTHONUNBUFFERED="1", TQDM_MININTERVAL="15")
        state.update(phase="training", command=command, log=str(RUN / "training.log"))
        with (RUN / "training.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT)
            state["training_pid"] = process.pid
            write_status(state)
            result = process.wait()
        state["training_exit_code"] = result
        if result:
            raise RuntimeError(f"Training exited with code {result}; inspect training.log")
        if not (RUN / "best.pt").is_file():
            raise RuntimeError("Training exited without a best checkpoint")
        state.update(phase="validation_review", training_finished_at=now(),
                     next_action="Review validation accuracy, per-image errors and label issues. Freeze a candidate before evaluating the reserved test set.")
        metrics_path = RUN / "best_validation_metrics.json"
        if metrics_path.exists():
            state["best_validation_metrics"] = json.loads(metrics_path.read_text())
        write_status(state)
    except BaseException as error:
        state.update(phase="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        write_status(state)
        raise


if __name__ == "__main__":
    main()
