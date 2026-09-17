"""Exercise the actual CUDA training path when a compatible GPU is available."""
import pytest
import torch

from pipe_counter.heatmap import modified_focal_loss
from pipe_counter.model import PipeCenterNet
from pipe_counter.utils import resolve_device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_autocast_and_scaled_optimizer_step():
    device = resolve_device("cuda")
    model = PipeCenterNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, foreach=False)
    scaler = torch.amp.GradScaler("cuda")
    images = torch.rand(2, 3, 128, 128, device=device)
    targets = torch.zeros(2, 1, 64, 64, device=device)
    targets[:, :, 32, 32] = 1
    before = model.head[-1].bias.detach().clone()
    # Dynamic scaling can legitimately skip initial steps while reducing an
    # overflowing scale. Require an actual finite update after calibration.
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images)
        loss = modified_focal_loss(logits.float(), targets)
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        if not torch.equal(before, model.head[-1].bias):
            break
    assert torch.isfinite(model.head[-1].bias).all()
    assert not torch.equal(before, model.head[-1].bias)
