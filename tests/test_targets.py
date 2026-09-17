from __future__ import annotations

import numpy as np
import torch

from tube_count.targets import decode_heatmap, render_targets


def test_render_and_decode_three_circles() -> None:
    image_size, stride = 256, 4
    boxes = np.array(
        [
            [40, 40, 80, 80],
            [120, 90, 180, 150],
            [200, 200, 240, 240],
        ],
        dtype=np.float32,
    )
    heatmap, radius, mask = render_targets(boxes, image_size, stride)
    assert heatmap.shape == (64, 64)
    assert int(mask.sum()) == 3
    hm = torch.from_numpy(heatmap)
    rad = torch.from_numpy(radius)
    decoded, scores = decode_heatmap(hm, rad, image_size, stride, conf_threshold=0.4)
    assert len(decoded) == 3
    assert (scores >= 0.4).all()
