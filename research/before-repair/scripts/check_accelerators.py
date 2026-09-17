from __future__ import annotations

import json
import platform

import torch


def main() -> None:
    xpu_available = bool(hasattr(torch, "xpu") and torch.xpu.is_available())
    result: dict[str, object] = {
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "pytorch_xpu_available": xpu_available,
        "pytorch_xpu_device": (
            torch.xpu.get_device_name(0) if xpu_available else None
        ),
        "pytorch_cuda_available": torch.cuda.is_available(),
        "openvino_version": None,
        "openvino_devices": [],
    }

    try:
        import openvino as ov

        result["openvino_version"] = ov.__version__
        result["openvino_devices"] = list(ov.Core().available_devices)
    except ImportError:
        result["openvino_note"] = (
            'Not installed (optional; install with pip install -e ".[intel-inference]")'
        )
    except Exception as error:  # Drivers/plugins can fail independently of import.
        result["openvino_error"] = f"{type(error).__name__}: {error}"

    print(json.dumps(result, indent=2))

    if not xpu_available:
        print(
            "\nIntel training acceleration is not active. Install the PyTorch XPU "
            "wheel and confirm the current Intel graphics driver is installed."
        )
    if "NPU" not in result["openvino_devices"]:
        print(
            "NPU inference is not active yet. This is optional during training; "
            "we will use it after exporting a trained checkpoint to OpenVINO."
        )


if __name__ == "__main__":
    main()
