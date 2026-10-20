"""Record the software and GPU environment (writes results/gate0/environment.json).

Exits non-zero when CUDA is unusable, including a wheel built without kernels for this GPU.
"""
import json
import platform
import sys
from importlib import metadata

from lookahead.config import results_dir

PACKAGES = ("torch", "transformers", "numpy", "scipy", "scikit-learn", "pandas", "pyarrow",
            "huggingface-hub", "accelerate", "datasets", "PyYAML")


def pkg_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def probe_cuda():
    info = {}
    try:
        import torch
    except ImportError:
        return {"usable": False, "error": "torch is not installed"}
    info["torch_cuda_build"] = torch.version.cuda
    info["available"] = torch.cuda.is_available()
    if not info["available"]:
        info["usable"] = False
        return info
    props = torch.cuda.get_device_properties(0)
    info.update(
        name=props.name,
        compute_capability=f"{props.major}.{props.minor}",
        total_vram_gb=round(props.total_memory / 1024**3, 2),
        bf16_supported=torch.cuda.is_bf16_supported(),
        arch_list=torch.cuda.get_arch_list(),
    )
    # A real kernel launch: this fails if the wheel has no kernels for this GPU's architecture.
    try:
        x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
        (x @ x).float().sum().item()
        torch.cuda.synchronize()
        info["matmul_bf16_ok"] = True
    except Exception as exc:  # noqa: BLE001
        info["matmul_bf16_ok"] = False
        info["matmul_error"] = repr(exc)
    info["usable"] = bool(info["matmul_bf16_ok"] and info["bf16_supported"])
    return info


def main() -> int:
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: pkg_version(name) for name in PACKAGES},
        "cuda": probe_cuda(),
    }
    out = results_dir("gate0") / "environment.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {out}")
    return 0 if report["cuda"].get("usable") else 1


if __name__ == "__main__":
    sys.exit(main())
