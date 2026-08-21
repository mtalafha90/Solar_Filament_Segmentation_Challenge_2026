#!/usr/bin/env python3
"""Check the environment and explain anything that is wrong.

Import failures in a mixed conda/pip environment tend to surface as messages
about missing symbols in shared libraries, which say nothing about what to do.
This checks each dependency, and for PyTorch specifically works out which CUDA
libraries are actually being loaded, so a conflict can be named rather than
guessed at.

Run it whenever something fails to import::

    python scripts/check_env.py
"""

from __future__ import annotations

import importlib
import os
import platform
import subprocess
import sys
from pathlib import Path

REQUIRED = [
    ("numpy", "core arrays"),
    ("scipy", "filtering and morphology"),
    ("skimage", "image processing (scikit-image)"),
    ("PIL", "reading JPEG images (Pillow)"),
    ("yaml", "config files (PyYAML)"),
    ("astropy", "reading FITS observations"),
    ("pycocotools", "the submission's RLE format"),
    ("torch", "the neural model"),
]
OPTIONAL = [
    ("torchvision", "pretrained encoder backbones"),
    ("pytest", "the test suite"),
]


def probe(module: str) -> tuple[bool, str]:
    """Import a module, returning whether it worked and what it said."""
    try:
        loaded = importlib.import_module(module)
    except Exception as error:  # noqa: BLE001 - the failure is the point
        return False, f"{type(error).__name__}: {error}"
    return True, str(getattr(loaded, "__version__", "installed"))


def find_libraries(name: str) -> list[Path]:
    """Every copy of a shared library visible to this interpreter."""
    roots = [Path(sys.prefix) / "lib"]
    roots += [Path(p) for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    for entry in sys.path:
        candidate = Path(entry) / "nvidia"
        if candidate.is_dir():
            roots.append(candidate)
    seen: dict[Path, None] = {}
    for root in dict.fromkeys(roots):
        if root.is_dir():
            for path in sorted(root.rglob(f"lib{name}.so*"))[:6]:
                seen.setdefault(path.resolve(), None)
    return list(seen)


def diagnose_torch(message: str) -> None:
    """Explain a PyTorch import failure and say how to clear it."""
    print("\n" + "=" * 70)
    print("PYTORCH DIAGNOSIS")
    print("=" * 70)
    print(f"  import failed with:\n    {message}\n")

    symbol_clash = "undefined symbol" in message
    library = ""
    for candidate in ("nccl", "cudnn", "cublas", "cudart"):
        if candidate in message.lower():
            library = candidate
            break

    if symbol_clash and library:
        print(f"  This is a version conflict in lib{library}, not a broken PyTorch.")
        print("  PyTorch's wheel bundles its own CUDA libraries, and an older copy")
        print("  is being loaded ahead of them.\n")
        copies = find_libraries(library)
        if copies:
            print(f"  Copies of lib{library} visible to this interpreter:")
            for path in copies:
                marker = "  <-- conda's, likely the culprit" if "envs" in str(
                    path
                ) and "site-packages" not in str(path) else ""
                print(f"    {path}{marker}")
        else:
            print(f"  No lib{library} found on the obvious paths.")

        ld = os.environ.get("LD_LIBRARY_PATH", "")
        print(f"\n  LD_LIBRARY_PATH: {ld or '(unset, which is what you want)'}")

        print("\n  FIX, in order:\n")
        print("    # 1. Unblock immediately with the CPU build, which has no CUDA")
        print("    #    libraries at all. Enough to run the post-processing sweep.")
        print("    pip install --force-reinstall --no-cache-dir \\")
        print("        torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        print()
        print("    # 2. Or repair CUDA: drop conda's copies, reinstall as one set.")
        print(f"    conda remove --force {library} cudatoolkit cudnn")
        print("    pip uninstall -y torch torchvision")
        print("    pip install --no-cache-dir torch torchvision \\")
        print("        --index-url https://download.pytorch.org/whl/cu124")
        print()
        print("    # 3. If LD_LIBRARY_PATH above named a system CUDA, clear it:")
        print("    unset LD_LIBRARY_PATH")
    else:
        print("  Reinstall PyTorch:")
        print("    pip install --force-reinstall --no-cache-dir torch torchvision")


def main() -> None:
    print("=" * 70)
    print("ENVIRONMENT")
    print("=" * 70)
    print(f"  python      {sys.version.split()[0]}  ({sys.executable})")
    print(f"  platform    {platform.platform()}")
    conda = os.environ.get("CONDA_DEFAULT_ENV")
    print(f"  conda env   {conda or '(not in a conda environment)'}")

    print("\n" + "=" * 70)
    print("DEPENDENCIES")
    print("=" * 70)
    torch_error = ""
    failures = 0
    for group, entries in (("required", REQUIRED), ("optional", OPTIONAL)):
        for module, purpose in entries:
            ok, detail = probe(module)
            mark = "ok  " if ok else "FAIL"
            print(f"  [{mark}] {module:14s} {detail[:44]:46s} {purpose}")
            if not ok:
                if module == "torch":
                    torch_error = detail
                if group == "required":
                    failures += 1

    if torch_error:
        diagnose_torch(torch_error)
        raise SystemExit(1)

    import torch

    print("\n" + "=" * 70)
    print("PYTORCH")
    print("=" * 70)
    print(f"  version        {torch.__version__}")
    print(f"  CUDA available {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device         {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info()
        print(f"  memory         {free / 1e9:.1f} GB free of {total / 1e9:.1f} GB")
    else:
        print("  Training will run on the CPU, which is slow but correct.")
        print("  The post-processing sweep is fine on CPU.")
        if "+cu" in torch.__version__:
            try:
                subprocess.run(["nvidia-smi"], check=True, capture_output=True)
                print("\n  nvidia-smi works, so the driver is fine but this build")
                print("  cannot reach it. Reinstall from the matching index:")
                print("    pip install --force-reinstall torch torchvision \\")
                print("        --index-url https://download.pytorch.org/whl/cu124")
            except (OSError, subprocess.CalledProcessError):
                print("  nvidia-smi is unavailable, so there is no GPU to use here.")

    if failures:
        raise SystemExit(1)
    print("\nEverything needed is present.")


if __name__ == "__main__":
    main()
