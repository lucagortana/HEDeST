#!/usr/bin/env python
"""
Installation verification script for PanoSpace.
Run this after installation to check if all components are working correctly.
"""
from __future__ import annotations

import sys
from typing import List
from typing import Tuple


def check_module(module_name: str) -> bool:
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def check_gpu() -> Tuple[bool, str]:
    """Check if GPU is available."""
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            return True, f"GPU detected: {device_name}"
        else:
            return False, "No GPU detected (CPU-only mode)"
    except ImportError:
        return False, "PyTorch not installed"


def main() -> int:
    """Run verification checks."""
    print("=" * 70)
    print("PanoSpace Installation Verification")
    print("=" * 70)
    print()

    # Core dependencies
    print("Checking core dependencies...")
    core_modules = [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("anndata", "AnnData"),
        ("scanpy", "Scanpy"),
        ("scipy", "SciPy"),
        ("sklearn", "Scikit-learn"),
        ("PIL", "Pillow"),
        ("cv2", "OpenCV"),
        ("matplotlib", "Matplotlib"),
        ("tqdm", "TQDM"),
        ("requests", "Requests"),
    ]

    all_passed = True
    for module, display_name in core_modules:
        status = "✓" if check_module(module) else "✗"
        print(f"  {status} {display_name}")
        if not check_module(module):
            all_passed = False

    print()

    # PyTorch and GPU
    print("Checking PyTorch and GPU...")
    torch_ok = check_module("torch")
    if torch_ok:
        print("  ✓ PyTorch")
        has_gpu, gpu_msg = check_gpu()
        print(f"  {'✓' if has_gpu else 'ℹ'}  {gpu_msg}")
    else:
        print("  ✗ PyTorch")
        all_passed = False

    print()

    # Advanced features
    print("Checking advanced features...")

    # Cell detection
    print("\n  [Cell Detection]")
    detection_modules = [
        ("torchvision", "TorchVision"),
        ("ray", "Ray"),
    ]
    for module, display_name in detection_modules:
        status = "✓" if check_module(module) else "✗"
        print(f"    {status} {display_name}")

    # Annotation
    print("\n  [Annotation & Deconvolution]")
    annotation_modules = [
        ("pyro", "Pyro"),
        ("pytorch_lightning", "PyTorch Lightning"),
        ("lightning", "Lightning"),
        ("transformers", "Transformers"),
        ("ot", "POT (Python Optimal Transport)"),
        ("scvi", "scvi-tools (for cell2location)"),
    ]
    for module, display_name in annotation_modules:
        status = "✓" if check_module(module) else "✗"
        print(f"    {status} {display_name}")

    # Check if gurobipy is available (optional)
    if check_module("gurobipy"):
        print("    ✓ Gurobi (optimization acceleration)")
    else:
        print("    ℹ Gurobi not installed (optional, annotation will work without it)")

    # Check pyscipopt
    if check_module("pyscipopt"):
        print("    ✓ PySCIPOpt (SCIP solver)")
    else:
        print("    ℹ PySCIPOpt not installed")

    # Check OR-Tools (default assignment solver)
    if check_module("ortools"):
        print("    ✓ OR-Tools (min-cost-flow assignment solver, default)")
    else:
        print("    ✗ OR-Tools missing -- the default 'flow' solver is unavailable")

    # Deconvolution backends
    print("\n  [Deconvolution backends]")
    deconv_modules = [
        ("qpsolvers", "QPSolvers (RCTD, spatialDWLS)"),
        ("ray", "Ray (RCTD)"),
    ]
    for module, display_name in deconv_modules:
        status = "✓" if check_module(module) else "✗"
        print(f"    {status} {display_name}")

    print()
    print("=" * 70)

    # Try importing PanoSpace
    print("\nTesting PanoSpace import...")
    try:
        import panospace as ps

        print("  ✓ PanoSpace imported successfully")
        print(f"  Version: {getattr(ps, '__version__', '1.0.0')}")
    except ImportError as e:
        print(f"  ✗ Failed to import PanoSpace: {e}")
        all_passed = False

    print()
    print("=" * 70)

    if all_passed:
        print("✓ All core dependencies installed successfully!")
        print()
        print("You're ready to use PanoSpace!")
        print()
        print("Next steps:")
        print("  1. Check out the Quick Start guide in README.md")
        print("  2. Run the pipeline: python run_panospace.py --help")
        return 0
    else:
        print("✗ Some dependencies are missing. Please check the installation.")
        print()
        print("Troubleshooting:")
        print("  1. Make sure you activated the PanoSpace environment:")
        print("     conda activate PanoSpace")
        print("  2. Try reinstalling:")
        print("     pip install -e .")
        print("  3. For GPU issues, check your CUDA installation")
        return 1


if __name__ == "__main__":
    sys.exit(main())
