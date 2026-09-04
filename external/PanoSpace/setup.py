from __future__ import annotations

from pathlib import Path

from setuptools import find_packages
from setuptools import setup

PACKAGE_NAME = "panospace"
REPO_URL = "https://github.com/hehuifeng/PanoSpace-core"
BASE_DIR = Path(__file__).parent.resolve()
README = (BASE_DIR / "README.md").read_text(encoding="utf8")

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
VERSION = "0.2.1"

# ---------------------------------------------------------------------------
# Dependency groups
# ---------------------------------------------------------------------------
BASE_REQUIREMENTS = [
    "numpy>=1.23",
    "pandas>=1.5",
    "anndata>=0.10",
    "scanpy>=1.9",
    "python-igraph>=0.10",  # Required for Leiden clustering in spatialDWLS
    "leidenalg>=0.9",  # Leiden clustering algorithm
    "scipy>=1.9",
    "scikit-learn>=1.2",
    "tqdm>=4.65",
    "Pillow>=9.4",
    "opencv-python>=4.8",
    "matplotlib>=3.7",
    "scikit-image>=0.22",
    "requests>=2.31",
    "ray>=2.7",
    "pot>=0.9",
    "qpsolvers>=4.0",
    "osqp",  # QP backend requested by RCTD / spatialDWLS
]

CELLVIT_REQUIREMENTS = [
    "torch>=2.0",
    "torchvision>=0.15",
    "einops>=0.6",  # tensor operations
    "shapely>=2.0",  # geometry operations for postprocessing
]

ANNOTATION_REQUIREMENTS = [
    "torch>=2.0",  # Required by pytorch-lightning and pyro-ppl
    "pyro-ppl>=1.8",
    "pytorch-lightning>=2.1",
    "lightning>=2.1",
    "transformers>=4.33",
    "scvi-tools>=1.0",  # for cell2location backend
    "ortools>=9.7",  # min-cost-flow assignment solver (default)
    "pyscipopt",  # SCIP solver (open-source, MILP)
    "gurobipy>=10",  # optional commercial solver (free academic license available)
]

EXTRAS_REQUIRE = {
    "cellvit": CELLVIT_REQUIREMENTS,
    "annotation": ANNOTATION_REQUIREMENTS,
}

ALL_OPTIONAL = sorted(set(CELLVIT_REQUIREMENTS + ANNOTATION_REQUIREMENTS))
EXTRAS_REQUIRE["all"] = ALL_OPTIONAL

INSTALL_REQUIRES = BASE_REQUIREMENTS

# ---------------------------------------------------------------------------
# Setup metadata
# ---------------------------------------------------------------------------
setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description="High-resolution spatial transcriptomics analysis toolkit.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Hui-Feng He",
    author_email="huifeng@mails.ccnu.edu.cn",
    url=REPO_URL,
    project_urls={
        "Documentation": REPO_URL,
        "Source": REPO_URL,
        "Tracker": f"{REPO_URL}/issues",
    },
    license="MIT",
    packages=find_packages(exclude=("tests", "tests.*")),
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    keywords=[
        "spatial transcriptomics",
        "single-cell",
        "deep learning",
        "bioinformatics",
    ],
)
