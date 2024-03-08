#!/usr/bin/env python
import importlib.util
from pathlib import Path

from setuptools import setup, find_packages

spec = importlib.util.spec_from_file_location(
    "fz_td_recipe.version",
    "fz_td_recipe/version.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
VERSION = module.__version__

setup(
    name="fz-td-recipe",
    author="bbp-ou-hpc",
    author_email="bbp-ou-hpc@groupes.epfl.ch",
    version=VERSION,
    description="Python package to read and modify the definitions and parameters used in circuit building.",
    long_description=Path("README.rst").read_text(encoding="utf-8"),
    long_description_content_type="text/x-rst",
    url="https://bbpteam.epfl.ch/documentation/projects/fz-td-recipe",
    project_urls={
        "Tracker": "https://bbpteam.epfl.ch/project/issues/projects/FUNCZ/issues",
        "Source": "git@bbpgitlab.epfl.ch:hpc/circuit-building/fz-td-recipe.git",
    },
    license="BBP-internal-confidential",
    install_requires=[
        "click",
        "jsonschema",
        "libsonata",
        "lxml<5",
        "numpy",
        "pandas[pyarrow]",
        "pyyaml",
    ],
    packages=find_packages(),
    package_data={
        "fz_td_recipe": ["data/*.yaml"],
    },
    python_requires=">=3.9",
    extras_require={"docs": ["sphinx", "sphinx-bluebrain-theme"]},
    entry_points={"console_scripts": ["fz-td-recipe=fz_td_recipe.cli:app"]},
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
