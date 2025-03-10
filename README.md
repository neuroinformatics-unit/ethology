[![Python Version](https://img.shields.io/pypi/pyversions/ethology.svg)](https://pypi.org/project/ethology)
[![PyPI Version](https://img.shields.io/pypi/v/ethology.svg)](https://pypi.org/project/ethology)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-orange.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![CI](https://img.shields.io/github/actions/workflow/status/neuroinformatics-unit/ethology/test_and_deploy.yml?label=CI)](https://github.com/neuroinformatics-unit/ethology/actions)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/format.json)](https://github.com/astral-sh/ruff)
[![codecov](https://codecov.io/gh/neuroinformatics-unit/ethology/branch/main/graph/badge.svg?token=P8CCH3TI8K)](https://codecov.io/gh/neuroinformatics-unit/ethology)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

# ethology

## Installation

First clone the repository at the desired location:

```bash
git clone https://github.com/neuroinformatics-unit/ethology.git
```

Then create a conda environment and install the package from source

```
conda create -n ethology-env python=3.12 -y
conda activate ethology-env
cd ethology
pip install .
```

To install the package in editable mode with developer dependencies, replace the last command with:

```
pip install -e .[dev]  # in mac: pip install -e ".[dev]"
```

---

Using cotracker requires running the following in terminal

```bash
git submodule init
git submodule update
```
