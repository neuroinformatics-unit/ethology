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
