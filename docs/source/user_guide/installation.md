(target-installation)=
# Installation

## Install ethology
:::{admonition} Use a conda environment
:class: note
To avoid dependency conflicts with other packages, it is best practice to install Python packages within a virtual environment.
We recommend using [conda](conda:) to create and manage this environment, as they simplify the installation process.
The following instructions assume that you have conda installed.
:::

## Users

First clone the repository at the desired location:

```bash
git clone https://github.com/neuroinformatics-unit/ethology.git
```

Then create a conda environment and install the package from source
```sh
conda create -n ethology-env python=3.12 -y
conda activate ethology-env
cd ethology
pip install .
```

To install the package in editable mode with developer dependencies, replace the last command with:

```sh
pip install -e .[dev]

# in mac:

pip install -e ".[dev]"
```

### Developers
If you are a developer looking to contribute to ethology, please refer to our [contributing guide](contributing.md) for detailed setup instructions and guidelines.

To uninstall an existing environment named `ethology-env`:
```sh
conda env remove -n ethology-env
```
:::{tip}
If you are unsure about the environment name, you can get a list of the environments on your system with:
```sh
conda env list
```
:::
Once the environment has been removed, you can create a new one following the [installation instructions](#install-ethology) above.
