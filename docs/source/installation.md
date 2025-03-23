(target-installation)=
# Installation

:::{admonition} Use a conda environment
:class: note
To avoid dependency conflicts with other packages, it is best practice to install Python packages within a virtual environment.
We recommend using [conda](conda:) to create and manage this environment, as they simplify the installation process.
The following instructions assume that you have conda installed.
:::

:::{warning}
🏗️ pip and conda installation available soon!
:::

## Install the package

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

## Developers
If you are a developer looking to contribute to ethology, please refer to our [contributing guide](community/contributing.rst) for detailed setup instructions and guidelines.

## Updating the Package

To update to the latest version of `ethology`:
```sh
cd ethology
git checkout main      # Ensure you're on the main branch
git fetch               # Fetch latest changes
git pull
```

Remove the old `ethology-env` Conda environment:
```sh
conda env remove -n ethology-env
```

:::{tip}
List all Conda environments to verify removal:
```sh
conda env list
```
:::

Reinstall the package:
```sh
pip install .
```
