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

:::{tip}
If you are unsure about the environment name, you can get a list of the environments on your system with:
```sh
conda env list
```
:::
Once the environment has been removed, you can create a new one following the [installation instructions](#installation) above.

## Updating the Package

To update to the latest version of `ethology`:
```sh
cd ethology
git checkout main      # Ensure you're on the main branch
git pull               # Fetch latest changes
```

Remove the `ethology-env` Conda environment:
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
