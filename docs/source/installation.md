(target-installation)=
# Installation

To avoid dependency conflicts with other packages, it is best practice to install Python packages within a virtual environment.
We recommend using [conda](conda:) to create and manage this environment, as they simplify the installation process.
The following instructions assume that you have conda installed.

## Install the package

:::{warning}
🏗️ pip and conda installation available soon!
:::


### Users
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

### Developers
If you are a developer looking to contribute to ethology, please refer to our [contributing guide](community/contributing.rst) for detailed setup instructions and guidelines.

## Update the package

To update to the latest version of `ethology`:
```sh
cd ethology
git checkout main      # Ensure you're on the main branch
git fetch               # Fetch latest changes
git pull
pip install .
```

To uninstall `ethology`, the simplest option is to delete the conda environment that contains it. To do so, run from a different conda environment (e.g., `base`):
```sh
conda env remove -n ethology-env
```

:::{tip}
You can list all conda environments afterwards to verify removal:
```sh
conda env list
```
:::
