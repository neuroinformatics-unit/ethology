(target-installation)=
# Installation

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
## Uninstalling `ethology`
To Uninstall the package run the following command:
```sh
pip uninstall ethology
```
Deactivate/Delete the Environment
```sh
conda deactivate          # Exit the environment
conda env remove -n ethology-env  # Delete the environment
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
