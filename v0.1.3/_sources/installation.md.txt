(target-installation)=
# Installation

The following instructions assume that you have either conda or uv installed. If you don't please check their installation instructions (for [conda](conda:docs/getting-started/miniconda/main), for [uv](uv:getting-started/installation/)).

## Create and activate a virtual environment
To avoid dependency conflicts with other packages, it is best practice to install Python packages within a virtual environment.
We recommend using [conda](conda:) or [uv](uv:getting-started/installation/) to create and manage this environment, as they simplify the installation process.

::::{tab-set}
:::{tab-item} conda
Create and activate a new [conda environment](conda:user-guide/tasks/manage-environments.html):
```sh
# Create conda environment
conda create -y -n ethology-env -c conda-forge python=3.13

# Activate it
conda activate ethology-env
```

We used `ethology-env` as the environment name, but you can choose any name you prefer.
:::

:::{tab-item} uv
Create and activate a new [virtual environment](uv:pip/environments/) inside your project directory:

```sh
# Create virtual environment
uv venv --python=3.13

# On macOS and Linux, activate it with:
source .venv/bin/activate

# On Windows PowerShell, activate it with:
.venv\Scripts\activate
```
:::
::::

## Install the package
With your environment activated, install `ethology` using one of the methods below.

::::{tab-set}
:::{tab-item} From PyPI using pip
Install the core package:
```sh
pip install ethology
```
:::

:::{tab-item} From PyPI using uv
Install the core package:
```sh
uv pip install ethology
```
:::

::::


:::{admonition} For developers
:class: tip

If you would like to contribute to `ethology`, see our [contributing guide](community/contributing.rst)
for detailed developer setup instructions and coding guidelines.
:::


## Update the package

Always update using the same package manager used for installation (either via `pip` or via `uv`).

To update to the latest version of `ethology`:
::::{tab-set}
:::{tab-item} pip
```sh
pip install -U ethology
```
:::

:::{tab-item} uv
```sh
uv pip install -U ethology
```
:::
::::

If the above fails, try installing `ethology` in a fresh new environment. To do this, first remove the existing environment:
::::{tab-set}
:::{tab-item} conda
```sh
conda env remove -n ethology-env
```

:::

:::{tab-item} uv
Delete the `.venv` folder in your project directory.

```powershell
# On macOS and Linux, run:
rm -rf .venv

# On Windows PowerShell, run:
rmdir /s /q .venv
```

Optionally, you can clean the `uv` cache for unused packages:
```sh
uv cache prune
```
:::
::::


Then you can create a new environment following the [instructions](#create-and-activate-a-virtual-environment) above.

:::{tip}
You can list all conda environments before and after the above command to verify removal:
```sh
conda env list
```
:::
