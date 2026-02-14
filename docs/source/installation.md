(target-installation)=
# Installation

To avoid dependency conflicts with other packages, it is best practice to install Python packages within a virtual environment.
We recommend using [conda](conda:) or [uv](uv:getting-started/installation/) to create and manage this environment, as they simplify the installation process.

The following instructions assume that you have either conda or uv installed. If you don't please check their installation instructions (for [conda](conda:docs/getting-started/miniconda/main), for [uv](uv:getting-started/installation/)).

## Install the package

`````{tab-set}
````{tab-item} From conda-forge using conda
First, create and activate a [conda](conda:) environment:
```sh
conda create -n ethology-env python=3.13 -y
conda activate ethology-env
```

Then install the package using pip:
```sh
pip install ethology
```
````

````{tab-item} From PyPI using pip
First, create and activate a virtual environment:
```sh
python -m venv ethology-env
```

On Windows:
```sh
ethology-env\Scripts\activate
```

On macOS/Linux:
```sh
source ethology-env/bin/activate
```

Then install the package:
```sh
pip install ethology
```
````

````{tab-item} From PyPI using uv
First, create and activate a virtual environment:
```sh
uv venv ethology-env
```

On Windows:
```sh
ethology-env\Scripts\activate
```

On macOS/Linux:
```sh
source ethology-env/bin/activate
```

Then install the package:
```sh
uv pip install ethology
```
````
`````

### Developers
If you are a developer looking to contribute to ethology, please refer to our [contributing guide](community/contributing.rst) for detailed setup instructions and guidelines.

## Update the package

Always update using the same package manager used for installation (either via `pip` or via `uv`).

To update to the latest version of `ethology`:
```sh
pip install --upgrade ethology
```

If the above fails, try installing `ethology` in a fresh new environment to avoid dependency conflicts. If you wish to use the same name for your new environment, you may wish to remove the existing environment first:
```sh
conda env remove -n ethology-env
```

:::{tip}
You can list all conda environments afterwards to verify removal:
```sh
conda env list
```
:::
