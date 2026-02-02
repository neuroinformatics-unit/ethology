(target-installation)=
# Installation

(conda)=
[conda](https://docs.conda.io/en/latest/)

(pip)=
[pip](https://pip.pypa.io/en/stable/)

(uv)=
[uv](https://docs.astral.sh/uv/)

To install `ethology`, we recommend using a virtual environment to avoid
dependency conflicts with other packages.
You can use {ref}`conda`, {ref}`pip`, or {ref}`uv` to create and manage this environment.

## Install the package

`````{tab-set}
````{tab-item} conda
From conda-forge using conda

First, create and activate a {ref}`conda` environment:
```sh
conda create -n ethology-env python=3.13 -y
conda activate ethology-env
```

Then install the package using pip:
```sh
pip install ethology
```
````

````{tab-item} pip
From PyPI using pip

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

````{tab-item} uv
From PyPI using uv

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

To update to the latest version of `ethology`:
```sh
pip install --upgrade ethology
```

To uninstall `ethology`, the simplest option is to delete the virtual environment that contains it. For {ref}`conda` environments, run from a different conda environment (e.g., `base`):
```sh
conda env remove -n ethology-env
```

:::{tip}
You can list all conda environments afterwards to verify removal:
```sh
conda env list
```
:::
