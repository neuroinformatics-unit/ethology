# How to Contribute

## Preparing to contribute

### Create a development environment

We recommended using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) to create a
development environment. In the following, we assume you have
`conda` installed.

To install `ethology` for development, first create and activate a `conda` environment:

```sh
conda create -n ethology-dev python=3.13
conda activate ethology-dev
```

Then install the development version of `ethology` by cloning the GitHub repository,
and pip-installing it in editable mode with the required dependencies:

```sh
# clone the repository
git clone https://github.com/neuroinformatics-unit/ethology.git

# then run from within the repository root folder:
pip install -e .[dev]  # works on most shells
pip install -e '.[dev]'  # works on zsh (the default shell on macOS)
```
This should install all the dependencies needed for development, such as `pytest` and `pre-commit`.

Finally, install the [pre-commit hooks](https://pre-commit.com/):

```bash
pre-commit install
```

Pre-commit hooks are a set of tools that run automatically before each commit and help maintain a consistent formatting style in the code (see the section [pre-commit hooks](#formatting-and-pre-commit-hooks) below for more details).

### Pull requests

Please submit code to the main repository with a pull request (PR).
We follow our sister project [movement](https://github.com/neuroinformatics-unit/movement/blob/main/CONTRIBUTING.md) and adhere to the same conventions:

- Please submit _draft_ PRs as early as possible to allow for discussion.
- The PR title should be descriptive e.g. "Add new function to do X" or "Fix bug in Y".
- The PR description should be used to provide context and motivation for the changes - we use a template to help you with this.
- One approval of a PR (by a repository owner) is enough for it to be merged.
- Unless someone approves the PR with optional comments, the PR is immediately squash and merged by the approving reviewer.
- Ask for a review from someone specific if you think they would be a particularly suited reviewer.
- PRs are preferably merged via the ["squash and merge"](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/incorporating-changes-from-a-pull-request/about-pull-request-merges#squash-and-merge-your-commits) option, to keep a clean commit history on the _main_ branch.

### Contribution workflow
A typical contribution workflow would be as follows:
* Locally, check out a new branch, make your changes, and stage them.
* When you try to commit, the [pre-commit hooks](#formatting-and-pre-commit-hooks) will be triggered.
* Stage any fixes automatically made by the hooks, or fix any of them manually, and commit the changes.
* Make sure to add tests for any new features or bug fixes. See the [testing](#testing) section below.
* Don't forget to update the documentation, if necessary. This includes docstrings, README, and any other relevant documentation.
* Make sure that all tests pass locally before pushing your changes.
* Push your changes to GitHub and open a draft pull request, with a meaningful title and a thorough description of the changes.
* If all continuous integration (CI) checks run successfully (e.g. linting, type checking, testing), you may mark the pull request as ready for review and tag a reviewer.
* Once the reviewer has provided feedback, you will need to respond to the review comments and implement any requested changes.
* After the review feedback has been addressed, one of the repository maintainers will approve the PR and add it to the [merge queue](https://github.blog/changelog/2023-02-08-pull-request-merge-queue-public-beta/).
* Success 🎉 !! Your PR will be (squash-)merged into the _main_ branch.

## Contributing code
### Formatting and pre-commit hooks

Running `pre-commit install` will set up our [pre-commit hooks](https://pre-commit.com/) - these are useful to ensure a consistent formatting style. Currently, our hooks run:
* [ruff](https://github.com/astral-sh/ruff), which does a number of jobs, including code linting and auto-formatting.
* [mypy](https://mypy.readthedocs.io/en/stable/index.html), which does static type checking.
* [check-manifest](https://github.com/mgedmin/check-manifest), to ensure that the right files are included in the pip package.
* [codespell](https://github.com/codespell-project/codespell) to check for common misspellings.

You may also run the pre-commit hooks manually at any time using one of the following commands before committing:
```sh
pre-commit run  # applies to the staged files
pre-commit run -a  # applies to all files in the repository
```

Some problems will be automatically fixed by the hooks. In this case, you should stage the auto-fixed changes and run the hooks again to ensure that there are no further issues:
```sh
git add .  # stage all changes
pre-commit run
```

If a problem cannot be auto-fixed, the corresponding tool will provide
information on what the issue is and how to fix it. For example, `ruff` might
output something like:

```sh
ethology/io/load_bboxes.py:551:80: E501 Line too long (90 > 79)
```

This pinpoints the problem to a single code line and a specific [ruff rule](https://docs.astral.sh/ruff/rules/) violation.
Sometimes you may have good reasons to ignore a particular rule for a specific line of code. You can do this by adding an inline comment, e.g. `# noqa: E501`. Replace `E501` with the code of the rule you want to ignore.

For docstrings, we adhere to the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) style.
Make sure to provide docstrings for all public functions, classes, and methods.
This is important as it allows for automatic generation of the API reference.

### Testing

We use [pytest](https://docs.pytest.org/en/latest/) for testing and aim for
~100% test coverage (as far as is reasonable).
All new features should be tested.
Write your test methods and classes in the _tests_ folder.

For some tests, you will need to use real data.
Please do not add sample data to the code repository, especially if they are large - we use an external data repository instead.
See the [sample data](#test-data) section for more information.

### Continuous integration
All pushes and pull requests trigger a [GitHub actions](https://docs.github.com/en/actions) workflow. This is defined in the file at `.github/workflows/test_and_deploy.yml` and runs:
* Linting checks (pre-commit).
* Testing in different operating systems and Python versions (only if linting checks pass)
* A release to PyPI (only if a git tag is present and if tests pass).

## Contributing documentation

The documentation is hosted via [GitHub pages](https://pages.github.com/) at
[ethology.neuroinformatics.dev](target-ethology).
Its source files are located in the `docs` folder of this repository.
They are written in either [Markdown](myst-parser:syntax/typography.html)
or [reStructuredText](https://docutils.sourceforge.io/rst.html).
The `index.md` file corresponds to the homepage of the documentation website.
Other `.md`  or `.rst` files are linked to the homepage via the `toctree` directive.

We use [Sphinx](sphinx-doc:) and the [PyData Sphinx Theme](https://pydata-sphinx-theme.readthedocs.io/en/stable/index.html)
to build the source files into HTML output.
This is handled by a GitHub actions workflow (`.github/workflows/docs_build_and_deploy.yml`).
The build job is triggered on each PR, ensuring that the documentation build is not broken by new changes.
The deployment job is only triggered whenever a tag is pushed to the _main_ branch,
ensuring that the documentation is published in sync with each PyPI release.

### Editing the documentation

To edit the documentation, first clone the repository, and install `ethology` in a
[development environment](#create-a-development-environment).

Then, install a few additional dependencies in your development environment to be able to build the documentation locally. To do this, run the following command from the root of the repository:
```sh
pip install -e .[docs]    # works on most shells
pip install -e '.[docs]'  # works on zsh (default on macOS)
```

Now create a new branch, edit the documentation source files (`.md` or `.rst` in the `docs` folder),
and commit your changes. Submit your documentation changes via a pull request,
following the [same guidelines as for code changes](#pull-requests).
Make sure that the header levels in your `.md` or `.rst` files are incremented
consistently (H1 > H2 > H3, etc.) without skipping any levels.

### Adding new pages
If you create a new documentation source file (e.g. `my_new_file.md` or `my_new_file.rst`),
you will need to add it to the `toctree` directive in `index.md`
for it to be included in the documentation website:

```rst
:maxdepth: 2
:hidden:

existing_file
my_new_file
```

### Linking to external URLs
If you are adding references to an external URL (e.g. `https://github.com/neuroinformatics-unit/ethology/issues/1`) in a `.md` file, you will need to check if a matching URL scheme (e.g. `https://github.com/neuroinformatics-unit/ethology/`) is defined in `myst_url_schemes` in `docs/source/conf.py`. If it is, the following `[](scheme:loc)` syntax will be converted to the [full URL](ethology-github:issues/1) during the build process:
```markdown
[link text](ethology-github:issues/1)
```

If it is not yet defined and you have multiple external URLs pointing to the same base URL, you will need to [add the URL scheme](myst-parser:syntax/cross-referencing.html#customising-external-url-resolution) to `myst_url_schemes` in `docs/source/conf.py`.

### Updating the API reference
The [API reference](target-api) is auto-generated by the `docs/make_api_index.py` script, and the [sphinx-autodoc](sphinx-doc:extensions/autodoc.html) and [sphinx-autosummary](sphinx-doc:extensions/autosummary.html) extensions.
The script generates the `docs/source/api_index.rst` file containing the list of modules to be included in the [API reference](target-api).
The plugins then generate the API reference pages for each module listed in `api_index.rst`, based on the docstrings in the source code.
So make sure that all your public functions/classes/methods have valid docstrings following the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) style.
Our `pre-commit` hooks include some checks (`ruff` rules) that ensure the docstrings are formatted consistently.

If your PR introduces new modules that should *not* be documented in the [API reference](target-api), or if there are changes to existing modules that necessitate their removal from the documentation, make sure to update the `exclude_modules` list within the `docs/make_api_index.py` script to reflect these exclusions.

### Updating the examples
We use [sphinx-gallery](sphinx-gallery:)
to create the [examples](target-examples).
To add new examples, you will need to create a new `.py` file in `examples/`,
or in `examples/advanced/` if your example targets experienced users.
The file should be structured as specified in the relevant
[sphinx-gallery documentation](sphinx-gallery:syntax).

We are using sphinx-gallery's [integration with binder](sphinx-gallery:configuration#binder-links)
to provide interactive versions of the examples.
If your examples rely on packages that are not among `movement`'s dependencies,
you will need to add them to the `docs/source/environment.yml` file.
That file is used by binder to create the conda environment in which the
examples are run. See the relevant section of the
[binder documentation](https://mybinder.readthedocs.io/en/latest/using/config_files.html).

### Cross-referencing Python objects
:::{note}
Docstrings in the `.py` files for the [API reference](target-api)  are converted into `.rst` files, so these should use reStructuredText syntax.
:::

#### Internal references
::::{tab-set}
:::{tab-item} Markdown
For referencing ethology objects in `.md` files, use the `` {role}`target` `` syntax with the appropriate [Python object role](sphinx-doc:domains/python.html#cross-referencing-python-objects).

For example, to reference the {mod}`ethology.io.annotations.load_bboxes` module, use:
```markdown
{mod}`ethology.io.annotations.load_bboxes`
```
:::
:::{tab-item} RestructuredText
For referencing ethology objects in `.rst` files, use the `` :role:`target` `` syntax with the appropriate [Python object role](sphinx-doc:domains/python.html#cross-referencing-python-objects).

For example, to reference the {mod}`ethology.io.annotations.load_bboxes` module, use:
```rst
:mod:`ethology.io.annotations.load_bboxes`
```
:::
::::

#### External references
For referencing external Python objects using [intersphinx](sphinx-doc:extensions/intersphinx.html),
ensure the mapping between module names and their documentation URLs is defined in [`intersphinx_mapping`](sphinx-doc:extensions/intersphinx.html#confval-intersphinx_mapping) in `docs/source/conf.py`.
Once the module is included in the mapping, use the same syntax as for [internal references](#internal-references).

::::{tab-set}
:::{tab-item} Markdown
For example, to reference the {meth}`xarray.Dataset.update` method, use:
```markdown
{meth}`xarray.Dataset.update`
```
:::

:::{tab-item} RestructuredText
For example, to reference the {meth}`xarray.Dataset.update` method, use:
```rst
:meth:`xarray.Dataset.update`
```
:::
::::

### Building the documentation locally
We recommend that you build and view the documentation website locally, before you push your proposed changes.

First, ensure your development environment with the required dependencies is active (see [Editing the documentation](#editing-the-documentation) for details on how to create it). Then, navigate to the `docs/` directory:
```sh
cd docs
```
All subsequent commands should be run from this directory.

:::{note}
Windows PowerShell users should prepend `make` commands with `.\` (e.g. `.\make html`).
:::

To build the documentation, run:

```sh
make html
```
The local build can be viewed by opening `docs/build/html/index.html` in a browser.

To re-build the documentation after making changes,
we recommend removing existing build files first.
The following command will remove all generated files in `docs/`,
including the auto-generated files `source/api_index.rst`, as well as all files in
 `build/`, `source/api/`.
 It will then re-build the documentation:

```sh
make clean html
```
Or alternatively:
```sh
make clean && make html
```
To check that external links are correctly resolved, run:

```sh
make linkcheck
```

If the linkcheck step incorrectly marks links with valid anchors as broken, you can skip checking the anchors in specific links by adding the URLs to `linkcheck_anchors_ignore_for_url` in `docs/source/conf.py`, e.g.:

```python
# The linkcheck builder will skip verifying that anchors exist when checking
# these URLs
linkcheck_anchors_ignore_for_url = [
    "https://gin.g-node.org/G-Node/Info/wiki/",
    "https://neuroinformatics.zulipchat.com/",
]
```

:::{tip}
The `make` commands can be combined to run multiple tasks sequentially.
For example, to re-build the documentation and check the links, run:
```sh
make clean html linkcheck
```
:::

We use [sphinx-gallery](sphinx-gallery:)
to create the [examples](target-examples).
To add new examples, you will need to create a new `.py` file in `examples/`.
The file should be structured as specified in the relevant
[sphinx-gallery documentation](sphinx-gallery:syntax).

We are using sphinx-gallery's [integration with binder](sphinx-gallery:configuration#binder-links), to provide interactive versions of the examples.
This is configured in `docs/source/conf.py` under the `sphinx_gallery_conf` variable,
and further customised for our repository by the `.binder/postBuild` script.
If your examples rely on packages that are not among `movement`'s dependencies,
you will need to add them to the `.binder/requirements.txt` file.

## Test data

We maintain some sample datasets to be used for testing on an
[external data repository](https://gin.g-node.org/neuroinformatics/ethology-test-data).
Our hosting platform of choice is called [GIN](https://gin.g-node.org).
GIN has a GitHub-like interface and git-like [CLI](https://gin.g-node.org/G-Node/Info/wiki/) functionalities.

Please refer to the [README](https://gin.g-node.org/neuroinformatics/ethology-test-data/src/master/README.md) in the data repository for more information on the datasets and how to access them.

### Fetching data
To fetch the data from GIN, we use the [pooch](https://www.fatiando.org/pooch/latest/index.html)
Python package, which can download data from pre-specified URLs and store them
locally for subsequent uses. It also provides some nice utilities,
like verification of sha256 hashes and decompression of archives.

### Adding new data
Only core `ethology` developers may add new files to the external data repository.
To add a new file, you will need to:

1. Create a [GIN](https://gin.g-node.org/) account
2. Ask to be added as a collaborator on the [ethology data repository](https://gin.g-node.org/neuroinformatics/ethology-test-data) (if not already)
3. Download the [GIN CLI](https://gin.g-node.org/G-Node/Info/wiki/GIN+CLI+Setup#quickstart) and set it up with your GIN credentials, by running `gin login` in a terminal.
4. Clone the `ethology` data repository to your local machine, by running `gin get neuroinformatics/ethology-test-data` in a terminal.
5. Add your new files as appropriate. Please follow the instructions in the [README](https://gin.g-node.org/neuroinformatics/ethology-test-data/src/master/README.md) and the existing file naming conventions as closely as possible.
6. Determine the sha256 checksum hash of each new file. You can do this in a terminal by running:

    ::::{tab-set}
    :::{tab-item} Ubuntu
    ```bash
    sha256sum <filename>
    ```
    :::

    :::{tab-item} MacOS
    ```bash
    shasum -a 256 <filename>
    ```
    :::

    :::{tab-item} Windows
    ```bash
    certutil -hashfile <filename> SHA256
    ```
    :::
    ::::

    For convenience, we've included a `get_sha256_hashes.py` script in the [ethology data repository](https://gin.g-node.org/neuroinformatics/ethology-test-data). If you run this from the root of the data repository, within a Python environment with `ethology` installed, it will calculate the sha256 hashes for all files in the `test_data` folder and write them to the file named `files-registry.txt`.

7. Commit a specific file with `gin commit -m <message> <filename>`, or `gin commit -m <message> .` to commit all changes.

9. Upload the committed changes to the GIN repository by running `gin upload`. Latest changes to the repository can be pulled via `gin download`. `gin sync` will synchronise the latest changes bidirectionally.

## Versioning and releases

We use [semantic versioning](https://semver.org/), which includes `MAJOR`.`MINOR`.`PATCH` version numbers:

* PATCH = small bugfix
* MINOR = new feature
* MAJOR = breaking change

We use [setuptools_scm](https://setuptools-scm.readthedocs.io/en/latest/) to automatically version `ethology`.
It has been pre-configured in the `pyproject.toml` file.
`setuptools_scm` will automatically [infer the version using git](https://setuptools-scm.readthedocs.io/en/latest/usage#default-versioning-scheme).
To manually set a new semantic version, create a tag and make sure the tag is pushed to GitHub.
Make sure you commit any changes you wish to be included in this version. E.g. to bump the version to `1.0.0`:

```sh
git add .
git commit -m "Add new changes"
git tag -a v1.0.0 -m "Bump to version 1.0.0"
git push --follow-tags
```
Alternatively, you can also use the GitHub web interface to create a new release and tag.

The addition of a GitHub tag triggers the package's deployment to PyPI (see the [continuous integration](#continuous-integration) section).
The version number is automatically determined from the latest tag on the _main_ branch.
