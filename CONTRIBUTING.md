# How to Contribute

## Creating a development environment

We recommended to use [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) to create a
development environment for `ethology`. In the following, we assume you have
`conda` installed.

To install `ethology` for development, first create and activate a `conda` environment:

```sh
conda create -n ethology-dev python=3.12
conda activate ethology-dev
```

Then install the development version of `ethology` by cloning the GitHub repository,
and pip-installing it in editable mode with the required dependencies:

```sh
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

## Pull requests

In all cases, please submit code to the main repository via a pull request (PR).
We follow our sister project [movement](https://github.com/neuroinformatics-unit/movement/blob/main/CONTRIBUTING.md) and adhere to the same conventions:

- Please submit _draft_ PRs as early as possible to allow for discussion.
- The PR title should be descriptive e.g. "Add new function to do X" or "Fix bug in Y".
- The PR description should be used to provide context and motivation for the changes - we use a template to help you with this.
- One approval of a PR (by a repository owner) is enough for it to be merged.
- Unless someone approves the PR with optional comments, the PR is immediately squash and merged by the approving reviewer.
- Ask for a review from someone specific if you think they would be a particularly suited reviewer.
- PRs are preferably merged via the ["squash and merge"](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/incorporating-changes-from-a-pull-request/about-pull-request-merges#squash-and-merge-your-commits) option, to keep a clean commit history on the _main_ branch.

## Contribution workflow
A typical contribution workflow would be as follows:
* Locally, check out a new branch, make your changes, and stage them.
* When you try to commit, the [pre-commit hooks](https://github.com/neuroinformatics-unit/movement/blob/main/CONTRIBUTING.md#formatting-and-pre-commit-hooks) will be triggered.
* Stage any fixes automatically made by the hooks, or fix any of them manually, and commit the changes.
* Make sure to add tests for any new features or bug fixes. See the [testing](#testing) section below.
* Don't forget to update the documentation, if necessary. This includes docstrings, README, and any other relevant documentation.
* Make sure that all tests pass locally before pushing your changes.
* Push your changes to GitHub and open a draft pull request, with a meaningful title and a thorough description of the changes.
* If all continuous integration (CI) checks run successfully (e.g. linting, type checking, testing), you may mark the pull request as ready for review and tag a reviewer.
* Once the reviewer has provided feedback, you will need to respond to the review comments and implement any requested changes.
* After the review feedback has been addressed, one of the repository maintainers will approve the PR and add it to the [merge queue](https://github.blog/changelog/2023-02-08-pull-request-merge-queue-public-beta/).
* Success 🎉 !! Your PR will be (squash-)merged into the _main_ branch.


## Formatting and pre-commit hooks

Running `pre-commit install` will set up our [pre-commit hooks](https://pre-commit.com/) to ensure a consistent formatting style. Currently, these include:
* running [ruff](https://github.com/astral-sh/ruff), which does a number of jobs, including code linting and auto-formatting.
* running [mypy](https://mypy.readthedocs.io/en/stable/index.html), which does static type checking.
* running [check-manifest](https://github.com/mgedmin/check-manifest), to ensure that the right files are included in the pip package.
* ruunning [codespell](https://github.com/codespell-project/codespell) to check for common misspellings.

You may run the pre-commit hooks manually at any time using one of the following commands before committing:
```sh
pre-commit run  # applies to the staged files
pre-commit run -a  # applies to all files in the repository
```

To run the pre-commit hooks individually, you can run the following from the root of the repository:
```sh
ruff .
mypy -p movement
check-manifest
codespell
```

Some problems will be automatically fixed by the hooks. In this case, you should stage the auto-fixed changes and run the hooks again to ensure that there are no further issues:
```sh
git add .
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

## Testing

We use [pytest](https://docs.pytest.org/en/latest/) for testing and aim for
~100% test coverage (as far as is reasonable).
All new features should be tested.
Write your test methods and classes in the _tests_ folder.

For some tests, you will need to use real data.
Please do not add sample data to the code repository, especially if they are large - we use an external data repository instead.
See the [sample data](#sample-data) section for more information.


## Continuous integration
All pushes and pull requests trigger a [GitHub actions](https://docs.github.com/en/actions) workflow. This is defined in the file at `.github/workflows/test_and_deploy.yml` and runs:
* Linting checks (pre-commit).
* Testing in different operating systems and Python versions (only if linting checks pass)
* A release to PyPI (only if a git tag is present and if tests pass).

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

The addition of a GitHub tag triggers the package's deployment to PyPI.
The version number is automatically determined from the latest tag on the _main_ branch.

## Test data

We maintain some sample datasets to be used for testing on an
[external data repository](https://gin.g-node.org/neuroinformatics/ethology-test-data).
Our hosting platform of choice is called [GIN](https://gin.g-node.org).
GIN has a GitHub-like interface and git-like
[CLI](https://gin.g-node.org/G-Node/Info/wiki#how-do-i-start) functionalities.

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
4. Clone the movement data repository to your local machine, by running `gin get neuroinformatics/ethology-test-data` in a terminal.
5. Add your new files as appropriate. Please follow the instructions in the README and the existing file naming conventions as closely as possible.
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
