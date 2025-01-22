# Contributing guidelines

## Setting up a local development environment

While you can install ReX through various methods, for development purposes we use [poetry](https://python-poetry.org/) for installation, development dependency management, and building the package. Install poetry following [its installation instructions](https://python-poetry.org/docs/) - it's recommended to install using `pipx`.

Clone this repo and `cd` into it.

Install ReX and its dependencies by running `poetry install`.
This will install the versions of the dependencies given in `poetry.lock`.
This ensures that the development environment is consistent for different people.

The development dependencies (for generating documentation, linting, and running tests) are marked as optional, so to install these you will need to run instead `poetry install --with dev`.

N.B. that poetry by default creates its own virtual environment for the project.
However if you run `poetry install` in an activated virtual environment, it will detect and respect this.
See the [poetry docs](https://python-poetry.org/docs/basic-usage/#using-your-virtual-environment) for more information.

## Testing

We use [pytest](https://docs.pytest.org/en/stable/index.html) with the [pytest-cov](https://github.com/pytest-dev/pytest-cov) plugin.

Run the tests by running `pytest`, which will automatically run all files of the form `test_*.py` or `*_test.py` in the current directory and its subdirectories.
Run `pytest --cov=rex_xai tests/` to get a coverage report printed to the terminal.
See [pytest-cov's documentation](https://pytest-cov.readthedocs.io/en/latest/) for additional reporting options.

As the end-to-end tests which run the whole ReX pipeline can take a while to run, we have split the tests into two sub-directories: `tests/unit_tests/` and `tests/long_tests/`.
During development you may wish to only run the faster unit tests.
You can do this by specifying the directory: `pytest tests/unit_tests/`
Both sets of tests are run by GitHub Actions upon a pull request.

### Updating snapshots

Most of the end-to-end tests which run the whole ReX pipeline are 'snapshot tests' using the [syrupy](https://github.com/syrupy-project/syrupy) package.
These tests involve comparing an object returned by the function under test to a previously saved 'snapshot' of that object.
This can help identify unintentional changes in results that are introduced by new development.
Note that snapshots are based on the text representation of an object, so don't necessarily capture *all* results you may care about.

If a snapshot test fails, follow the steps below to confirm if the changes are expected or not and update the snapshots:

* Run `pytest -vv` to see a detailed comparison of changes compared to the snapshot
* Check whether these are expected or not. For example, if you have added an additional parameter in the `CausalArgs` class, you expect that parameter value to be missing from the snapshot.
* If you only see expected differences in the snapshot, you can update the snapshots to match the new results by running `pytest --snapshot-update` and commit the updated files.

## Generating documentation with Sphinx

Docs are automatically built on PRs and on updates to the repo's default branch, and are available at <http://rex-xai.readthedocs.io/>.

To build documentation locally using Sphinx and sphinx-autoapi:

```sh
cd docs/
make html
```

This will automatically generate documentation based on the code and docstrings and produce html files in `docs/_build/html`.

### Docstring style

We prefer [Google-style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) docstrings, and use `sphinx.ext.napoleon` to parse them.

## Code linting and formatting

This project uses [ruff](https://docs.astral.sh/ruff/) for code linting and formatting, to ensure a consistent code style and identify issues like unused imports.
Install by running `poetry install --with dev`.

Run the linter on all files in the current working directory with `ruff check`.
Ruff can automatically fix some issues if you run `ruff check --fix`.

Run `ruff format` to automatically format all files in the current working directory.
Run `ruff format --diff` to get a preview of any changes that would be made.

## Type checking

We use [Pyright](https://microsoft.github.io/pyright/#/) for type checking.
You can [install](https://microsoft.github.io/pyright/#/installation) the command line tool and/or an extension for your favourite editor.
Upon a pull request, a check is run that compares the number of errors and warnings from Pyright in the branches being compared in the PR.
Ideally, the number of errors/warnings will not increase!
