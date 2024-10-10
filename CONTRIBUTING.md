# Contributing guidelines

## Setting up a local development environment

Install [poetry](https://python-poetry.org/) following [its installation instrucutions](https://python-poetry.org/docs/) - it's recommended to install using `pipx`.

Clone this repo and `cd` into it.

Install ReX and its dependencies by running `poetry install`.
This will install the versions of the dependencies given in `poetry.lock`.
This ensures that the development environment is consistent for different people.

The development dependencies (for generating documentation, linting, and running tests) are marked as optional, so to install these you will need to run instead `poetry install --with dev`.

N.B. that poetry by default creates its own virtual environment for the project.
However if you run `poetry install` in an activated virtual environment, it will detect and respect this.
See the [poetry docs](https://python-poetry.org/docs/basic-usage/#using-your-virtual-environment) for more information.

## Testing

We use [pytest](https://docs.pytest.org/en/stable/index.html).

Run the tests by running `pytest`, which will automatically run all files of the form `test_*.py` or `*_test.py` in the current directory and its subdirectories.

## Generating documentation with Sphinx

Docs are automatically built on PRs, and are available at <http://rex-xai.readthedocs.io/>. 

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

## TODO

- linting with ruff
- running tests
- test coverage
