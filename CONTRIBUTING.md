# Contributing guidelines

## Setting up a local development environment

Install [poetry](https://python-poetry.org/) following [its installation instrucutions](https://python-poetry.org/docs/) - it's recommended to install using `pipx`.

Clone this repo and `cd` into it.

Install ReX and its dependencies by running `poetry install`.
This will install the versions of the dependencies given in `poetry.lock`.
This ensures that the development environment is consistent for different people.

The development dependencies (e.g. for generating documentation) are marked as optional, so to install these you will need to run instead `poetry install --with docs`.

N.B. that poetry by default creates its own virtual environment for the project.
However if you run `poetry install` in an activated virtual environment, it will detect and respect this.
See the [poetry docs](https://python-poetry.org/docs/basic-usage/#using-your-virtual-environment) for more information.

## Generating documentation with Sphinx

To generate documentation using Sphinx and sphinx-autoapi:

```sh
cd docs/
make html
```

This will automatically generate documentation based on the code and docstrings and produce html files in `docs/_build/html`.

### Docstring style

We prefer [Google-style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) docstrings, and use `sphinx.ext.napoleon` to parse them.

## TODO

- linting with ruff
- running tests
- test coverage
