# Contributing guidelines

## Setting up a local development environment

While you can install ReX through various methods, for development purposes we use [poetry](https://python-poetry.org/) for installation, development dependency management, and building the package. Install poetry following [its installation instructions](https://python-poetry.org/docs/) - it's recommended to install using `pipx`.

Clone this repo and `cd` into it.

Install ReX and its dependencies by running `poetry install`.
This will install the versions of the dependencies given in `poetry.lock`.
This ensures that the development environment is consistent for different people.

The development dependencies (for generating documentation, linting, and running tests) are marked as optional, so to install these you will need to run instead `poetry install --with dev`.

There are also some additional optional dependencies that are only required for working with 3D data.
You can install these using `poetry install --extras 3D`.

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

### Working with notebooks

The [introduction to working with ReX interactively](https://rex-xai.readthedocs.io/en/latest/notebooks/intro.html) is written as a Jupyter notebook in markdown format.
We use [MyST-NB](https://myst-nb.readthedocs.io/en/latest/index.html) to compile the notebook into html as part of the documentation.
To more easily work with the notebook locally, you can use [Jupytext](https://jupytext.readthedocs.io/en/latest/) to generate an .ipynb notebook from the markdown file, edit the tutorial, and then convert the edited notebook back into md.

```sh
# convert to .ipynb
jupytext docs/notebooks/intro.md --to ipynb
# convert back to markdown
jupytext docs/notebooks/intro.ipynb --to myst
```

Markdown format allows much clearer diffs when tracking the notebook with version control, so please don't add the .ipynb files to version control.

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
Upon a pull request, a check is run that identifies Pyright errors/warnings in the lines that have been added in the PR.
A review comment will be left for each change.
Ideally, no new errors/warnings will be introduced in a PR, but this is not an enforced requirement to merge.

## GitHub Actions

We use GitHub Actions to automatically run certain checks upon pull requests, and to automate releasing a new ReX version to PyPI.

On a pull request, the following workflows run:

* linting and type checking
* installing the package and running tests (using Python 3.10 and 3.13)
  * test coverage is also measured
* the docs are also built by ReadTheDocs (separate from GitHub Actions)

When a new [release](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases) is created, the following workflows run:

* installing the package and running tests (using Python 3.13)
* building the package
* checking that the installed package version matches the release tag
* uploading the release to PyPI

## Publishing the package on PyPI

To publish a new ReX version to PyPI, create a [release](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).
This will trigger a set of GitHub Actions workflows, which will run tests, check for version number consistency, and then publish the package to PyPI.

When creating the release, typically the target branch should be `main`.
The target branch should contain all the commits you want to be included in the new release.

The release should be associated with a tag that has the form "vX.Y.Z" - note the "v" prefix!
This can be a new tag that is created for the most recent commit at the time of the release, or can be a pre-eexisting tag.

Give the release a title - this can just be the version number.

Write some release notes explaining the changes incorporated in this release.
Github offers the option to [automatically generate release notes](https://docs.github.com/en/repositories/releasing-projects-on-github/automatically-generated-release-notes) based on PRs merged in since the last release, which can be a good starting point.
Here is [one possible example](https://gist.github.com/andreasonny83/24c733ae50cadf00fcf83bc8beaa8e6a) of how release notes can be structured, to give some ideas of what to include.

The release can be saved as a draft.
When you are ready, use the "Publish release" button to publish the release and trigger the Github Actions workflow that will publish it to PyPI.

If there are any issues with the workflow, it can also be re-run manually.
Navigate to the workflow in the Actions tab of the repo and use the "Run workflow" button to run it manually (after fixing any known issues).
