# ReX: Causal *R*esponsibility *EX*planations for image classifiers

<picture>
 <source media="(prefers-color-scheme: dark)" srcset="assets/rex_logo.png">
 <source media="(prefers-color-scheme: light)" srcset="assets/rex_logo.png">
 <img alt="ReX Logo with dinosaur" src="YOUR-DEFAULT-IMAGE">
</picture>

<!--- BADGES: START --->

[![Docs](https://readthedocs.org/projects/rex-xai/badge/?version=latest)](https://rex-xai.readthedocs.io/en/latest/)
[![Tests](https://github.com/ReX-XAI/ReX/actions/workflows/build-and-test-python-package.yml/badge.svg)](https://github.com/ReX-XAI/ReX/actions/workflows/build-and-test-python-package.yml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/ReX-XAI/ReX.jl/blob/main/LICENSE)

<!--- BADGES: END --->

---

ReX is a causal explainability tool for image classifiers. It also works on tabular and 3D data.

Given an input image and a classifier, ReX calculates a causal responsibility map across the data and identifies a minimal, sufficient, explanation.

![ladybird](tests/test_data/ladybird.jpg "Original Image") ![responsibility map](assets/ladybird_rm.png "Responsibility Map") ![minimal explanation](assets/ladybird_301.png "Explanation")

ReX is black-box, that is, agnostic to the internal structure of the classifier.
ReX finds single explanations, non-contiguous explanations (for partially obscured images), multiple independent explanations, contrastive explanations and lots of other things!
It has a host of options and parameters, allowing you to fine tune it to your data.

For background information and detailed usage instructions, see our [documentation](https://rex-xai.readthedocs.io/en/latest/).

<!--inclusion-marker-start-do-not-remove-->

## Installation

Clone this repository and `cd` into it.

```bash
git clone git@github.com:ReX-XAI/ReX.git
cd ReX/
```

We recommend creating a virtual environment to install ReX.
ReX has been tested using versions of Python >= 3.10.
The following instructions assume `conda`:

```bash
conda create -n rex python=3.12
conda activate rex
pip install .
```

This should install an executable `rex` in your path.

> **Note:**
>
> By default, `onnxruntime` will be installed.
> If you wish to use a GPU, you should uninstall `onnxruntime` and install `onnxruntime-gpu` instead.
> You can alternatively edit the `pyproject.toml` to read "onnxruntime-gpu >= 1.17.0" rather than "onnxruntime >= 1.17.0".

<!--inclusion-marker-end-do-not-remove-->

## Feedback

Bug reports, questions, and suggestions for enhancements are welcome - please [check the GitHub Issues](https://github.com/ReX-XAI/ReX/issues) to see if there is already a relevant issue, or [open a new one](https://github.com/ReX-XAI/ReX/issues/new)!

## How to Contribute

Your contributions are highly valued and welcomed. To get started, please review the guidelines outlined in the [CONTRIBUTING.md](/CONTRIBUTING.md) file. We look forward to your participation!
