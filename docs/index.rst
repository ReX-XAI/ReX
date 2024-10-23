.. ReX documentation master file, created by
   sphinx-quickstart on Fri Sep 27 12:09:33 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ReX: Causal Responsibility Explanations for image classifiers
=================

**ReX** is a causal explainability tool for image classifiers.
ReX is black-box, that is, agnostic to the internal structure of the classifier.
We assume that we can modify the inputs and send them to the classifier, observing the output.
ReX outperforms other tools on single explanations, non-contiguous explanations (for partially obscured images), and multiple explanations.

.. image:: ../assets/rex-structure-600x129.png
  :width: 600
  :alt: ReX organisation

For more information and links to the papers, see the :doc:`background` page.

Installation
----

Clone the ReX repository and ``cd`` into it::

   git clone git@github.com:ReX-XAI/ReX.git
   cd ReX/

We recommend creating a virtual environment to install ReX.
ReX has been tested using versions of Python >= 3.10.
The following instructions assume ``conda``::

   conda create -n rex python=3.12
   conda activate rex
   pip install .

This should install an executable ``rex`` in your path.

  **Note:**
  
  By default, ``onnxruntime`` will be installed.
  If you wish to use a GPU, you should uninstall ``onnxruntime`` and install ``onnxruntime-gpu`` instead.
  You can alternatively edit the ``pyproject.toml`` to read "onnxruntime >= 1.17.0" rather than "onnxruntime-gpu >= 1.17.0" before running ``pip install .``. 


Quickstart
----

ReX requires as input an image and a model. 
ReX natively understands onnx files. Train or download a model (e.g. `Resnet50 <https://github.com/onnx/models/blob/main/validated/vision/classification/resnet/model/resnet50-v1-7.onnx>`_) and, from this directory, run::

   rex imgs/dog.jpg --model resnet50-v1-7.onnx -vv --output dog_exp.jpg

To view an interactive plot for the responsibility map, run::

   rex imgs/dog.jpg --model resnet50-v1-7.onnx -vv --surface
   
Other options::

   # with spatial search (the default)
   rex <path_to_image> --model <path_to_model>

   # with linear search
   rex <path_to_image> --model <path_to_model> --strategy linear

   # to save the extracted explanation
   rex <path_to_image> --model <path_to_model> --output <path_and_extension>

   # to view an interactive responsibility landscape
   rex <path_to_image> --model <path_to_model>  --surface

   # to save a responsibility landscape
   rex <path_to_image> --model <path_to_model>  --surface <path_and_extension>

   # to run multiple explanations
   rex <path_to_image> --model <path_to_model> --strategy multi

ReX configuration is mainly handled via a config file; some options can also be set on the command line.
ReX looks for the config file ``rex.toml`` in the current working directory and then ``$HOME/.config/rex.toml`` on unix-like systems.

If you want to use a custom location, use::

   rex <path_to_image> --model <path_to_model> --config <path_to_config>

An example config file is included in the repo as ``example.rex.toml``.
Rename this to ``rex.toml`` if you wish to use it.

Command line usage 
----

::

   usage: ReX [-h] [--output [OUTPUT]] [-c CONFIG] [--processed]
            [--script SCRIPT] [-v] [--surface [SURFACE]] [--heatmap [HEATMAP]]
            [--model MODEL] [--strategy STRATEGY] [--database DATABASE]
            [--iters ITERS] [--analyze] [--analyse] [--show-all] [--mode MODE]
            filename

   Explaining AI through causal reasoning

   positional arguments:
   filename              file to be processed, assumes that file is 3 channel
                           (RGB or BRG)

   options:
   -h, --help            show this help message and exit
   --output [OUTPUT]     show minimal explanation, optionally saved to
                           <OUTPUT>. Requires a PIL compatible file extension
   -c CONFIG, --config CONFIG
                           config file to use for rex
   --processed           don't perform any processing with rex itself
   --script SCRIPT       custom loading and preprocessing script, for us with pytorch
   -v, --verbose         verbosity level, either -v or -vv, or -vvv
   --surface [SURFACE]   surface plot, optionally saved to <SURFACE>
   --heatmap [HEATMAP]   heatmap plot, optionally saved to <HEATMAP>
   --model MODEL         model, must be onnx format
   --strategy STRATEGY, -s STRATEGY
                           explanation strategy, one of < multi | spatial |
                           linear | spotlight >
   --database DATABASE, -db DATABASE
                           store output in sqlite database <DATABASE>, creating
                           db if necessary
   --iters ITERS         manually override the number of iterations set in the
                           config file
   --analyze             area, entropy different and insertion/deletion curves
   --analyse             area, entropy different and insertion/deletion curves
   --mode MODE, -m MODE  assist ReX with your input type, one of <tabular>,
                           <spectral>, <RGB>, <L>


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   background.md

