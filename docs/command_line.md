# Command line usage

<!--inclusion-marker-start-do-not-remove-->

```none
usage: ReX [-h] [--output [OUTPUT]] [-c CONFIG] [--processed] [--script SCRIPT] [-v] [--surface [SURFACE]] [--heatmap [HEATMAP]] [--model MODEL] [--strategy STRATEGY]
       [--database DATABASE] [--multi [MULTI]] [--contrastive [CONTRASTIVE]] [--iters ITERS] [--analyze] [--analyse] [--mode MODE] [--spectral] filename

                       @@@@@@@@@@@@@@@
                       @@  @@@@@@@@@@@
                     @@@@  @@@@@@@@@@@@@@@@@
                     @@@@@@@@@@@@@@@@@@@@@@@
                     @@@@@@@@@@@@@@@@@@@@@@@
                     @@@@@@@@@@
                     @@@@@@@@@@@@@@@@         %%%
   @@              @@@@@@@@@@@                %%%%
   @@              @@@@@@@   %%@               %%%%             %%
   @@            @@@@@@@@ @%%%%%%%              %%%%           %%
   @@@@        @@@@@@@@@@ %%%%  %%%%%%           %%%%        %%
   @@@@@@    @@@@@@@@@@@@ %%%%     %%%%%%      %   %%%     %%
   @@@@@@@@@@@@@@@@@@@@@@ %%%%       %      %%      %%%   %
     @@@@@@@@@@@@@@@@@@@@ %%%%    %%      %          %%%%
       @@@@@@@@@@@@@@@@@@ %%%   %%     %%%%%%%%%%%    %%%
         @@@@@@@@@@@@@@@@ %%%  %%%%   %%%%           % %%%
         @@@@@@@@@@@@@@@  %%%   %%%%   %%%%        %@   %%%
           @@@@@@@@@@@@@  %%%    %%%%    %%%     %%      %%%
             @@@@@@  @@@  %%%     %%%%    %%%% %%         %%%%
             @@@@        %%%%      %%%%%   %%%%            %%%%  %
             @@          %%%%        %%%%   %%%%%@          %%%%%
             @@@@        %            %%      %              %%

   Explaining AI throught causal Responsibilty EXplanations

positional arguments:
  filename              file to be processed, either an image, a numpy array, a mat file, or a nifti file

options:
  -h, --help            show this help message and exit
  --output [OUTPUT]     show minimal, sufficient causal explanation, optionally saved to <OUTPUT>. Requires a PIL compatible file extension
  -c CONFIG, --config CONFIG
                        optional config file to use for ReX
  --processed           prevent ReX from performing any preprocessing
  --script SCRIPT       custom loading and preprocessing script, mostly for use with pytorch models
  -v, --verbose         verbosity level, either -v or -vv, or -vvv
  --surface [SURFACE]   surface plot of the responsibility map, optionally saved to <SURFACE>
  --heatmap [HEATMAP]   heatmap plot of the responsibility map, optionally saved to <HEATMAP>
  --model MODEL         model in .onnx format
  --strategy STRATEGY, -s STRATEGY
                        explanation strategy, one of < multi | spatial | global | spotlight >, defaults to global
  --database DATABASE, -db DATABASE
                        store output in sqlite database <DATABASE>, creating db if necessary. Please include the extension in the filename
  --multi [MULTI]       multiple explanations, with optional number <x> of spotlights, defaults to value in <rex.toml>, or 10 if undefined
  --contrastive [CONTRASTIVE]
                        a contrastive explanation, minimal, necessary and sufficient. Needs optional number <x> of floodlights, defaults to value in <rex.toml>, or 10 if undefined
  --iters ITERS         manually override the number of iterations set in the config file
  --analyze             area, entropy and (possibly) insertion/deletion curves
  --analyse             area, entropy and (possibly) insertion/deletion curves
  --mode MODE, -m MODE  assist ReX with your input type, one of <tabular>, <spectral>, <RGB>, <voxel>, <audio>
  --spectral            set ReX input type to <spectral>, shortcut for --mode spectral
```

<!--inclusion-marker-end-do-not-remove-->

## Model formats

### Onnx

ReX natively understands onnx files. Train or download a model (e.g. [Resnet50](https://github.com/onnx/models/blob/main/validated/vision/classification/resnet/model/resnet50-v1-7.onnx)) and, from this directory, run:

```bash
rex tests/test_data/dog.jpg --model resnet50-v1-7.onnx -vv --output dog_exp.jpg
```

### Pytorch

ReX also works with PyTorch, but you will need to write some custom code to provide ReX with the prediction function and model shape, as well as preprocess the input data.
See the sample scripts in `scripts/`.

```bash
rex tests/test_data/dog.jpg --script scripts/pytorch.py -vv --output dog_exp.jpg
```

## Saving output in a database

To store all output in a sqlite database, use:

```bash
rex <path_to_image> --model <path_to_model> -db <name_of_db_and_extension>
```

ReX will create the db if it does not already exist.
It will append to any db with the given name, so be careful not to use the same database if you are restarting an experiment.
ReX comes with a number of database functions which allow you to load it as a pandas dataframe for analysis.

## Config

ReX looks for the config file `rex.toml` in the current working directory and then `$HOME/.config/rex.toml` on unix-like systems.

If you want to use a custom location, use:

```bash
rex <path_to_image> --model <path_to_model> --config <path_to_config>
```

An example config file is included in the repo as `example.rex.toml`.
Rename this to `rex.toml` if you wish to use it. ReX will ignore the file `example.rex.toml` is you do not rename it.

### Overriding the config

Some options from the config file can be overridden at the command line when calling ReX.
In particular, you can change the number of iterations of the algorithm:

```bash
rex <path_to_image> --model <path_to_model> --iters 5
```

## Preprocessing

Input data should be transformed in the same way the model's training data was transformed before training.
For PyTorch models, you should specify the preprocessing steps in the custom script.
See the sample scripts in `scripts/` for examples of using models provided by the [torchvision](https://pytorch.org/vision/stable/index.html) and [timm](https://huggingface.co/docs/timm/index) packages.
You can also write a script to load an `onnx` model as well if you have custom preprocessing to do.

Otherwise, ReX tries to make reasonable guesses for image preprocessing.
This includes resizing the image to match that needed for the model, converting it to a PyTorch tensor, and normalising the data. This can be controlled in `rex.toml`.
In the event the the model input is multi-channel and the image is greyscale, then ReX will convert the image to pseudo-RGB.
If you want more control over the conversion, you can do the conversion yourself and pass in the converted image.

<!-- If the image has already been resized appropriately for the model, then use the `--processed` flag:

```bash
rex <path_to_image> --model <path_to_model> --processed
``` -->

<!--### Preprocess Script-->
<!---->
<!--If you have very specific requirements for preprocessing, you can write a standalone function, `preprocess(array)` which ReX will try to load dynamically and call.-->
<!---->
<!--```bash-->
<!--rex <path_to_image> --model <path_to_model> --process_script <path_to_script.py>-->
<!--```-->
<!---->
<!--An example `preprocess` function is included in `tests/scripts/pytorch_resnet50.py`.-->
