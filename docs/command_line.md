# Command line usage

```none
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
```

## Model formats

### Onnx

ReX natively understands onnx files. Train or download a model (e.g. [Resnet50](https://github.com/onnx/models/blob/main/validated/vision/classification/resnet/model/resnet50-v1-7.onnx)) and, from this directory, run:

```bash
rex imgs/dog.jpg --model resnet50-v1-7.onnx -vv --output dog_exp.jpg
```

### Pytorch

ReX also works with PyTorch, but some custom preprocessing is required.
See the sample script in `scripts/`.

```bash
rex imgs/dog.jpg --script scripts/pytorch.py -vv --output dog_exp.jpg
```

## Saving output in a database

To store all output in a sqlite database, use:

```bash
rex <path_to_image> --model <path_to_model> -db <name_of_db_and_extension>
```

ReX will create the db if it does not already exist.
It will append to any db with the given name, so be careful not to use the same database if you are restarting an experiment.

## Config

ReX looks for the config file `rex.toml` in the current working directory and then `$HOME/.config/rex.toml` on unix-like systems.

If you want to use a custom location, use:

```bash
rex <path_to_image> --model <path_to_model> --config <path_to_config>
```

An example config file is included in the repo as `example.rex.toml`.
Rename this to `rex.toml` if you wish to use it.

### Overriding the config

Some options from the config file can be overridden at the command line when calling ReX.
In particular, you can change the number of iterations of the algorithm:

```bash
rex <path_to_image> --model <path_to_model>  --iters 5
```

## Preprocessing

ReX by default tries to make reasonable guesses for image preprocessing.
If the image has already been resized appropriately for the model, then use the processed flag:

```bash
rex <path_to_image> --model <path_to_model> --processed
```

ReX will still normalize the image and convert it into a numpy array.
In the event the the model input is single channel and the image is multi-channel, then ReX will try to convert the image to greyscale.
If you want to avoid this, then pass in a greyscale image.

### Preprocess Script

If you have very specific requirements for preprocessing, you can write a standalone function, `preprocess(array)` which ReX will try to load dynamically and call.

```bash
rex <path_to_image> --model <path_to_model> --process_script <path_to_script.py>
```

An example is included in `scripts/example_preprocess.py`.