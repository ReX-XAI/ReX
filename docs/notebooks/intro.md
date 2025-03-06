---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: .venv
  language: python
  name: python3
---

# Using ReX interactively

+++

There are three key components that need to be set up in order to use ReX: the input parameters, the model, and the input data. In this tutorial we will walk through how to set up each of these for a simple ReX analysis using an image classification model. We will use ReX to calculate a responsibility landscape and identify a minimal explanation for the classification of an image by the [ResNet50 model](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html#torchvision.models.resnet50). We will also plot this explanation and the responsibility landscape on the original image. 

## Set up

First, we will set up the input parameters as a `CausalArgs` object.

You can create a `CausalArgs` object with default parameters and then modify it, or use [`load_config`](https://rex-xai.readthedocs.io/en/latest/autoapi/rex_xai/config/index.html#rex_xai.config.load_config) to read your desired set of input parameters from a `rex.toml` file.

```{code-cell} ipython3
from rex_xai.config import CausalArgs

args = CausalArgs()
print(args)
```

For the purpose of this tutorial we will set `gpu = False` and use 10 iterations. We will also set a seed to ensure reproducible outputs.

```{code-cell} ipython3
args.gpu = False
args.iters = 10
args.seed = 123
print(args)
```

Next, we will set up the model details. We need to provide ReX with three things:

- the shape the model expects the input data to be
- a preprocessing function that will apply the appropriate transforms to the input data
- a prediction function to be applied to the transformed input data

For this tutorial we will use the [ResNet50 model](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html#torchvision.models.resnet50) as provided by the `torchvision` library.

```{code-cell} ipython3
:tags: [hide-output]

from torchvision.models import resnet50

model = resnet50(weights="ResNet50_Weights.DEFAULT")
model.eval()
model.to("cpu")
```

We need to define a `model_shape` object which is a list with the expected shape of the input data for the model. In this case the order is `[batch, channels, height, width]`. We use "N" for the batch size in this case as this model accepts dynamic batch sizes. The actual batch size used can be set in the `CausalArgs` object. 

```{code-cell} ipython3
model_shape = ["N", 3, 224, 224]
```

We also need to define a `preprocess` function which will be applied to our input data file, and return a ReX `Data` object that has been appropriately transformed.
The transformations (e.g. resizing, normalisation) should be the same as were used for the model's training. For this model, we use the following transformations:

```{code-cell} ipython3
from torchvision import transforms as T
from PIL import Image

from rex_xai.input_data import Data

def preprocess(path, shape, device, mode) -> Data:
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = Image.open(path).convert("RGB")
    data = Data(img, shape, device, mode=mode, process=False)
    data.data = transform(img).unsqueeze(0).to(device) 
    data.mode = "RGB"
    data.model_shape = shape
    data.model_height = 224
    data.model_width = 224
    data.model_channels = 3
    data.transposed = True
    data.model_order = "first"

    return data
```

Finally, we will define a `prediction_function` that will be applied to our input data and mutants:

```{code-cell} ipython3
import torch as tt
import torch.nn.functional as F

from rex_xai.prediction import from_pytorch_tensor

def prediction_function(mutants, target=None, raw=False, binary_threshold=None):
    with tt.no_grad():
        tensor = model(mutants)
        if raw:
            return F.softmax(tensor, dim=1)
        return from_pytorch_tensor(tensor)
```

Now we are almost ready to run ReX. The final step is to set up the input data.

```{code-cell} ipython3
from rex_xai.explanation import load_and_preprocess_data, predict_target
from rex_xai._utils import get_device

device = get_device(gpu=False)

args.path = '../../tests/test_data/ladybird.jpg'
data = load_and_preprocess_data(model_shape, device, args)
```

This is our input image:

```{code-cell} ipython3
Image.open(args.path)
```

We will now set the mask value to be used to mask the data when creating mutants, and predict the class of the original input image (the 'target' for the mutants).

ReX allows functions to be used to set the mask value (e.g. the 'min' or 'mean' of the normalised image), but the default mask value of 0 generally performs well enough for images.

```{code-cell} ipython3
data.set_mask_value(0)
data.target = predict_target(data, prediction_function)

print(data.target)
```

## Running ReX

We are now ready to run ReX and identify a causal explanation for this classification.

The main function in ReX is `calculate_responsibility`, which returns a `ResponsibilityMaps` object and some statistics about the function execution.
We can then create an `Explanation` object containing the calculated responsibility landscape and `extract` an explanation from this object.
Here we will use the `Global` strategy. Additional strategies are available in the `rex_xai._utils.Strategy` Enum.

```{code-cell} ipython3
:tags: [hide-output]

from rex_xai.explanation import calculate_responsibility
from rex_xai.extraction import Explanation
from rex_xai._utils import Strategy

resp_maps, stats = calculate_responsibility(data, args, prediction_function)
exp = Explanation(resp_maps, prediction_function, data, args, stats)
exp.extract(Strategy.Global)
```

## Examining the results

The mask corresponding to the final explanation is stored in the `Explanation` object and can be printed:

```{code-cell} ipython3
print(exp.final_mask)
```

ReX also provides some plotting methods for easier visualisation of the explanation and the responsibility landscape.

```{code-cell} ipython3
display(exp.show())
```

The responsibility landscape can be plotted as a heatmap or 3D surface plot:

```{code-cell} ipython3
exp.heatmap_plot()
exp.surface_plot()
```

If you are not satisfied with the quality of the explanation output, a good place to start is to check the run statistics output by `calculate_responsibility`. 
In particular, check the tree depth and numbers of mutants examined, as low tree depth or low numbers of mutants examined can lead to strange results. The easiest way to increase both of these and refine an explanation is to increase the number of iterations you use.

```{code-cell} ipython3
print(stats)
```

In this case, the tree depth is 9, almost 1000 mutants have been assessed, and the returned explanation matches well with what we would expect, so we are happy with the quality of the explanation and don’t need to increase the iterations.

Another way to check explanation quality would be to analyze the `Explanation` object and calculate some common metrics.

```{code-cell} ipython3
from rex_xai.explanation import analyze

analyze(exp, data.mode)
```

## Multiple Explanations

There may be multiple regions of an image that are sufficient to explain its classification.
ReX provides a way to identify multiple explanations by using the `MultiExplanation` class and `MultiSpotlight` strategy.

We will use a different image to illustrate this approach, so we need to set up the Data object for this new image.
We will copy the `args` object and change the path to point to the new data.

```{code-cell} ipython3
import copy
peacock_args = copy.copy(args)
peacock_args.path = "../../tests/test_data/peacock.jpg"
data = load_and_preprocess_data(model_shape, device, peacock_args)
Image.open(peacock_args.path)
```

We set the mask value to zero again and predict the class of the new image. 

```{code-cell} ipython3
data.set_mask_value(0)
data.target = predict_target(data, prediction_function)

print(data.target)
```

Next, we will run `calculate_responsibility` on the new image, create a `MultiExplanation` object, and extract explanations.

```{code-cell} ipython3
from rex_xai.multi_explanation import MultiExplanation

resp_maps, stats = calculate_responsibility(data, peacock_args, prediction_function)

multi_exp = MultiExplanation(resp_maps, prediction_function, data, peacock_args, stats)
multi_exp.extract(Strategy.MultiSpotlight)
```

Here we have used the default settings, which generate (up to) ten explanations.
The first explanation is always the explanation identified with `Strategy.Global`.

```{code-cell} ipython3
multi_exp.show(multi_style="separate")
```

We can use the `separate_by` method to identify sets of explanations that do not overlap with each other (or that have an overlap less than some maximum threshold).
Here we will use an overlap of zero to find explanations that have no overlap with each other. 

```{code-cell} ipython3
clauses = multi_exp.separate_by(0)
print(clauses)
```

We have identified three groups of 6 explanations that have no overlap with each other.
The "composite" plotting style can be used to plot these explanations in a single plot.
Here we will just plot the first group. 

```{code-cell} ipython3
multi_exp.show(multi_style="composite", clauses=[clauses[0]])
```

## Saving results

We may want to run ReX on multiple input datasets and save the results for further analysis later.
We can save results from an `Explanation` or `MultiExplanation` object in a sqlite database.
We first initialise the database and then save the ladybird explanation and the multiple explanations of the peacock image.

We did not measure the time taken to calculate the responsibility landscape and extract explanations in this tutorial, so here we use zero in place of the time.
If you wish to calculate this you can use the `time.time()` function to calculate timestamps before and after the steps you wish to time. 

Before saving the results, we should update the `strategy` saved in the `Explanation` or `MultiExplanation` object to ensure it matches the strategy we used, as this will be saved in the database. 

```{code-cell} ipython3
from rex_xai.database import initialise_rex_db, update_database

db = initialise_rex_db("rex.db")

exp.args.strategy = Strategy.Global
update_database(
    db,
    exp.data.target,
    exp,
    0,
    exp.run_stats["total_passing"],
    exp.run_stats["total_failing"],
    exp.run_stats["max_depth_reached"],
    exp.run_stats["avg_box_size"],
    )

multi_exp.args.strategy = Strategy.MultiSpotlight
update_database(
    db,
    multi_exp.data.target,
    multi_exp,
    0,
    multi_exp.run_stats["total_passing"],
    multi_exp.run_stats["total_failing"],
    multi_exp.run_stats["max_depth_reached"],
    multi_exp.run_stats["avg_box_size"],
    multi=True,
)
```

ReX provides a helper function to read the results from the database into a [Pandas](https://pandas.pydata.org/) dataframe for further analysis.

```{code-cell} ipython3
from rex_xai.database import db_to_pandas

df = db_to_pandas("rex.db")
print(df)
```

```{code-cell} ipython3
:tags: [hide-cell]

import os
os.remove("rex.db")
```
