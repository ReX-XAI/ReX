# Configuration

This page describes the different sections in the ReX config file, `rex.toml`, and their possible values.
TOML is a [minimal configuration file format](https://toml.io/en/).
An example `rex.toml` file that configures a few of the possible parameters might look like this:

```none
[rex]

gpu = false

batch_size = 4

[rex.onnx]

binary_threshold = 0.5

[rex.visual]

progress_bar = true

[causal]

search_limit = 1000

iters = 30

[causal.distribution]

distribution = 'uniform'

[explanation]

[explanation.spatial]

[explanation.multi]

[explanation.evaluation]
```

N.B. that all section/subsection headers must be present even if no values in that section are set to non-default values.

## [rex] section

This section is for options that affect the general features of ReX.

mask_value = 0

Masking value for mutations, can be either an integer, float or one of the following built-in occlusions 'spectral', 'min', 'mean'.
Default: 0.

seed = 42

Random seed, only set for reproducibility.
Default: None.

gpu = true

Whether to use gpu or not.
Default: true.

batch_size = 64

Batch size for the model.
Default: 1.

### [rex.onnx] section

means = [0.485, 0.456, 0.406]

Means for min-max normalization.
Default: None.

stds = [0.229, 0.224, 0.225]

Standard deviations for min-max normalization.
Default: None.

binary_threshold = 0.5

Binary model confidence threshold.
Anything >= threshold will be classified as 1, otherwise 0.
Default: None.

norm
<!-- unclear to me what this is used for -->

Optional.
Default: None.

### [rex.visual] section

This section is for options that control the appearance of the progress bar and output data visualisation.

progress_bar = false

Whether to show progress bar in the terminal.
Defaults: true.

resize = true

Resize the explanation to the size of the original image.
This uses cubic interpolation and will not be as visually accurate as not resizing.
Default: false.

info = false

Include classification and confidence information in title of plot.
Default: true.

raw = false

Produce unvarnished image with actual masking value.
Defaults: false.

colour = 150

Pretty printing colour for explanations.
Defaults: 200.

heatmap = 'coolwarm'

Matplotlib colourscheme for responsibility map plotting.
Default: 'magma'.
See the [Matplotlib documentation](https://matplotlib.org/stable/users/explain/colors/colormaps.html) for a list of all possible colourmaps.

alpha = 0.2

Alpha blend for main image (PIL Image.blend parameter).
Default: 0.2.

grid = false

Overlay a 10*10 grid on an explanation.
Defaults: false.

mark_segments = false

Mark quickshift segmentation on image.
Default: false.

## [causal] section

This sections is for options relevant to the causal responsibility calculations.

tree_depth = 30

Maximum depth of tree.
Default: 10.
Note that search can actually go beyond this number on occasion, as the check only occurs at the end of an iteration.

search_limit = 1000

Limit on the number of combinations to consider *per iteration*.
Optional.
Default: None.
It is **not** the total work done by ReX over all iterations.
Leaving the search limit at none can potentially be very expensive.

iters = 30

Number of times to run the algorithm.
Defaults: 20.
<!-- from code looks like default is actually 30 -->

min_box_size = 10

Minimum child size, in pixels.
Default: 10.
<!-- is this width or area? -->

confidence_filter = 0.5

Remove passing mutants which have a confidence less than `confidence_filter`.
Default: 0.0 (meaning all mutants are considered).

weighted = false

Whether to weight responsibility by prediction confidence.
Default: false.

queue_style = "area"

queue_style = "intersection" | "area" | "all" | "dc"
Default: "area".

queue_len = 1

Maximum number of things to hold in search queue, either an integer or 'all'.
Default: 1.

concentrate = false

Weight responsibility by size and depth of passing partition.
Default: false.

segmentation

Default: false.

### [causal.distribution] section

This section is for options that control the distribution used to split boxes.

distribution = 'uniform'

Distribution for splitting the box.
Default: 'uniform'.
Possible choices are 'uniform' | 'binom' | 'betabinom' | 'adaptive'.

blend = 0.5

Default: 0.0.

dist_args = [1.1, 1.1]

Supplemental arguments for distribution creation, these are ignored if `distribution` does not take any parameters.
Optional.
Default: None.

## [explanation] section

This section is for options related to identifying the explanation(s).

chunk = 10

Iterate through pixel ranking in chunks, defaults to `causal.min_box_size`.

### [explanation.spatial] section

This section is for options used for spatial search for explanations.

initial_radius = 25

Initial search radius.
Default: 25.

radius_eta = 0.2

Increment to change radius.
Default: 0.2.

no_expansions = 50

Number of times to expand before quitting
Default: 1.
<!-- from code looks like default is 50? -->

### [explanation.multi] section

method = 'spotlight'
<!-- from code looks like this is not used currently -->

Multi-explanation method (only spotlight is currently implemented).

spotlights = 10

Number of spotlights to launch.
Default: 10.

spotlight_size = 24

Default size of spotlight.
Default: 20.

spotlight_eta = 0.2

Decrease spotlight by this amount.
Default: 0.2

obj_function = 'mean'

Objective function for spotlight search.
Possible options 'mean' | 'max' | 'min'.

spotlight_step

Default: 5.

### [explanation.evaluation] section

These options are used if ReX is run with the `--analyse` option.

insertion_step = 100

Insertion/deletion curve step size.
Default: 100.
