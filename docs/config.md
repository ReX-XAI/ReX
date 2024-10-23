# Configuration

This page describes the different sections in the ReX config file, `rex.toml`, and their possible values.

## [rex] section

mask_value = 0

Masking value for mutations, can be either an integer, float or one of the following built-in occlusions 'spectral', 'min', 'mean'.

seed = 42

Random seed, only set for reproducibility.

gpu = true

Whether to use gpu or not, defaults to true.

batch_size = 64

Batch size for the model.

### [rex.onnx] section

means = [0.485, 0.456, 0.406]

Means for min-max normalization.

stds = [0.229, 0.224, 0.225]

Standard deviations for min-max normalization.

binary_threshold = 0.5

Binary model confidence threshold.
Anything >= threshold will be classified as 1, otherwise 0.

### [rex.visual] section

progress_bar = false

Whether to show progress bar in the terminal, defalts to true.

resize = true

Resize the explanation to the size of the original image.
This uses cubic interpolation and will not be as visually accurate as not resizing, defaults to false.

info = false

Include classification and confidence information in title of plot, defaults to true.

raw = false

Produce unvarnished image with actual masking value, defaults to false.

colour = 150

Pretty printing colour for explanations, defaults to 200.

heatmap = 'coolwarm'

matplotlib colourscheme for responsibility map plotting, defaults to 'magma'.

alpha = 0.2

Alpha blend for main image, defaults to 0.2 (PIL Image.blend parameter).

grid = false

Overlay a 10*10 grid on an explanation, defaults to false.

mark_segments = false

Mark quickshift segmentation on image.

## [causal] section

tree_depth = 30

Maximum depth of tree, defaults to 10.
Note that search can actually go beyond this number on occasion, as the check only occurs at the end of an iteration.

search_limit = 1000

Limit on the number of combinations to consider *per iteration*, defaults to none.
It is **not** the total work done by ReX over all iterations.
Leaving the search limit at none can potentially be very expensive.

iters = 30

Number of times to run the algorithm, defaults to 20.

min_box_size = 10

Minimum child size, in pixels.

confidence_filter = 0.5

Remove passing mutants which have a confidence less than `confidence_filter`.
Defaults to 0.0 (meaning all mutants are considered).

weighted = false

Whether to weight responsibility by prediction confidence, default to false.

queue_style = "area"

queue_style = "intersection" | "area" | "all" | "dc", defaults to "area".

queue_len = 1

Maximum number of things to hold in search queue, either an integer or 'all'.

concentrate = false

`concentrate`: weight responsibility by size and depth of passing partition.
Defaults to false.

### [causal.distribution] section

distribution = 'uniform'

Distribution for splitting the box, defaults to uniform.
Possible choices are 'uniform' | 'binom' | 'betabinom' | 'adaptive'.

blend = 0.5

dist_args = [1.1, 1.1]

Supplemental arguments for distribution creation, these are ignored if `distribution` does not take any parameters.

## [explanation] section

chunk = 10

Iterate through pixel ranking in chunks, defaults to `causal.min_box_size`.

### [explanation.spatial] section

initial_radius = 25

Initial search radius.

radius_eta = 0.2

Increment to change radius.

no_expansions = 50

Number of times to expand before quitting, defaults to 1.

### [explanation.multi] section

method = 'spotlight'

Multi method (only spotlight is currently implemented).

spotlights = 10

Number of spotlights to launch.

spotlight_size = 24

Default size of spotlight.

spotlight_eta = 0.2

Decrease spotlight by this amount.

obj_function = 'mean'

Objective function for spotlight search.
Possible options 'mean' | 'max' | 'min'.

### [explanation.evaluation] section

insertion_step = 100

Insertion/deletion curve step size.
