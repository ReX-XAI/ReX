# Configuration

This page describes the different sections in the ReX config file, `rex.toml`, and their possible values.
TOML is a [minimal configuration file format](https://toml.io/en/).
An example `rex.toml` file that configures a few of the possible parameters might look like this:

```none
[rex]

gpu = false

batch_size = 32

[rex.onnx]

[rex.visual]

progress_bar = true

[causal]

search_limit = 1000

iters = 35

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

`mask_value = 0`

Masking value for mutations. Can be an integer, float or one of the following built-in occlusions 'spectral', 'min', 'mean'.
For image data, we recommend the default value of 0, and for spectral data we recommend 'spectral'.
Note that this masking value is applied to the data after any data normalisation, and therefore does not correspond to a specific colour.
Default: 0.

`seed = 42`

Random seed, only set for reproducibility.
Default: None.

`gpu = true`

Whether to use GPU (if available) or not.
Default: true.

`batch_size = 64`

Batch size for inference. Some models specify a certain batch size.
If the batch size specified by the model is smaller than this value, the batch size specified by the model will be used.
For best performance, set the batch size as high as your hardware allows.
Default: 64.

### [rex.onnx] section

This section is for options used when specifying a model as an onnx format file.

`norm = 255`

Value used to scale input data - input data are divided by this value.
The default value of 255 is appropriate for images, as it scales the values to the range [0, 1].
Default: 255.

`means = [0.485, 0.456, 0.406]`

Means for input data normalization.
The mean for each channel will be subtracted from the channel values, after they have been scaled (see `norm`).
These examples are for ImageNet.
If you are using your own model, you should calculate the mean and standard deviation from your own training data.
Default: None.

`stds = [0.229, 0.224, 0.225]`

Standard deviations for input data normalization.
Values for each channel will be divided by the standard deviation for each channel, after they have been scaled (see `norm`) and the mean has been subtracted (see `mean`).
These examples are for ImageNet.
If you are using your own model, you should calculate the mean and standard deviation from your own training data.
Default: None.

`binary_threshold = 0.5`

Binary model confidence threshold.
Anything >= threshold will be classified as 1, otherwise 0.
Default: None.

### [rex.visual] section

This section is for options that control the appearance of the progress bar and output data visualisations.

`progress_bar = false`

Whether to show progress bar in the terminal.
Defaults: true.

`resize = true`

Resize the explanation to the size of the original image, rather than the size used for the model.
This uses cubic interpolation and will not be as visually accurate as not resizing.
Used with option `--output`.
Default: false.

`raw = false`

Produce unvarnished image with masked regions set to black.
Used with option `--output`.
Default: false.

`colour = 150`

Shade of grey used to mask image and highlight the explanation.
Used with option `--output`, if `raw` is `false`.
Default: 200.

`alpha = 0.2`

Alpha blend for main image (PIL Image.blend parameter).
Used with option `--output`, if `raw` is `false`.
Default: 0.2.

`grid = false`

Overlay a 10*10 grid on an explanation.
Used with option `--output`.
Defaults: false.

`mark_segments = false`

Mark quickshift segmentation on image.
Segmentation is performed with `skimage.segmentation.slic`. Used with option `--output`.
Default: false.

`info = false`

Include classification and confidence information in title of plot, and label centre of mass and maximum points of responsibility map.
Used with option `--surface`.
Default: true.

`heatmap = 'coolwarm'`

Matplotlib colourscheme for responsibility map plotting.
Used with option `--heatmap` and `--surface`.
Default: 'magma'.
See the [Matplotlib documentation](https://matplotlib.org/stable/users/explain/colors/colormaps.html) for a list of all possible colourmaps.

## [causal] section

This sections is for options relevant to the causal responsibility calculations.

`iters = 30`

Number of times to run the algorithm.
More iterations will generally lead to smaller explanations.
Default: 20.

`weighted = false`

Whether to weight responsibility by prediction confidence.
Default: false.

`tree_depth = 20`

Maximum depth of tree.
Default: 10.
Note that search can actually go beyond this number on occasion, as the check only occurs at the end of an iteration.

<!-- TODO: add a diagram?
30 is very large (getting to individual pixels), 10 is easily reachable. 
'iteration' here doesn't mean what's set by 'iters', but rather per item in the search queue -->

`search_limit = 1000`

Limit on the number of combinations to consider *per iteration*.
Optional.
Default: None.
It is **not** the total work done by ReX over all iterations.
Leaving the search limit at none can potentially be very expensive.

<!-- number of mutants per iteration, no total search limit
'iteration' here *is* what's set by 'iters' -->

`min_box_size = 10`

Minimum child box area, in pixels.
Default: 10.
<!-- Area of box. In practice never gets this small. -->

`confidence_filter = 0.5`

Remove passing mutants which have a confidence less than `confidence_filter` * the confidence on the full input data.
Default: 0.0 (meaning all mutants are considered).
<!-- Advanced option -->

`queue_style = "area"`

This parameter controls how the queue of mutants to analyse is ordered.
"area" means that the passing mutant with the smallest area will be ordered first.
"intersection" means that ordering is based on the smallest occlusion that occurs in the highest number of passing mutants.
"all" will keep all mutants and may lead to significant slowdown.
Possible values: "intersection" | "area" | "all" | "dc"
Default: "area".
<!-- Advanced.
Aim is to not have to follow every path
Area: choose passing mutant that has smallest area
All: keep everything - very slow
Intersection: smallest occlusion that occurs in highest number of passing mutants - but area works better.
Affects how queue is ordered 
TODO: clarify - how does intersection work? -->

`queue_len = 1`

How many items from the search queue to keep, either an integer or 'all'.
Note that specifying 'all' may lead to significant slowdown.
Default: 1.

<!-- `concentrate = false`

Weight responsibility by size and depth of passing partition.
Default: false. 
Experimental! doesn't work yet -->

### [causal.distribution] section

This section is for options that control the distribution used to split boxes.

`distribution = 'uniform'`

Distribution for splitting the box.
Only 'uniform' is currently recommended, the other options are experimental.
Default: 'uniform'.

<!-- Possible choices are 'uniform' | 'binom' | 'betabinom' | 'adaptive'. -->
<!-- Uniform works, the others might not work at the moment. -->

`dist_args = [1.1, 1.1]`

Supplemental arguments for distribution creation, these are ignored if `distribution` does not take any parameters.
Optional.
Default: None.
<!-- Currently don't seem to be passed in to random_coords, so not used -->

<!-- `blend = 0.5`

Experimental.
Used for 'adaptive'.
Default: 0.0. -->
<!-- Not currently used -->

## [explanation] section

This section is for options related to identifying the explanation(s).

`chunk = 10`

To find a minimal sufficient explanation, ReX adds pixels according to the pixel ranking in chunks, then checks the classification of the output.
This option sets the number of pixels that will be added at each iteration.
If your model is slow to classify, you may want to increase this value.
Defaults to `causal.min_box_size`.
<!-- Adds this number of pixels at a time - faster.
User may want to adjust this - if model is slow need bigger values. -->

### [explanation.spatial] section

This section is for options used for spatial search for explanations.
The spatial search looks for explanations within a 'spotlight' region centred on the highest ranked pixel.
If a sufficient explanation cannot be found, the spotlight size will be increased.

<!-- Needs expanding into a separate page with illustrations. -->

`initial_radius = 25`

Initial search radius.
Default: 25.

`radius_eta = 0.2`

Increment to increase radius.
0.2 means an increase of 20%.
Default: 0.2.

<!-- `no_expansions = 50`

Number of times to expand before quitting.
Default: 1. -->
<!-- Not currently used. -->
<!-- from code looks like default is 50? -->
<!-- Should the default be the batch size? warning if batch size is 1?
Check what happens if model batch size is dynamic and batch size is not set.
Could calculate % of image that would be covered? -->

### [explanation.multi] section

<!-- 
- place spotlight at random location
- expand - uses no_expansions above
- look for target classification
- if found then start at centre of circle, add pixels according to chunk size, and find minimal explanation
- if not found then revert to original size, move, and start again - how many times? TODO find out what this is and expose to user
- if can't find anything, returns global explanation -->

This option is currently disabled.

<!-- method = 'spotlight' -->
<!-- Multi-explanation method (only spotlight is currently implemented). -->
<!-- from code looks like this is not used currently -->
<!-- Keep as hidden legacy option? Not currently planning to implement others -->

`spotlights = 10`

Number of spotlights to launch.
Default: 10.

`spotlight_size = 24`

Initial spotlight radius.
Default: 20.
<!-- Could default to values set above. Radius of approx circle. -->

`spotlight_eta = 0.2`

Increase spotlight radius by this amount.
Default: 0.2

`obj_function = 'mean'`

Objective function for spotlight search.
Possible options 'mean' | 'max' | 'min'.
If no explanation is found, the spotlight is moved towards a location defined by this function.
<!-- Simplified stochastic hill climb - move spotlight towards area with mean/max/min repsonsibility -->

`spotlight_step = 5`

If no explanation is found, the spotlight is moved this distance in pixels towards a location defined by `obj_function`.
Default: 5.
<!-- How far spotlight is moved. -->

### [explanation.evaluation] section

These options are used if ReX is run with the `--analyse` option, which triggers the calculation of several metrics from the literature used to assess explanation quality.

`insertion_step = 100`

Insertion/deletion curve step size.
Default: 100.
