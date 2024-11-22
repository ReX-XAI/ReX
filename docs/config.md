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

`mask_value = 0`

Masking value for mutations, can be either an integer, float or one of the following built-in occlusions 'spectral', 'min', 'mean'.
For image data, we recommend the default value of 0, and for spectral data we recommend 'spectral'.
Note that this masking value is applied to the data after any data normalisation, and therefore does not correspond to a specific colour.
Default: 0.

<!-- TODO: check if this can be a float?
Strongly suggest for images use zero
Spectral/tabular need their own treatment
Expand into full doc just on this.
Masking is applied to the float matrix, not the RGB matrix - doesn't correspond to a particular colour. -->

`seed = 42`

Random seed, only set for reproducibility.
Default: None.

`gpu = true`

Whether to use gpu or not.
Default: true.

`batch_size = 64`

Batch size for inference. Some models specify a certain batch size, in which case the smaller of the two values will be used.
For best performance, set the batch size as high as your hardware allows.
Default: 64.

<!-- Some models tell you the batch size - first element in the model shape. String e.g. N or batch indicates dynamic batch size
Otherwise use smaller of val set here and reported by the model.
This should be set to be the upper limit of what your hardware can give you. -->

### [rex.onnx] section

`means = [0.485, 0.456, 0.406]`

Means for min-max normalization.
Default: None.
<!-- These examples are for imagenet data.
Obtain from model source - should include details of how to pre-process.
Calculated over training data. -->

`stds = [0.229, 0.224, 0.225]`

Standard deviations for min-max normalization.
Default: None.

`binary_threshold = 0.5`

Binary model confidence threshold.
Anything >= threshold will be classified as 1, otherwise 0.
Default: None.

`norm`

Optional.
Default: 255.

<!-- Should be 255 for images.
For other data types raise a warning if left as default. -->

### [rex.visual] section

This section is for options that control the appearance of the progress bar and output data visualisation.

`progress_bar = false`

Whether to show progress bar in the terminal.
Defaults: true.

`resize = true`

Resize the explanation to the size of the original image.
This uses cubic interpolation and will not be as visually accurate as not resizing.
Default: false.

<!-- Notes: explanation produced with the size used for the model. resize adjusts this back to the input image size. -->

`info = false`

Include classification and confidence information in title of plot.
Default: true.

`raw = false`

Produce unvarnished image with actual masking value (or black as default).
Defaults: false.

<!-- Or: show only pixels actually input to model, all else blacked out - check whether code needs to be changed so always black. -->

`colour = 150`

Pretty printing colour for explanations.
Defaults: 200.

<!-- colour used to grey out pixels - 150.
Used as R,G,B.
In theory can pass in 3 numbers instead - not implemented. -->

`heatmap = 'coolwarm'`

Matplotlib colourscheme for responsibility map plotting.
Default: 'magma'.
See the [Matplotlib documentation](https://matplotlib.org/stable/users/explain/colors/colormaps.html) for a list of all possible colourmaps.

`alpha = 0.2`

Alpha blend for main image (PIL Image.blend parameter).
Default: 0.2.
<!-- Used for pretty printing of output image. -->

`grid = false`

Overlay a 10*10 grid on an explanation.
Defaults: false.

`mark_segments = false`

Mark quickshift segmentation on image.
Default: false.

<!-- Helpful to see if segmentation matches/spans explanation.
Not used during explanation. -->

## [causal] section

This sections is for options relevant to the causal responsibility calculations.

`tree_depth = 30`

Maximum depth of tree.
Default: 10.
Note that search can actually go beyond this number on occasion, as the check only occurs at the end of an iteration.

<!-- TODO: add a diagram?
30 is very large (getting to individual pixels), 10 is easily reachable. -->

`search_limit = 1000`

Limit on the number of combinations to consider *per iteration*.
Optional.
Default: None.
It is **not** the total work done by ReX over all iterations.
Leaving the search limit at none can potentially be very expensive.

<!-- number of mutants per iteration
Is there a total work limit as well? - check -->

`iters = 30`

Number of times to run the algorithm.
Default: 20.
<!-- More iters -> smaller explanations.
Probably the only one most people will set. -->

`min_box_size = 10`

Minimum child size, in pixels.
Default: 10.
<!-- Area of box. In practice never gets this small. -->

`confidence_filter = 0.5`

Remove passing mutants which have a confidence less than `confidence_filter`.
Default: 0.0 (meaning all mutants are considered).
<!-- everything above this gets added to resp map
Advanced option -->

`weighted = false`

Whether to weight responsibility by prediction confidence.
Default: false.
<!-- One people might want to set. -->

`queue_style = "area"`

queue_style = "intersection" | "area" | "all" | "dc"
Default: "area".
<!-- Advanced.
Aim is to not have to follow every path
Area: choose passing mutant that has smallest area
All: keep everything - very slow
Intersection: smallest occlusion that occurs in highest number of passing mutants - but area works better.
Affects how queue is ordered -->

`queue_len = 1`

Maximum number of things to hold in search queue, either an integer or 'all'.
Default: 1.
<!-- How many items from queue to process
'all' will be slow -->

`concentrate = false`

Weight responsibility by size and depth of passing partition.
Default: false.
<!-- Experimental! remove? doesn't work yet -->

`segmentation`

Default: false.
<!-- Remove this! -->

### [causal.distribution] section

This section is for options that control the distribution used to split boxes.

`distribution = 'uniform'`

Distribution for splitting the box.
Default: 'uniform'.
Possible choices are 'uniform' | 'binom' | 'betabinom' | 'adaptive'.

<!-- Uniform works, the others might not work at the moment. -->

`blend = 0.5`

Default: 0.0.
Ignored for uniform - used for adaptive. Experimental.

`dist_args = [1.1, 1.1]`

Supplemental arguments for distribution creation, these are ignored if `distribution` does not take any parameters.
Optional.
Default: None.
<!-- Ignored for uniform -->

## [explanation] section

This section is for options related to identifying the explanation(s).

`chunk = 10`

Iterate through pixel ranking in chunks, defaults to `causal.min_box_size`. 
<!-- Adds this number of pixels at a time - faster.
User may want to adjust this - if model is slow need bigger values. -->

### [explanation.spatial] section

This section is for options used for spatial search for explanations.

<!-- Needs expanding into a separate page with illustrations. -->

`initial_radius = 25`

Initial search radius.
<!-- Shines a "spotlight" on the image, focuses on looking for explanation in that region. Centred on top pixel. -->
Default: 25.

`radius_eta = 0.2`

Increment to change radius. 
<!-- - i.e. increase by 20%. -->
Default: 0.2.

`no_expansions = 50`

Number of times to expand before quitting
Default: 1.
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

method = 'spotlight'
<!-- from code looks like this is not used currently -->
<!-- Keep as hidden legacy option? Not currently planning to implement others -->

Multi-explanation method (only spotlight is currently implemented).

`spotlights = 10`

Number of spotlights to launch.
Default: 10.

`spotlight_size = 24`

Default size of spotlight.
Default: 20.
<!-- Could default to values set above. Radius of approx circle. -->

`spotlight_eta = 0.2`

Increase spotlight by this amount.
Default: 0.2

`obj_function = 'mean'`

Objective function for spotlight search.
Possible options 'mean' | 'max' | 'min'.
<!-- Simplified stochastic hill climb - move spotlight towards area with mean/max/min repsonsibility -->

`spotlight_step = 5`

Default: 5.
<!-- How far spotlight is moved. -->

### [explanation.evaluation] section

Metrics from literature used to assess explanation quality.

These options are used if ReX is run with the `--analyse` option.

`insertion_step = 100`

Insertion/deletion curve step size.
Default: 100.
