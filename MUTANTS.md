# ReX: Causal Responsibility Explanations for image classifiers

<picture>
 <source media="(prefers-color-scheme: dark)" srcset="assets/rex_logo.png">
 <source media="(prefers-color-scheme: light)" srcset="assets/rex_logo.png">
 <img alt="ReX Logo with dinosaur" src="YOUR-DEFAULT-IMAGE">
</picture>

<!--- BADGES: START --->

[![CI Pipeline](https://github.com/ReX-XAI/ReX/actions/workflows/python-package.yml/badge.svg)](https://github.com/ReX-XAI/ReX/actions/workflows/python-package.yml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/ReX-XAI/ReX.jl/blob/main/LICENSE)

<!--- BADGES: END --->

***

<!--inclusion-marker-start-do-not-remove-->

## Overview of Mutant Generation and Testing

The core logic of ReX's causal explanation algorithm is the generation and testing of mutants. A mutant is a partially occluded variant of the input data. Mutants with increasingly large occlusions are generated and run through the input model in order to hone in on the exact area or areas responsible for the initial prediction given by the input model. If occluding a small section of the input data and passing it to the input model yields a different result than the initial target calculated from the un-occluded input data, then it follows that the small section which has been occluded has a direct responsibility for the model's output.

The mutant checking process begins within the `causal_explanation()` method in `responsibility.py`. First, a tree and a double-ended queue are both initialised. The tree is an instance of the `Box` class. A concise explanation of `Box` is that it determines which parts of the input data should be occluded to produce a mutant which can then be fed into the input model. An occluded section of input data is called *active*, while an un-occluded section is called *static*. The `Box` tree's root is the entirety of the input data space, with no occlusions. The queue is where the names of passing mutants (which contain information about the current active region of the current box) are stored. 

The queue is then traversed by popping the first mutant name off, which upon just entering the loop for the first time is the root of the tree. This is then divided into four children using the `subbox` method. This method calls the `Box` method `add_children_to_tree`, which encapsulates an internal method that uses random coordinate generation to divide the root `Box` into four areas, which in turn can be divided further into children of their own, thus forming a tree structure. 

These child `Box` instances are then converted into mutants, with one mutant for every possible combination of children being set to either active or static. Each of these mutants is then fed into the input model and classified as either *passing*, meaning that the output produced by the model from the current mutant matches the initial target, or *failing*, meaning that current mutant yields a different prediction than the initial target.

Once the tree reaches the maximum depth or search limit specified in the configuration options, the mutant checking process is over and the information needed to construct an explanation, including lists of passing and failing mutants, is returned.

### `prune()`, `technique`, and `queue_style`

Checking all the passing mutants would be extremely expensive. This is why the list of passing mutants must be pruned, leaving behind only the most viable candidates. The criteria for determining which mutants should be removed from the list depend on the `technique` passed into the `prune()` function through the `queue_style` variable in the main `causal_explanation()` loop. If the `queue_style` is `Queue.All`, then no mutants are pruned from the list and all are returned for consideration. If the technique is `Queue.Intersection`, mutants with duplicate combinations of active boxes are removed. If the technique is `Queue.Area`, only the mutants with the smallest un-occluded regions are returned for consideration. For `Queue.Intersection` and `Queue.Area`, the number of mutants returned is determined by the `keep` parameter, which is set to the queue length in the main `causal_explanation()` loop.


## Class: `Box`

The `Box` class operates as the framework by which sections of an input are chosen for mutant creation, in order to determine which region(s) have the strongest responsibility for the input model's decision. A `Box` can have up to 4 children which are set to either *active* (occluded) or *static* (un-occluded).

Its parent class is BoxInternal, which contains the code that determines the random coordinates used to split a parent Box into up to four children. This internal code is accessed through the Box method `add_children_to_tree()`, which in turn is called in `subbox`.

### `initialise_tree()`

This method creates an instance of a `Box` which encompasses the entire input, using the data dimensions which were set in the data preprocessing step at the beginning of the ReX stack. The distribution and corresponding arguments which are also passed into this method represent the method by which coordinates are chosen from the box in order to create sub-boxes, or child boxes. This `Box` and its children form the basis of mutant checking, which follows a tree search pattern with a set depth (chosen to optimise for speed).

## Class: Mutant

The `Mutant` class stores information about a *mutant*, or a partially occluded variant of the input data given to ReX. An instance of `Mutant` is created for each possible combination of active boxes (as retrieved by the `get_combinations()` method). The `masking_func` parameter in the `Mutant` initialisation function represents the type of occlusion which should be applied to the input data. For image inputs, the default is a constant value represented by the `mask_value` configuration option, which will blank out active sections of the image. For spectral data, the `masking_func` should be a function which linearises the active sections. 


### `get_combinations()`

This method returns a list of all combinations of `Box` children which can be set to active. The four children are represented by numbers 0-3. 

## Class: ResponsibilityMaps

The causal explanation of a model's output is based in part on the size and strength of the impact each section of the input has on the ensuing output. If one section of the input is active, how does that impact the output, as opposed to if a second section is occluded alongside it? This distribution of impact is *responsibility*.

Responsibility for the output of the model when fed a certain mutant is, by default, divided equally between active sections in the mutant. For example, if there are 2 active sections, each will have a responsibility of \frac{1}{2}. If the *weighted* option is chosen, the responsibility for each active section is multiplied by the confidence of the model's prediction. These responsibilities are stored in a `ResponsibilityMaps` object.

As more areas of the input are set to active or static depending on the structure of the `Box` tree, the `ResponsibilityMaps` object is updated with new responsibility values.

