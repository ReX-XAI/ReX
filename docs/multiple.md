# Multiple Explanations

This page describes **multiple** explanations, see [Multiple Different Black Box Explanations for Image Classifiers](http://www.hanachockler.com/multirex/).

## Example

An image classification may have more than one sufficient explanation. Take this image of a peacock

```{image} ../assets/peacock.jpg
:alt: Peacock
:align: center
```

The global explanation is:


```{image} ../assets/peacock_exp.png
:alt: Peacock Explanation
:scale: 120%
:align: center
```

But it's very likely that there's more than one. This small part of the tail is enough to get the classification `peacock`, but there are many 
other possible sources of information that match that classification. ReX can try to find them.

ReX searches the responsibility map for sufficient explanations. It does this by launched `spotlights` which explore the space, using the responsibility
as a guide. How many spotlights are launched is a parameter (by default: 10) and is set with the `--multi` flag; `--multi` takes an optional
integer argument.

```bash
rex peacock.jpg --script ../tests/scripts/pytorch_resnet50.py --multi 5 --vv --output peacock_exp.png
```
we get

```{image} ../assets/peacock_comp.png
:alt: Peacock Explanation
:align: center
:scale: 120%
```

ReX has found 4 distinct, non-overlapping explanations. The original global explanation is still there, but we also have 3 other explanations.
Two of these explanations (highlighted in white and red respectively) are [*disjoint* explanations](https://arxiv.org/pdf/2411.08875).

## Notes
Multi-ReX has many options and parameters, see [config](explanation_multi) for the complete list.

