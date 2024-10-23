# Background information

ReX is a causal explainability tool for image classifiers.
ReX is black-box, that is, agnostic to the internal structure of the classifier.
We assume that we can modify the inputs and send them to the classifier, observing the output.
ReX outperforms other tools on single explanations, non-contiguous explanations (for partially obscured images), and multiple explanations.

![ReX organisation](../assets/rex-structure-600x129.png)

## Assumptions

![ReX assumptions](../assets/rex-assumptions-768x259.png)

## Presentations about ReX

* [Attacking your black box classifier with ReX](https://www.hanachockler.com/rex-2/)
* [Causal Explanations For Image Classifiers](https://www.hanachockler.com/hana-chockler-causal-xai-workshop-102023/)

## Papers

1. [Multiple Different Explanations for Image Classifiers](http://www.hanachockler.com/multirex/). Under review. This paper introduces MULTI-ReX for multiple explanations.
1. [Explanations for Occluded Images](http://www.hanachockler.com/iccv2021/). In ICCV’21. This paper introduces causality to the tool. Note: the tool is called DC-Causal in this paper.
1. [Explaining Image Classifiers using Statistical Fault Localization](http://www.hanachockler.com/eccv/). In ECCV’20. The first paper on ReX. Note: the tool is called DeepCover in this paper.
