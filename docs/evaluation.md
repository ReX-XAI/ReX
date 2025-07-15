# Evaluation

ReX has a few basic analyses built in. These are 

1. area 
2. KL divergence (from a uniform distribution)
3. robustness to noise
4. insertion curve
5. deletion curve
6. total time (including analysis time)

If you call ReX with the `--analyse` flag, you might get something like this:

```bash
INFO:ReX:path tests/test_data/peacock.jpg, classification 84, area 0.0167, KL divergence 4.055, 
  robustness 0.0, insertion curve 1.06958, deletion curve 0.682799, time 8.3669
```

`area` is the size of the sufficient explanation as a ratio of the entire image. 

`KL divergence` is the 
[Kullback-Liebler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) of the 
**responsibility map** against a uniform distribution: it measures how far the responsibility map is from flatness in bits. 
Flatness usually indicates that the explanation is not *local* (it is distributed evenly over the image) or that 
something has gone very wrong! 

`robustness` takes the sufficient explanation and superimposes it over random backgrounds. It measures how
often the sufficient explanation manages to maintain its classification in the face of noise. When an explanation is small, robuestness is usually low. 
In this case, the explanation is just over 1% of the image, and 
robustness in the face of noise is 0.0. You might try increasing the `--confidence` threshold, or looking for a [contrastive](contrastive) explanation
if you want something more robust.

`insertion` and `deletion` are the AUC for insertion/deletion curves (see the [RISE](https://arxiv.org/abs/1806.07421) paper). 
By default, these are normalised by the original confidence of the model.

`time` indictates the total time spent, including analysis time.
