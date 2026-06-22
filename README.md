# Monocular Depth Estimation

A compact encoder decoder that predicts a single channel depth map from one RGB
image, trained with a scale invariant loss. Everything runs on CPU with synthetic
data, so the whole project trains and tests in seconds and needs no downloads.

## Why scale invariant

Recovering absolute depth from a single image is ambiguous: a small near object
and a large far object can look identical. A common way to handle this is to stop
penalising the network for getting the overall scale wrong and instead reward it
for getting the relative structure right. The loss here follows Eigen, Puhrsch
and Fergus (2014). It works in log space and subtracts out any constant offset,
which corresponds to a global multiplicative factor in linear depth. Multiplying
every predicted depth by the same positive number leaves the loss unchanged.

For per pixel log differences `d_i = log(pred_i) - log(target_i)` over `N` pixels:

```
L = mean(d_i^2) - lam * mean(d_i)^2
```

With `lam = 1` the loss is exactly the variance of the log error and is fully
invariant to scale. Values below 1 trade a little of that invariance for a mild
pull toward the correct absolute scale, which is the usual setting in practice.

## Model

`DepthNet` is a small U-Net style encoder decoder. Two downsampling stages feed
a bottleneck, two upsampling stages with skip connections bring the resolution
back, and a one by one convolution produces a single channel output at the same
height and width as the input. The head has no activation, so the network
predicts depth in log space, which matches the loss directly.

## Layout

```
src/
  model.py    DepthNet encoder decoder
  loss.py     scale invariant log loss
  data.py     synthetic image and depth map generator
  train.py    short training loop and a runnable demo
tests/        pytest behaviour checks
```

## Synthetic data

Each depth map is a sloped background plane with a few random Gaussian blobs
added on top, kept strictly positive. The matching image is a deterministic
shading of that depth map across three channels with a little noise, so there is
a genuine appearance to depth signal that the small network can fit.

## Running

Install the requirements and run the tests:

```
pip install -r requirements.txt
python -m pytest tests/ -q
```

Run the training demo on one fixed synthetic batch:

```
python -m src.train
```

On a sample run the loss fell from about 0.26 at the first step to about 0.0007
after 100 steps, which shows the model fitting the relative depth structure of
the batch. Your numbers will vary with the seed and step count.

## Tests

The suite checks three behaviours.

1. The forward pass returns a depth map whose shape matches the target, for both
   square and non square inputs.
2. The scale invariant loss is unchanged when the prediction is multiplied by any
   positive scalar, and it is essentially zero when the prediction equals the
   target up to a constant factor.
3. A short training run on a fixed batch reduces the loss well below its starting
   value, and a single optimisation step actually moves the parameters.

## Reference

David Eigen, Christian Puhrsch and Rob Fergus, "Depth Map Prediction from a
Single Image using a Multi Scale Deep Network", NeurIPS 2014.
