---
title: Diffusion Models Beat GANs on Image Synthesis
authors:
  - Prafulla Dhariwal
  - Alex Nichol
---

## Abstract

We show that diffusion models can achieve image sample quality superior to
the current state-of-the-art generative adversarial networks (GANs). We
achieve this on unconditional image synthesis by finding a better
architecture through a series of ablations. For conditional image synthesis,
we further improve sample quality with classifier guidance: a simple,
compute-efficient method for trading off diversity for sample fidelity using
gradients from a classifier.

## Introduction

Generative adversarial networks (GANs) currently dominate the space of image
generation on most metrics involving sample quality with high resolution
and diversity, despite the theoretical benefits of likelihood-based models
and the wide range of promising architectures explored for diffusion models.
GANs capture less diversity than state-of-the-art likelihood-based models,
and are often difficult to train, collapsing without carefully selected
hyperparameters and regularizers.

## Method

We propose several architectural improvements to the standard diffusion
model backbone, including increasing depth versus width, holding model size
relatively constant, increasing the number of attention heads, using
attention at multiple resolutions rather than a single resolution, using
the BigGAN residual block for upsampling and downsampling, and rescaling
residual connections with a factor of one over the square root of two.

## Results

Our best models achieve an FID of 2.97 on ImageNet 128x128, 4.59 on
ImageNet 256x256, and 7.72 on ImageNet 512x512, and we match BigGAN-deep
even with as few as 25 forward passes per sample, all while maintaining
better coverage of the distribution as measured by recall.

## Limitations

Diffusion models require many forward passes to produce a single sample,
making them substantially slower to sample from than GANs, and classifier
guidance requires training an additional noisy classifier, adding
complexity to the overall pipeline.
