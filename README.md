# Efficient Arbitrary-Scale Image Super-Resolution via Functional Tensor Decomposition

## Introduction
This repository contains the official PyTorch implementation for the paper "Efficient Arbitrary-Scale Image Super-Resolution via Functional Tensor Decomposition".
<div align="center">
  <img src="figs/main.png" alt="Framework" />
  <br>
  Framework of our method
</div>

## Abstract
Existing arbitrary-scale super-resolution (ASSR) methods suffer from quadratic computational complexity w.r.t. image scale due to the reliance on multi-layer perceptrons (MLPs) to query dense spatial coordinate matrices. The inefficiency becomes particularly pronounced when extending to high-dimensional imaging modalities. To address these limitations, we propose a novel functional tensor decomposition (FTD) framework that fundamentally reconfigures the computational paradigm for ASSR. Specifically, we propose 1) a separation mechanism that employs distinct MLPs to query separable spatial coordinate vectors, substantially reducing decoder MLP invocations, and 2) functional tensor Tucker or CP decompositions for efficient factor matrix integration. The FTD framework delivers three key advantages: 1) Superior scalability to high-dimensional imaging modalities, such as hyperspectral images (HSIs), by virtue of the FTD design; 2) Significantly enhanced inference speed across scales; 3) Faster convergence towards a desired training model. Extensive experiments validate FTD's exceptional performance in HSI joint spatial-spectral ASSR, achieving up to 90.04\% reduction in inference time and substantial performance improvements. For conventional image ASSR, our method improves both inference speed and convergence efficiency, achieving up to 88.82\% inference time reduction and superior few-shot generalization capabilities due to faster convergence.

## Environments
- python >= 3.10
- pytorch >=2.4.0

## Quick Start
1. Download the pretrained models and place the files in `./checkpoints`.
2. Download from the [DIV2K official website](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and place the files in the `./load` folder. Both DIV2K_train_HR and DIV2K_valid_HR are needed.
3. Test on DIV2K:
`python test_lrtfr.py --config configs\test\test-div2k-2.yaml --model checkpoints/checkpoints\rdn_tucker-liif.pth`
1. Generate the SR result for a single image:
`python demo_lrtfr.py`


## Training & Testing
### 
1. Download the pretrained encoders from ... and place the files in `./weights`.
2. Train your model: `python ./train_liif_lrtfr.py --config ./configs/train-liiflrtfr/train_edsr-baseline-liiflrtfr-tucker.yaml`.

2. Test your model: `python ./test_lrtfr.py --config ./configstest\test-div2k-2.yaml --model .\save\train_edsr-baseline-liiflrtfr-tucker\epoch-last.pth`
