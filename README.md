# Patch AutoAugment
Learning the optimal augmentation policies for different regions of an image and achieving the joint optimal on the whole image.

## Introduction
Patch AutoAugment implementation in PyTorch.
The paper can be found [here](https://arxiv.org/abs/2103.11099). The code is coming soon.

<patchlevel-automatic-data-augmentation src="https://github.com/LinShiqi047/PatchAutoAugment/blob/main/figure/imagelevel_v.s_patchlevel.jpg" width="100px">


## Preparation
This project is run on GPU (NVIDIA 2TX 2080Ti).
We conduct experiments under python 3.8, pytorch 1.6.0, cuda 10.1 and cudnn7. You can download dockerfile.

## Cite Us
Please cite us if you find this work helps.
```
@article{lin2021patch,
  title={Patch AutoAugment},
  author={Lin, Shiqi and Yu, Tao and Feng, Ruoyu and Chen, Zhibo},
  journal={arXiv preprint arXiv:2103.11099},
  year={2021}
}
```
