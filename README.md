# This is the non-official implementation for Non-local Neural Networks (Pytorch Version)

## Introduction

Test baseline: resnet-50 experiment on Kinetics dataset.(no non-local block yet)

## Experiment Records

- Kinetics dataset prepared
- Model prepared
- Model parameters converter prepared (caffe2 to pytorch)

## Problems

The test result is low, check:
- Model parameter loading. (OK)
- Kinetics dataset RGB range [-1,1]. (OK)

Well, this confuses me, I still can't figure out what is wrong except for parameter loading and RGB range of Kinetics. The official caffe2 code hurts my brain at sometime.
If anyone could offer some help, I'd be grateful. : )

## Ref:
### [https://github.com/AlexHex7/Non-local_pytorch](https://github.com/AlexHex7/Non-local_pytorch)
### [https://github.com/facebookresearch/video-nonlocal-net](https://github.com/facebookresearch/video-nonlocal-net)
