# HighMMT

HighMMT is a general-purpose model for high-modality (large number of modalities beyond the prototypical language, visual, and acoustic modalities) and partially-observable (across many tasks, where each task is defined only over a small subset of all modalities we are interested in modeling) scenarios.

HighMMT uses multitask learning with shared unimodal and multimodal layers to enable stable parameter counts (addressing scalability) and cross-modal transfer learning to enable information sharing across modalities and tasks (addressing partial observability).

The same HighMMT model (architecture and parameters) is able to simultaneously encode joint representations between different subsets spanning images, text, audio, sets, time-series, and graphs.



Usage:

Data Download

This repo is built on top of the MultiBench repository, so to download the dataset, follow the same instructions as https://github.com/pliang279/MultiBench.git

Easy setting experiment code

TODO: Xiang

Medium setting experiment code:

TODO: Shentong

Hard setting experiment code:

To run multitask training on 1/2/3/4 tasks, use singletask.py/twomultitask.py/threemultitask.py/fourmultitask.py respectively in private_test_scripts/perceivers folder


