# HighMMT

HighMMT is a general-purpose model for high-modality (large number of modalities beyond the prototypical language, visual, and acoustic modalities) and partially-observable (across many tasks, where each task is defined only over a small subset of all modalities we are interested in modeling) scenarios.

HighMMT uses multitask learning with shared unimodal and multimodal layers to enable stable parameter counts (addressing scalability) and cross-modal transfer learning to enable information sharing across modalities and tasks (addressing partial observability).

The same HighMMT model (architecture and parameters) is able to simultaneously encode joint representations between different subsets spanning images, text, audio, sets, time-series, and graphs.
