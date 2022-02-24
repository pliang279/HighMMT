# HighMMT

HighMMT is a general-purpose model for high-modality (large number of modalities beyond the prototypical language, visual, and acoustic modalities) and partially-observable (across many tasks, where each task is defined only over a small subset of all modalities we are interested in modeling) scenarios.

HighMMT uses multitask learning with shared unimodal and multimodal layers to enable stable parameter counts (addressing scalability) and cross-modal transfer learning to enable information sharing across modalities and tasks (addressing partial observability).

The same HighMMT model (architecture and parameters) is able to simultaneously encode joint representations between different subsets spanning images, text, audio, sets, time-series, and graphs.

## Contributors

Correspondence to: 
  - [Paul Pu Liang](http://www.cs.cmu.edu/~pliang/) (pliang@cs.cmu.edu)
  - [Yiwei Lyu](https://github.com/lvyiwei1) (ylyu1@andrew.cmu.edu)
  - [Xiang Fan](https://github.com/sfanxiang) (xiangfan@cmu.edu)
  - [Shentong Mo](https://scholar.google.com/citations?user=6aYncPAAAAAJ&hl=en) 
  - [Dani Yogatama](https://dyogatama.github.io/)
  - [Louis-Philippe Morency](https://www.cs.cmu.edu/~morency/)
  - [Ruslan Salakhutdinov](https://www.cs.cmu.edu/~rsalakhu/)



## Usage

### Data Download

This repo is built on top of the MultiBench repository, so to download the dataset, follow the same instructions as https://github.com/pliang279/MultiBench.git

### Easy setting experiment code

From the root of this repo, run
```sh
python private_test_scripts/perceivers/roboticstasks.py model.pt
```
The model will be saved to `model.pt`.

### Medium setting experiment code

To run medium tasks, please run
```
python private_test_scripts/perceivers/medium_tasks.py
```

### Hard setting experiment code

To run multitask training on 1/2/3/4 tasks, use singletask.py/twomultitask.py/threemultitask.py/fourmultitask.py respectively in private_test_scripts/perceivers folder

