# HighMMT

HighMMT is a general-purpose model for high-modality (large number of modalities beyond the prototypical language, visual, and acoustic modalities) and partially-observable (across many tasks, where each task is defined only over a small subset of all modalities we are interested in modeling) scenarios.

HighMMT uses multitask learning with shared unimodal and multimodal layers to enable stable parameter counts (addressing scalability) and cross-modal transfer learning to enable information sharing across modalities and tasks (addressing partial observability).

The same HighMMT model (architecture and parameters) is able to simultaneously encode joint representations between different subsets spanning images, text, audio, sets, time-series, and graphs.

## Paper

[**High-Modality Multimodal Transformer: Quantifying Modality \& Interaction Heterogeneity for High-Modality Representation Learning**](https://openreview.net/forum?id=ttzypy3kT7)<br>
Paul Pu Liang, Yiwei Lyu, Xiang Fan, Shentong Mo, Dani Yogatama, Louis-Philippe Morency, Ruslan Salakhutdinov<br>
TMLR 2022.

If you find this repository useful, please cite our paper:
```
@article{liang2022high,
  title={High-Modality Multimodal Transformer: Quantifying Modality \& Interaction Heterogeneity for High-Modality Representation Learning},
  author={Liang, Paul Pu and Lyu, Yiwei and Fan, Xiang and Tsaw, Jeffrey and Liu, Yudong and Mo, Shentong and Yogatama, Dani and Morency, Louis-Philippe and Salakhutdinov, Russ},
  journal={Transactions on Machine Learning Research},
  year={2022}
}
```

## Contributors

Correspondence to: 
  - [Paul Pu Liang](http://www.cs.cmu.edu/~pliang/) (pliang@cs.cmu.edu)
  - [Yiwei Lyu](https://github.com/lvyiwei1) (ylyu1@andrew.cmu.edu)
  - [Xiang Fan](https://github.com/sfanxiang) (xiangfan@cmu.edu)
  - [Jeffrey Tsaw](https://github.com/jeffreytsaw) (jtsaw@andrew.cmu.edu)
  - Yudong Liu (yudongl@andrew.cmu.edu)
  - [Shentong Mo](https://scholar.google.com/citations?user=6aYncPAAAAAJ&hl=en) (shentonm@andrew.cmu.edu)
  - [Dani Yogatama](https://dyogatama.github.io/)
  - [Louis-Philippe Morency](https://www.cs.cmu.edu/~morency/)
  - [Ruslan Salakhutdinov](https://www.cs.cmu.edu/~rsalakhu/)

## Usage
### Environment Setup Using Conda
```
conda env create -f env_HighMMT.yml
```
###

### Quick Start
The instructions for running the code and data retreival can be found after typing
```
./run.sh help
```
You can also find detailed instructions below
###

### Data Download
three datasets: robotics, enrico and RTFM can be setup directly using script ./download_datasets.sh
Run 
```
./download_datasets.sh help
```
for instructions
To setup each dataset, run "./download_datasets.sh <datasetname>"
For example
```
./download_datasets.sh robotics
```
downloads the robotics dataset to the directory datasets/robotics
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

To run multitask training on 1/2/3/4 tasks, please run
```
python private_test_scripts/perceivers/singletask.py
python private_test_scripts/perceivers/twomultitask.py
python private_test_scripts/perceivers/threemultitask.py
python private_test_scripts/perceivers/fourmultitask.py
```

### Parameter Sharing Experiments
To run the parameter sharing experiments, please run 
```
python private_test_scripts/perceivers/shared_fourmulti.py
```

A baseline can be trained as a starting point for finetuning by running the fourmultitask.py file like described above. You can specify the baseline in shared_fourmulti.py. 

Parameter groupings can also be specified in the shared_fourmulti.py file.

### Heterogeneity Matrix 

To run get the heterogeneity matrix between individual modalitiesa and pairs of modalities, please run
```
python private_test_scripts/perceivers/tasksim.py
```
