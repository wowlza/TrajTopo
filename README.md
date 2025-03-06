<div align="center">

# Enhancing Lane Segment Perception and Topology Reasoning with Crowdsourcing Trajectory Priors

[![arXiv](https://img.shields.io/badge/arXiv-2312.16108-479ee2.svg)](https://arxiv.org/abs/2411.17161)


</div>


## News
- **`[2024/11]`** TrajTopo [paper](https://arxiv.org/abs/2411.17161) is available on arXiv. Code is also released!

---

![method](figs/overview.png "Pipeline of TrajTopo")

<div align="center">
<b>Overall pipeline of LaneSegNet</b>
</div>

## Table of Contents

- [Installation](#installation)
- [Prepare Dataset](#prepare-dataset)
- [Train and Evaluate](#train-and-evaluate)
- [License and Citation](#license-and-citation)


## Prerequisites

- Linux
- Python 3.8.x
- NVIDIA GPU + CUDA 11.1
- PyTorch 1.9.1

## Installation

We recommend using [conda](https://docs.conda.io/en/latest/miniconda.html) to run the code.
```bash
conda create -n lanesegnet python=3.8 -y
conda activate lanesegnet

# (optional) If you have CUDA installed on your computer, skip this step.
conda install cudatoolkit=11.1.1 -c conda-forge

pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

Install mm-series packages.
```bash
pip install mmcv-full==1.5.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmdet==2.26.0
pip install mmsegmentation==0.29.1
pip install mmdet3d==1.0.0rc6
```

Install other required packages.
```bash
pip install -r requirements.txt
```

## Prepare Dataset

Following [OpenLane-V2 repo](https://github.com/OpenDriveLab/OpenLane-V2/blob/v2.1.0/data) to download the **Image** and the **Map Element Bucket** data. Run the following script to collect data for this repo. 

## Train and Evaluate



## License and Citation
All assets and code are under the [Apache 2.0 license](./LICENSE) unless specified otherwise.

If this work is helpful for your research, please consider citing the following BibTeX entry.

``` bibtex
@article{jia2024enhancing,
  title={Enhancing Lane Segment Perception and Topology Reasoning with Crowdsourcing Trajectory Priors},
  author={Jia, Peijin and Luo, Ziang and Wen, Tuopu and Yang, Mengmeng and Jiang, Kun and Cui, Le and Yang, Diange},
  journal={arXiv preprint arXiv:2411.17161},
  year={2024}
}

```

