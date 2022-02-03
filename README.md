TorchAudio: Building Blocks for Audio and Speech Processing
===========================================================

This repository contains the code of the experiments in the paper

[TorchAudio: Building Blocks for Audio and Speech Processing](https://arxiv.org/abs/2110.15018)

Authors: [Yao-Yuan Yang](http://yyyang.me), Moto Hira, Zhaoheng Ni, Anjali Chourdia, Artyom Astafurov, Caroline Chen, Ching-Feng Yeh, Christian Puhrsch, David Pollack, Dmitriy Genzel, Donny Greenberg, Edward Z. Yang, Jason Lian, Jay Mahadeokar, Jeff Hwang, Ji Chen, Peter Goldsborough, Prabhat Roy, Sean Narenthiran, Shinji Watanabe, Soumith Chintala, Vincent Quenneville-Bélair, Yangyang Shi

# Abstract

This document describes version 0.10 of torchaudio: building blocks for machine learning applications in the audio and speech processing domain. The objective of torchaudio is to accelerate the development and deployment of machine learning applications for researchers and engineers by providing off-the-shelf building blocks. The building blocks are designed to be GPU-compatible, automatically differentiable, and production-ready. torchaudio can be easily installed from Python Package Index repository and the source code is publicly available under a BSD-2-Clause License (as of September 2021) at [https://github.com/pytorch/audio](https://github.com/pytorch/audio). In this document, we provide an overview of the design principles, functionalities, and benchmarks of torchaudio. We also benchmark our implementation of several audio and speech operations and models. We verify through the benchmarks that our implementations of various operations and models are valid and perform similarly to other publicly available implementations.

---

# Installation

```bash
pip install numpy scipy espnet tqdm librosa
```

Install SoX
```bash
wget https://netactuate.dl.sourceforge.net/project/sox/sox/14.4.2/sox-14.4.2.tar.gz

tar -xf sox-14.4.2.tar.gz
cd sox-14.4.2
./configure
make
```

To run benchmark for filters, you need to install sox and run
```bash
PATH=$HOME/torchaudio-benchmark/sox-14.4.2/src:$PATH python filters.py
```


Install python-pesq

```bash
git clone https://github.com/ludlows/python-pesq.git
cd python-pesq
pip install .
```


Nvidia Tacotron2

```bash
pip install numpy scipy librosa unidecode inflect librosa
```

Citation
--------

If you find this useful, please cite as:

```bibtex
@article{yang2021torchaudio,
  title={TorchAudio: Building Blocks for Audio and Speech Processing},
  author={Yao-Yuan Yang and Moto Hira and Zhaoheng Ni and Anjali Chourdia and Artyom Astafurov and Caroline Chen and Ching-Feng Yeh and Christian Puhrsch and David Pollack and Dmitriy Genzel and Donny Greenberg and Edward Z. Yang and Jason Lian and Jay Mahadeokar and Jeff Hwang and Ji Chen and Peter Goldsborough and Prabhat Roy and Sean Narenthiran and Shinji Watanabe and Soumith Chintala and Vincent Quenneville-Bélair and Yangyang Shi},
  journal={arXiv preprint arXiv:2110.15018},
  year={2021}
}
```
