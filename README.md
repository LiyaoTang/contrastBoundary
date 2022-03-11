# Contrastive Boundary Learning for Point Cloud Segmentation (CVPR 2022)
![image info](./imgs/cbl-full.png)

This is the implementation of our CVPR 2022 paper: <br>
**Contrastive Boundary Learning for Point Cloud Segmentation** [[arXiv]()]

If you find our work useful in your research, please consider citing:

```
@misc{tang2022contrastive,
    title={Contrastive Boundary Learning for Point Cloud Segmentation},
    author={Liyao Tang and Yibing Zhan and Zhe Chen and Baosheng Yu and Dacheng Tao},
    year={2022},
    eprint={2203.05272},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Setup
For point-transformer baseline, please follow [pytorch/README](https://github.com/LiyaoTang/contrastBoundary/blob/master/pytorch/README.md).

For ConvNet and other baselines, please follow [tensorflow/README](https://github.com/LiyaoTang/contrastBoundary/blob/master/tensorflow/README.md).

## Pre-trained models
Pretrained models can be accessed [here](https://drive.google.com/drive/folders/1_ppwnrAu6VRqENTPWPt-3KFqCCTtfsFC?usp=sharing). Choose the desired baseline and unzip into the corresponding code directory (tensorflow/pytorch) and follow the README their for testing instruction.

## Qualitative results
![image info](./imgs/cbl-compare.png)

## Acknowledgement
Codes are built based on a series of previous works, including: <br>
[KPConv](https://github.com/HuguesTHOMAS/KPConv), <br>
[RandLA-Net](https://github.com/QingyongHu/RandLA-Net), <br>
[CloserLook3D](https://github.com/zeliu98/CloserLook3D), <br>
[Point-Transformer](https://github.com/POSTECH-CVLab/point-transformer). <br>
Thanks for their excellent work.


## License
This repo is licensed under the terms of the MIT license (see LICENSE file for details).

