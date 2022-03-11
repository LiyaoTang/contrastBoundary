
## Installation

This code has been tested with Python 3.7, Tensorflow 1.14, CUDA 10.0 and cuDNN 7.4.1 on Ubuntu 16.04.

- clone the repo.

        git clone https://github.com/LiyaoTang/contrastBoundary
        cd contrastBoundary

- Setup python environment.

        conda create -n cbl python=3.7
        source activate cbl

- Follow [Tensorflow installation procedure](https://www.tensorflow.org/install), then install other dependencies.

        pip install -r py_requirements.txt


- Compile the customized tf ops.

        bash compile_ops.sh

You should now be able to run the contrastive boundary learning with different baselines.

## Dataset Preparation

### S3DIS
S3DIS dataset can be found [here](https://goo.gl/forms/4SoGp4KtH1jfRqEj2). Download the files named "Stanford3dDataset_v1.2.zip". Uncompress the folder and move it to `/Data/s3dis/Stanford3dDataset_v1.2`.

### ScanNet

ScanNet dataset can be downloaded following the instruction [here](http://kaldir.vc.in.tum.de/scannet_benchmark/documentation) in using its download script. <br>
We need to download four types of files: `_vh_clean_2.ply`, `_vh_clean_2.0.010000.segs.json` and the `.aggregation.json`. We also need its meta-data `scannetv2-labels.combined.tsv`. <br>
Then, move the folder to be under `Data/scannet`.

### Semantic3D

Incoming

### NPM3D

Incoming

## Training

Simply run the following script to start the training:

        python main.py -c config.[dataset].[model_name] --mode train

The config is provided under `config/`, one config file for each dataset. Check the config file as well as the `main.py` for more options to play around.

For example, to train on S3DIS dataset with ConvNet baseline, run the following command:

        python main.py -c config.s3dis.conv_0


## Testing
Simply provide the training command with `mode=test` and the `path` for finding saved model, i.e.

        python main.py -c config.[dataset].[model_name] --mode test --model [path]

For example, to test on S3DIS dataset with the [pretrained model](https://drive.google.com/drive/folders/1_ppwnrAu6VRqENTPWPt-3KFqCCTtfsFC?usp=sharing):

        python main.py -c config.s3dis.conv_0 --mode test --model results/s3dis/conv_multi-Ua-concat-latent_contrast-Ua-softnn-latent-label-l2-w.1/Log_pretrain

For more specific usage, please consult the `main.py`.


## Boundary Evaluation Protocal
We provide the boundary evaluation protocal that takes h5 file, which needs to use python terminal, for example:

        >>> from utils.tester import ModelTester
        >>> from config.s3dis import default as cfg
        >>> tester = ModelTester(cfg)
        >>> tester.solve_extra_ops_from_file(h5_path, extra_ops='boundary')

The above command evaluate the B-IoU with default setting. For more specific usage and other analytical report, please consult the `utils/tester.py`.

Note that, the h5 file provided by `h5_path` needs to set these two attributes: `split` (e.g. `validation`) and `dataset` (e.g. `S3DIS`).

