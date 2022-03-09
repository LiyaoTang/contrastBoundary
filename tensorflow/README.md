
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

### Pretrained Model

Coming soon.


## S3DIS
S3DIS dataset can be found [here](https://goo.gl/forms/4SoGp4KtH1jfRqEj2). Download the files named "Stanford3dDataset_v1.2.zip". Uncompress the folder and move it to `/Data/s3dis/Stanford3dDataset_v1.2`.

### Training

Simply run the following script to start the training on ConvNet baseline:

        python training_S3DIS.py -c config.s3dis.conv_0

The config is provided in `config/s3dis.py`. Check the config file for more config options to play around.

## ScanNet



## Semantic3D

Incoming

## NPM3D

Incoming

### Test the trained model




## Boundary Evaluation Protocal
Coming soon.