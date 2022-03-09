#!/bin/sh

export PYTHONPATH=./
# eval "$(conda shell.bash hook)"
PYTHON=python

TRAIN_CODE=train.py
TEST_CODE=test.py

dataset=$1
exp_name=$2
exp_dir=exp/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/${dataset}/${exp_name}.yaml

mkdir -p ${model_dir} ${result_dir}
mkdir -p ${result_dir}/last
mkdir -p ${result_dir}/best
cp tool/train.sh tool/${TRAIN_CODE} ${config} tool/test.sh tool/${TEST_CODE} ${exp_dir}


export NCCL_IB_DISABLE=1

now=$(date +"%Y%m%d_%H%M%S")
$PYTHON ${exp_dir}/${TRAIN_CODE} \
  --config=${config} \
  resume "${3}" \
  save_path ${exp_dir} \
  2>&1 | tee ${exp_dir}/train-$now.log


# # testing
# now=$(date +"%Y%m%d_%H%M%S")
# cp ${config} tool/test.sh tool/${TEST_CODE} ${exp_dir}

# cuda=${4}
# if [ "$cuda" == "" ]; then
#   cuda=0
# fi

# #: '
# CUDA_VISIBLE_DEVICES=$cuda $PYTHON -u ${exp_dir}/${TEST_CODE} \
#   --config=${config} \
#   save_folder ${result_dir}/best \
#   model_path ${model_dir}/model_best.pth \
#   2>&1 | tee ${exp_dir}/test_best-$now.log
# #'

# #: '
# CUDA_VISIBLE_DEVICES=$cuda $PYTHON -u ${exp_dir}/${TEST_CODE} \
#   --config=${config} \
#   save_folder ${result_dir}/last \
#   model_path ${model_dir}/model_last.pth \
#   2>&1 | tee ${exp_dir}/test_last-$now.log
# #'
