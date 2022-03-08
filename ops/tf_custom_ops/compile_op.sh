#!/bin/bash

rm *.so

# Get TF variables
TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

# check & link libtensorflow_framework
python3 -c "
import os, glob; l='libtensorflow_framework.so'; fs=glob.glob(f'$TF_LIB/{l}*')
if all(f.split('/')[-1]!=l for f in fs):
    if fs: src=fs[0]; dst=f'$TF_LIB/{l}'; os.system(f'ln -s {src} {dst}'); print(f'\033[94mlinking {src}->{dst}\033[0m')
    else: os.system(f'll $TF_LIB'); raise OSError(f'\033[91mno {l} in $TF_LIB\033[0m')
else: print(f'\033[96mfound in $TF_LIB: {l}\033[0m')
"

TF_IS_OLD=$(python -c 'import tensorflow as tf; print(int(float(".".join(tf.__version__.split(".")[:2])) < 1.15))')
GCC_NEW=$(expr `gcc -dumpversion | cut -f1 -d.` \> 4)
# USE_CXX11_ABI=$TF_IS_OLD  # avoid undefined behavior due to tf cpp API change
USE_CXX11_ABI=$GCC_NEW    # old gcc not supporting cxx11

echo TF_IS_OLD=$TF_IS_OLD, GCC_NEW=$GCC_NEW, USE_CXX11_ABI=$USE_CXX11_ABI

# Neighbors op
g++ -std=c++11 -shared tf_neighbors/tf_neighbors.cpp tf_neighbors/neighbors/neighbors.cpp cpp_utils/cloud/cloud.cpp -o tf_neighbors.so -fPIC -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=$USE_CXX11_ABI
g++ -std=c++11 -shared tf_neighbors/tf_batch_neighbors.cpp tf_neighbors/neighbors/neighbors.cpp cpp_utils/cloud/cloud.cpp -o tf_batch_neighbors.so -fPIC -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=$USE_CXX11_ABI

# Subsampling op
g++ -std=c++11 -shared tf_subsampling/tf_subsampling.cpp tf_subsampling/grid_subsampling/grid_subsampling.cpp cpp_utils/cloud/cloud.cpp -o tf_subsampling.so -fPIC -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=$USE_CXX11_ABI
g++ -std=c++11 -shared tf_subsampling/tf_batch_subsampling.cpp tf_subsampling/grid_subsampling/grid_subsampling.cpp cpp_utils/cloud/cloud.cpp -o tf_batch_subsampling.so -fPIC -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=$USE_CXX11_ABI
