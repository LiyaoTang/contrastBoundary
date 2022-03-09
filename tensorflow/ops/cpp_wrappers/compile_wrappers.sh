#!/bin/bash

# Compile cpp subsampling
cd cpp_subsampling
rm -r build
rm *.so
python3 setup.py build_ext --inplace
cd ..
