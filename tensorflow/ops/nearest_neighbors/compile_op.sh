#!/bin/bash

rm -r build
rm -r lib
rm knn.cpp
python setup.py install --home="."
