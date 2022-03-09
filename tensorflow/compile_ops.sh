#!/bin/bash
green_start="\033[32m"
green_end="\033[0m"
set -e # exit as soon as any error occur

function info() {
    echo -e "$green_start$1$green_end"
}

info "\ncompiling cpp_wrappers\n"
cd ops/cpp_wrappers
bash compile_wrappers.sh

info "\ncompiling tf_custom_ops\n"
cd ../tf_custom_ops
bash compile_op.sh

info "\ncompiling nearest_neighbors\n"
cd ../nearest_neighbors
bash compile_op.sh

info "\nfinished\n"
cd ../..
