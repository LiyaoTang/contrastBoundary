#!/bin/sh

echo "[PT INFO] ============================== Installing cuda operations..."
cd lib/pointops
python setup.py install
cd ../..
echo "[PT INFO] ============================== Done !"
